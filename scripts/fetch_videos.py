#!/usr/bin/env python
"""Download candidate clips per content category from free-license stock sources.

Sources (pick with --source; several can be combined, they fill the same quota):
  commons  Wikimedia Commons. No key needed. CC / public-domain licenses (recorded per file).
  pixabay  Pixabay. Free key after signup: https://pixabay.com/api/docs/  -> PIXABAY_API_KEY
  pexels   Pexels. Needs an existing key in PEXELS_API_KEY (new keys are currently paused).

Clips are saved to <data>/raw/<category>/<source>_<id>.<ext> and attribution rows
(author, page URL, license) are appended to <data>/raw/sources.csv. Anything you drop
into raw/<category>/ yourself is also picked up by prepare.py.

Categories: podcast (low motion), gathering (mid), high_motion (high).

Examples:
    python scripts/fetch_videos.py --data $VSPS_DATA --source commons --per-category 30
    PIXABAY_API_KEY=... python scripts/fetch_videos.py --data $VSPS_DATA --source pixabay commons
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

QUERIES = {
    "podcast": ["podcast", "podcast interview", "interview studio", "talking head interview",
                "video call", "news anchor", "vlog talking"],
    "gathering": ["speech crowd", "speaker audience", "political rally", "conference speaker",
                  "protest speech", "public speech", "town hall meeting"],
    "high_motion": ["football match", "concert crowd lights", "street dance", "basketball game",
                    "skateboarding", "fireworks", "marathon running", "motocross"],
}

# Commons full-text search matches anything on the file page (descriptions, categories),
# so search titles only, with words that tend to appear in titles of that kind of video.
COMMONS_QUERIES = {
    "podcast": ["podcast", "interview", "vlog", "webinar", "talk show", "Q&A", "video message"],
    "gathering": ["rally", "speech", "protest", "town hall", "campaign event", "demonstration", "keynote"],
    "high_motion": ["football", "concert", "dance", "fireworks", "skateboarding", "basketball", "race", "parkour"],
}

FIELDS = ["category", "file", "source", "source_id", "query", "page_url", "file_url", "author",
          "width", "height", "duration", "license"]

UA = os.environ.get("VSPS_UA", "vsbench-video-fetcher/0.1 (research benchmark; python-requests)")


class Candidate(dict):
    """source, source_id, query, page_url, file_url, author, width, height, duration, license, ext"""


# ------------------------------------------------------------------ sources

def search_pixabay(session: requests.Session, query: str, page: int, a) -> Iterator[Candidate]:
    key = os.environ["PIXABAY_API_KEY"]
    r = session.get("https://pixabay.com/api/videos/", timeout=30, params={
        "key": key, "q": query, "video_type": "film", "per_page": 50, "page": page, "safesearch": "true"})
    if r.status_code == 400:  # page past the last result
        return
    r.raise_for_status()
    for h in r.json().get("hits", []):
        vids = h.get("videos", {})
        # 'medium' is usually 1920x1080; fall back to 'large' (4K) or 'small' (720p)
        f = next((vids[k] for k in ("medium", "large", "small") if vids.get(k, {}).get("url")
                  and vids[k].get("height", 0) >= a.min_height and vids[k]["width"] >= vids[k]["height"]), None)
        if f is None:
            continue
        yield Candidate(source="pixabay", source_id=str(h["id"]), query=query, page_url=h.get("pageURL"),
                        file_url=f["url"], author=h.get("user"), width=f["width"], height=f["height"],
                        duration=float(h.get("duration") or 0), license="Pixabay Content License", ext=".mp4",
                        bytes=f.get("size") or 0)
    time.sleep(1.0)  # stay far below 100 requests/minute


def search_pexels(session: requests.Session, query: str, page: int, a) -> Iterator[Candidate]:
    r = session.get("https://api.pexels.com/videos/search", timeout=30,
                    headers={"Authorization": os.environ["PEXELS_API_KEY"]},
                    params={"query": query, "orientation": "landscape", "size": "medium", "per_page": 30, "page": page})
    r.raise_for_status()
    for v in r.json().get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("width") and f.get("height")
                 and f["width"] >= f["height"] and f["height"] >= a.min_height and f.get("file_type") == "video/mp4"]
        if not files:
            continue
        f = min(files, key=lambda f: abs(f["height"] - 1080))
        yield Candidate(source="pexels", source_id=str(v["id"]), query=query, page_url=v.get("url"),
                        file_url=f["link"], author=v.get("user", {}).get("name"), width=f["width"], height=f["height"],
                        duration=float(v.get("duration") or 0), license="Pexels License", ext=".mp4", bytes=0)
    time.sleep(1.0)


def _strip_html(s: str | None) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def search_commons(session: requests.Session, query: str, page: int, a) -> Iterator[Candidate]:
    limit = 40
    r = get_with_retry(session, "https://commons.wikimedia.org/w/api.php", timeout=60, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f'filetype:video intitle:"{query}"', "gsrlimit": limit, "gsroffset": (page - 1) * limit,
        "prop": "videoinfo", "viprop": "url|size|mime|extmetadata|derivatives",
        "viextmetadatafilter": "LicenseShortName|Artist"})
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        vi = (p.get("videoinfo") or [{}])[0]
        w, h = int(vi.get("width") or 0), int(vi.get("height") or 0)
        if not w or h < a.min_height or w < h:
            continue
        # original first, then server-side transcodes; prefer the one closest to 1080p
        options = [{"src": vi["url"], "width": w, "height": h, "orig": True}]
        for d in vi.get("derivatives", []):
            if d.get("width") and d.get("height") and "webm" in d.get("type", ""):
                options.append({"src": d["src"], "width": int(d["width"]), "height": int(d["height"]), "orig": False})
        options = [o for o in options if o["height"] >= a.min_height]
        if not options:
            continue
        best = min(options, key=lambda o: (abs(o["height"] - 1080), o["orig"]))
        meta = vi.get("extmetadata", {})
        ext = Path(best["src"].split("?")[0]).suffix.lower() or ".webm"
        if ext not in (".webm", ".ogv", ".mp4", ".mpg", ".mpeg", ".mov"):
            ext = ".webm"
        yield Candidate(source="commons", source_id=str(p["pageid"]), query=query,
                        page_url=vi.get("descriptionurl"), file_url=best["src"],
                        author=_strip_html(meta.get("Artist", {}).get("value"))[:200],
                        width=best["width"], height=best["height"], duration=float(vi.get("duration") or 0),
                        license=meta.get("LicenseShortName", {}).get("value", "see page"), ext=ext,
                        bytes=int(vi.get("size") or 0) if best["orig"] else 0)
    time.sleep(1.0)


SOURCES = {"commons": search_commons, "pixabay": search_pixabay, "pexels": search_pexels}
NEEDS_KEY = {"pixabay": "PIXABAY_API_KEY", "pexels": "PEXELS_API_KEY"}


# ------------------------------------------------------------------ main

def get_with_retry(session: requests.Session, url: str, tries: int = 6, **kw) -> requests.Response:
    """GET that backs off on 429 / 5xx (Wikimedia rate-limits bursts), honouring Retry-After."""
    delay = 5.0
    for i in range(tries):
        r = session.get(url, **kw)
        if r.status_code != 429 and r.status_code < 500:
            return r
        try:
            wait = max(float(r.headers.get("Retry-After") or 0), delay)
        except ValueError:  # Retry-After may be an HTTP date
            wait = delay
        r.close()
        print(f"      HTTP {r.status_code}, waiting {wait:.0f}s ({i + 1}/{tries})")
        time.sleep(wait)
        delay = min(delay * 2, 120)
    r.raise_for_status()
    return r


def download(session: requests.Session, url: str, dst: Path, max_bytes: int) -> bool:
    tmp = dst.with_name(dst.name + ".part")
    n = 0
    with get_with_retry(session, url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                n += len(chunk)
                if n > max_bytes:
                    break
                fh.write(chunk)
    if n > max_bytes or n == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("VSPS_DATA", "data"))
    ap.add_argument("--source", nargs="+", default=["commons"], choices=list(SOURCES))
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--categories", nargs="+", default=list(QUERIES), choices=list(QUERIES))
    ap.add_argument("--min-height", type=int, default=720)
    ap.add_argument("--min-duration", type=float, default=12.0)
    ap.add_argument("--max-duration", type=float, default=240.0,
                    help="skip longer videos (their files get large); prepare.py cuts 10 s anyway")
    ap.add_argument("--max-mb", type=float, default=200.0, help="abort downloads larger than this")
    ap.add_argument("--max-pages", type=int, default=5)
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between downloads (be polite)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    a = ap.parse_args()

    for s in a.source:
        if s in NEEDS_KEY and not os.environ.get(NEEDS_KEY[s]):
            print(f"--source {s} needs {NEEDS_KEY[s]} in the environment", file=sys.stderr)
            return 2
    max_bytes = int(a.max_mb * 1e6)

    session = requests.Session()
    session.headers["User-Agent"] = UA
    raw = Path(a.data) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources_csv = raw / "sources.csv"
    seen: set[tuple[str, str]] = set()
    if sources_csv.exists():
        with open(sources_csv, newline="", encoding="utf-8") as f:
            seen = {(r["source"], r["source_id"]) for r in csv.DictReader(f)}
    new = not sources_csv.exists()
    out = open(sources_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    if new:
        writer.writeheader()

    for cat in a.categories:
        cdir = raw / cat
        cdir.mkdir(exist_ok=True)
        have = sum(1 for p in cdir.iterdir() if p.suffix and not p.name.endswith(".part"))
        print(f"== {cat}: {have} present, target {a.per_category}")
        # spread picks over the queries (and therefore uploaders / scenes): at most
        # `per_query` new clips per query per page, so one search term cannot fill the quota
        per_query = max(2, -(-a.per_category // (len(QUERIES[cat]) * len(a.source))))
        for page in range(1, a.max_pages + 1):
            if have >= a.per_category:
                break
            for src in a.source:
                for q in (COMMONS_QUERIES if src == "commons" else QUERIES)[cat]:
                    if have >= a.per_category:
                        break
                    try:
                        cands = list(SOURCES[src](session, q, page, a))
                    except requests.RequestException as e:
                        print(f"   {src} {q!r} page {page}: {e}")
                        continue
                    taken = 0
                    for c in cands:
                        if have >= a.per_category or taken >= per_query:
                            break
                        if (c["source"], c["source_id"]) in seen:
                            continue
                        if not (a.min_duration <= c["duration"] <= a.max_duration):
                            continue
                        if c.get("bytes") and c["bytes"] > max_bytes:
                            continue
                        dst = cdir / f"{c['source']}_{c['source_id']}{c['ext']}"
                        print(f"   {src:7s} {q!r}: {c['source_id']} {c['width']}x{c['height']} "
                              f"{c['duration']:.0f}s {c['license']}")
                        seen.add((c["source"], c["source_id"]))
                        if a.dry_run:
                            have += 1
                            taken += 1
                            continue
                        try:
                            ok = download(session, c["file_url"], dst, max_bytes)
                        except (requests.RequestException, OSError) as e:
                            print(f"      download failed: {e}")
                            continue
                        finally:
                            time.sleep(a.pause)
                        if not ok:
                            print("      skipped: larger than --max-mb")
                            continue
                        writer.writerow({"category": cat, "file": str(dst.relative_to(raw)).replace(os.sep, "/"),
                                         **{k: c.get(k) for k in FIELDS if k not in ("category", "file")}})
                        out.flush()
                        have += 1
                        taken += 1
        print(f"   {cat}: {have} clips")
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
