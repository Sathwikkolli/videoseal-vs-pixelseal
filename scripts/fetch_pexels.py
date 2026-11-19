#!/usr/bin/env python
"""Download candidate clips from Pexels for each content category.

Needs a free API key from https://www.pexels.com/api/ in the PEXELS_API_KEY env var.
Downloads ~N landscape clips per category (>= 10 s, ~1080p) into
    <data>/raw/<category>/pexels_<id>.mp4
and appends attribution rows to <data>/raw/sources.csv (Pexels License: free to use,
attribution appreciated). You can also drop your own .mp4 files into raw/<category>/;
prepare.py picks up everything in those folders.

Example:
    python scripts/fetch_pexels.py --data $VSPS_DATA --per-category 30
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.pexels.com/videos/search"

QUERIES = {
    # low motion: static camera, 1-2 people talking
    "podcast": ["podcast studio", "podcast interview", "two people talking microphone",
                "talking head interview", "video call person talking", "news anchor desk"],
    # mid motion: speaker in front of a moving audience, slow camera moves
    "gathering": ["public speech crowd", "speaker on stage audience", "political rally",
                  "conference speaker audience", "protest speech megaphone", "outdoor event speaker"],
    # high motion: fast action, flashing lights, handheld or fast camera
    "high_motion": ["football match", "concert crowd lights", "street dance", "basketball game",
                    "skateboarding", "fireworks", "running crowd marathon", "motocross"],
}

FIELDS = ["category", "file", "pexels_id", "query", "page_url", "file_url", "author", "author_url",
          "width", "height", "fps", "duration", "license"]


def pick_file(video: dict, target_w: int = 1920) -> dict | None:
    files = [f for f in video.get("video_files", [])
             if f.get("width") and f.get("height") and f["width"] >= f["height"]
             and f.get("file_type") == "video/mp4"]
    if not files:
        return None
    big = [f for f in files if f["width"] >= target_w]
    pool = big or files
    return min(pool, key=lambda f: abs(f["width"] - target_w))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("VSPS_DATA", "data"))
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=10.0)
    ap.add_argument("--max-duration", type=float, default=90.0, help="skip very long files to save disk")
    ap.add_argument("--categories", nargs="+", default=list(QUERIES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        print("Set PEXELS_API_KEY (free at https://www.pexels.com/api/)", file=sys.stderr)
        return 2
    session = requests.Session()
    session.headers["Authorization"] = key

    raw = Path(args.data) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = raw / "sources.csv"
    seen: set[str] = set()
    if sources.exists():
        with open(sources, newline="") as f:
            seen = {r["pexels_id"] for r in csv.DictReader(f)}
    new_file = not sources.exists()
    out = open(sources, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    for cat in args.categories:
        cat_dir = raw / cat
        cat_dir.mkdir(exist_ok=True)
        have = len(list(cat_dir.glob("*.mp4")))
        print(f"== {cat}: {have} already present, target {args.per_category}")
        queries = QUERIES[cat]
        page = 1
        while have < args.per_category and page <= 10:
            progressed = False
            for q in queries:
                if have >= args.per_category:
                    break
                r = session.get(API, params={"query": q, "orientation": "landscape", "size": "medium",
                                             "per_page": 15, "page": page}, timeout=30)
                if r.status_code == 429:
                    print("rate limited, sleeping 60 s")
                    time.sleep(60)
                    continue
                r.raise_for_status()
                for v in r.json().get("videos", []):
                    vid = str(v["id"])
                    dur = float(v.get("duration") or 0)
                    if vid in seen or not (args.min_duration <= dur <= args.max_duration):
                        continue
                    f = pick_file(v)
                    if f is None or f["width"] < 1280:
                        continue
                    dst = cat_dir / f"pexels_{vid}.mp4"
                    print(f"  {q!r}: {vid} {f['width']}x{f['height']} {dur:.0f}s")
                    if not args.dry_run:
                        with session.get(f["link"], stream=True, timeout=120) as dl:
                            dl.raise_for_status()
                            with open(dst, "wb") as fh:
                                for chunk in dl.iter_content(1 << 20):
                                    fh.write(chunk)
                        writer.writerow({
                            "category": cat, "file": str(dst.relative_to(raw)), "pexels_id": vid, "query": q,
                            "page_url": v.get("url"), "file_url": f["link"],
                            "author": v.get("user", {}).get("name"), "author_url": v.get("user", {}).get("url"),
                            "width": f["width"], "height": f["height"], "fps": f.get("fps"),
                            "duration": dur, "license": "Pexels License",
                        })
                        out.flush()
                    seen.add(vid)
                    have += 1
                    progressed = True
                    if have >= args.per_category:
                        break
            page += 1
            if not progressed:
                continue
        print(f"   {cat}: {have} clips")
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
