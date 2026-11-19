#!/usr/bin/env python
"""Normalize every raw clip and measure its motion (TI) and texture (SI).

For each video in <data>/raw/<category>/ (.mp4 .webm .ogv .mov .mkv ...):
  * cut `--duration` seconds starting at `--start` (default: 20% in for clips > 30 s, to skip intros),
  * scale and centre-crop to 1920x1080 (no black bars), resample to 30 fps,
  * encode near-lossless (x264 CRF 10, yuv420p, no audio) to <data>/norm/<category>__<stem>.mp4,
  * compute ITU-T P.910 SI/TI on the normalized clip.
Writes <data>/candidates.csv. Already-normalized clips are reused.

Example:
    python scripts/prepare.py --data $VSPS_DATA --workers 16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsbench.metrics import si_ti  # noqa: E402
from vsbench.videoio import probe, run_ffmpeg  # noqa: E402

EXTS = {".mp4", ".mov", ".mkv", ".webm", ".ogv", ".ogg", ".mpg", ".mpeg", ".avi", ".m4v"}


def normalize(src: Path, dst: Path, start: float, duration: float, w: int, h: int, fps: int) -> None:
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h},fps={fps}"
    run_ffmpeg(["-ss", f"{start}", "-i", str(src), "-t", f"{duration}", "-vf", vf,
                "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", str(dst)])


def auto_start(total: float, clip: float) -> float:
    """Skip intros / title cards: start 20% in for long videos (> 30 s), else at 0."""
    if not total or total <= 30:
        return 0.0
    return round(min(0.2 * total, total - clip - 1.0), 2)


def process(src: Path, category: str, norm_dir: Path, a) -> dict:
    vid = f"{category}__{src.stem}"
    dst = norm_dir / f"{vid}.mp4"
    try:
        info = probe(src)
        start = auto_start(info.duration, a.duration) if a.start == "auto" else float(a.start)
        if info.duration and info.duration < start + a.duration - 0.05:
            return {"id": vid, "error": f"too short ({info.duration:.1f}s)"}
        if not dst.exists():
            tmp = dst.with_suffix(".tmp.mp4")
            normalize(src, tmp, start, a.duration, a.width, a.height, a.fps)
            tmp.replace(dst)
        thumb = norm_dir.parent / "thumbs" / f"{vid}.jpg"
        if not thumb.exists():
            thumb.parent.mkdir(exist_ok=True)
            run_ffmpeg(["-ss", f"{a.duration / 2:.2f}", "-i", str(dst), "-frames:v", "1",
                        "-vf", "scale=384:-2", "-q:v", "4", str(thumb)])
        m = si_ti(dst)
        return {"id": vid, "category": category, "path": str(dst), "raw": str(src),
                "src_width": info.width, "src_height": info.height, "src_fps": round(info.fps, 3), **m}
    except Exception as e:  # keep going; report at the end
        return {"id": vid, "error": " ".join(str(e)[-400:].split())}


def write_contact_sheet(data: Path, rows: list[dict]) -> Path:
    """candidates.html: a thumbnail per clip, grouped by category and sorted by TI, to spot
    off-topic downloads before selection. Click a thumbnail to play the normalized clip."""
    import html
    css = ("body{font-family:system-ui,sans-serif;margin:20px;background:#fafaf8;color:#1d1d1b}"
           "@media(prefers-color-scheme:dark){body{background:#1b1b1a;color:#ecebe6}}"
           ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}"
           "figure{margin:0;font-size:12px}img{width:100%;border-radius:6px}code{user-select:all}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>Candidates</title><style>{css}</style>",
             "<h1>Candidate clips</h1><p>Sorted by TI (motion). Copy the id of any off-topic clip into "
             "<code>exclude.txt</code>, one per line, then re-run select_videos.py.</p>"]
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]
        parts.append(f"<h2>{html.escape(cat)} ({len(rs)})</h2><div class='g'>")
        for r in rs:
            parts.append(f"<figure><a href='norm/{html.escape(r['id'])}.mp4'><img loading='lazy' "
                         f"src='thumbs/{html.escape(r['id'])}.jpg'></a><figcaption><code>{html.escape(r['id'])}</code>"
                         f"<br>TI {r['ti_mean']:.1f} · SI {r['si_mean']:.1f}</figcaption></figure>")
        parts.append("</div>")
    out = data / "candidates.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("VSPS_DATA", "data"))
    ap.add_argument("--start", default="auto",
                    help="seconds to skip, or 'auto' (20%% into videos longer than 30 s)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    a = ap.parse_args()

    data = Path(a.data)
    raw, norm = data / "raw", data / "norm"
    norm.mkdir(parents=True, exist_ok=True)
    jobs = [(p, d.name) for d in sorted(raw.iterdir()) if d.is_dir()
            for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    if not jobs:
        print(f"No clips found under {raw}/<category>/", file=sys.stderr)
        return 1
    print(f"{len(jobs)} raw clips, {a.workers} workers")

    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, p, c, norm, a) for p, c in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            (errors if "error" in r else rows).append(r)
            if "error" in r:
                print(f"[{i}/{len(jobs)}] SKIP {r['id']}: {r['error']}")
            else:
                print(f"[{i}/{len(jobs)}] {r['id']}: TI={r['ti_mean']:.1f} (max {r['ti']:.1f})  SI={r['si_mean']:.1f}")

    rows.sort(key=lambda r: (r["category"], r["ti_mean"]))
    out = data / "candidates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sheet = write_contact_sheet(data, rows)
    print(f"\nwrote {out}: {len(rows)} clips ({len(errors)} skipped)")
    print(f"review {sheet}; put ids of off-topic clips (one per line) in {data / 'exclude.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
