#!/usr/bin/env python
"""Normalize every raw clip and measure its motion (TI) and texture (SI).

For each <data>/raw/<category>/*.mp4 (and .mov/.mkv/.webm):
  * cut `--duration` seconds starting at `--start`,
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

EXTS = {".mp4", ".mov", ".mkv", ".webm"}


def normalize(src: Path, dst: Path, start: float, duration: float, w: int, h: int, fps: int) -> None:
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h},fps={fps}"
    run_ffmpeg(["-ss", f"{start}", "-i", str(src), "-t", f"{duration}", "-vf", vf,
                "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", str(dst)])


def process(src: Path, category: str, norm_dir: Path, a) -> dict:
    vid = f"{category}__{src.stem}"
    dst = norm_dir / f"{vid}.mp4"
    try:
        info = probe(src)
        if info.duration and info.duration < a.start + a.duration - 0.05:
            return {"id": vid, "error": f"too short ({info.duration:.1f}s)"}
        if not dst.exists():
            tmp = dst.with_suffix(".tmp.mp4")
            normalize(src, tmp, a.start, a.duration, a.width, a.height, a.fps)
            tmp.replace(dst)
        m = si_ti(dst)
        return {"id": vid, "category": category, "path": str(dst), "raw": str(src),
                "src_width": info.width, "src_height": info.height, "src_fps": round(info.fps, 3), **m}
    except Exception as e:  # keep going; report at the end
        return {"id": vid, "error": str(e)[:300]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("VSPS_DATA", "data"))
    ap.add_argument("--start", type=float, default=0.0)
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
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}: {len(rows)} clips ({len(errors)} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
