#!/usr/bin/env python
"""Run the full benchmark for ONE manifest row (one Slurm array task).

For the clip at --index:
  1. negative control: extract from the unwatermarked clip with every model
  2. for each model: embed (timed) -> quality (PSNR/SSIM/LPIPS/VMAF) -> 10 attacks -> extract
  3. viewing files: previews, side-by-side, amplified-difference videos, PNG snapshots
Writes <out>/results/<id>.json and <out>/videos/<id>/...

Example:
    python scripts/run_one.py --manifest $VSPS_DATA/manifest.csv --index 0 --out $VSPS_DATA/out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from vsbench.attacks import ATTACKS, DEFAULT_ATTACKS, apply_attack  # noqa: E402
from vsbench.metrics import score_message, vmaf  # noqa: E402
from vsbench.models import (embed_video, extract_message, load_model, message_for,  # noqa: E402
                            nbits_of, parse_spec, warmup)
from vsbench.videoio import make_preview, probe, run_ffmpeg  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def side_by_side(paths: list[Path], dst: Path, height: int = 540) -> None:
    """Panes left-to-right in the order given, each scaled to `height`, browser-playable."""
    inputs = sum((["-i", str(p)] for p in paths), [])
    scaled = ";".join(f"[{i}:v]scale=-2:{height}[v{i}]" for i in range(len(paths)))
    stack = "".join(f"[v{i}]" for i in range(len(paths))) + f"hstack=inputs={len(paths)}[out]"
    run_ffmpeg([*inputs, "-filter_complex", f"{scaled};{stack}", "-map", "[out]", "-c:v", "libx264",
                "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["videoseal", "pixelseal"],
                    help='model specs, e.g. videoseal pixelseal "pixelseal@k8"')
    ap.add_argument("--step-size", type=int, default=4, help="watermark propagation step k for all models")
    ap.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS, choices=list(ATTACKS))
    ap.add_argument("--segment", type=int, default=32, help="frames per embed call (multiple of every k)")
    ap.add_argument("--ckpt-dir", default=os.environ.get("VSPS_CKPT", "ckpts"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep-attacked", action="store_true", help="keep attacked videos (large)")
    ap.add_argument("--no-viewer", action="store_true", help="skip previews / side-by-side / diff videos")
    ap.add_argument("--no-vmaf", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if the result JSON exists")
    a = ap.parse_args()

    with open(a.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    row = rows[a.index]
    vid, src = row["id"], Path(row["path"])
    out = Path(a.out)
    res_path = out / "results" / f"{vid}.json"
    if res_path.exists() and not a.force:
        log(f"{res_path} exists, skipping (use --force)")
        return 0
    vdir = out / "videos" / vid
    vdir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("TMPDIR", "/tmp")) / f"vsps_{vid}_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    gpu = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else platform.processor() or "cpu"
    log(f"{vid} tier={row['tier']} {info.width}x{info.height}@{info.fps:.0f} {info.nframes} frames on {gpu}")
    result = {
        "id": vid, "tier": row["tier"], "category": row.get("category"),
        "ti_mean": float(row["ti_mean"]), "si_mean": float(row["si_mean"]),
        "width": info.width, "height": info.height, "fps": info.fps, "nframes": info.nframes,
        "device": gpu, "step_size": a.step_size, "models": {},
    }
    viewer = not a.no_viewer
    if viewer:
        make_preview(src, vdir / "original.mp4")

    lossless: dict[str, Path] = {}
    for spec_str in a.models:
        spec = parse_spec(spec_str, a.step_size)
        log(f"== {spec.label} (k={spec.step_size})")
        model = load_model(spec, a.device, a.ckpt_dir)
        nbits = nbits_of(model)
        msg = message_for(vid, nbits)
        warmup(model, a.device, size=(info.height, info.width))
        mres: dict = {"card": spec.card, "step_size": spec.step_size, "nbits": nbits}

        # negative control: an unwatermarked clip should decode to ~50% bits, p ~ 1
        dec, _ = extract_message(model, src, a.device)
        mres["unwatermarked"] = score_message(msg.numpy(), dec)

        mdir = vdir / spec.label
        mdir.mkdir(exist_ok=True)
        wm_path = work / f"{spec.label}_wm_lossless.mp4"
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        er = embed_video(model, src, wm_path, msg, a.device, segment=a.segment,
                         diff_path=(mdir / "diff_x20.mp4") if viewer else None,
                         snapshot_dir=mdir if viewer else None)
        secs_video = er.nframes / info.fps
        mres["embed"] = {
            "model_seconds": er.model_seconds, "wall_seconds": er.wall_seconds,
            "model_seconds_per_video_second": er.model_seconds / secs_video,
            "wall_seconds_per_video_second": er.wall_seconds / secs_video,
            "model_fps": er.nframes / er.model_seconds,
            "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if a.device.startswith("cuda") else None,
        }
        q = dict(er.quality)
        q["vmaf"] = None if a.no_vmaf else vmaf(wm_path, src, threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
        mres["quality"] = q
        log(f"   embed {er.model_seconds:.2f}s model / {er.wall_seconds:.2f}s wall | "
            f"PSNR {q['psnr']:.2f} SSIM {q['ssim']:.4f} LPIPS {q['lpips']:.4f} VMAF {q['vmaf']}")
        if viewer:
            make_preview(wm_path, mdir / "watermarked.mp4")

        attacks = {}
        dec, ex = extract_message(model, wm_path, a.device)
        attacks["none"] = {"family": "none", "strength": "-", **score_message(msg.numpy(), dec), **ex}
        for name in a.attacks:
            atk = ATTACKS[name]
            apath = work / f"{spec.label}_{name}.mp4"
            apply_attack(wm_path, apath, atk)
            dec, ex = extract_message(model, apath, a.device)
            attacks[name] = {"family": atk.family, "strength": atk.strength, **score_message(msg.numpy(), dec), **ex}
            log(f"   {name:16s} bit_acc={attacks[name]['bit_acc']:.3f} log10p={attacks[name]['log10_p']:.1f}")
            if a.keep_attacked:
                shutil.move(str(apath), mdir / f"attacked_{name}.mp4")
            else:
                apath.unlink(missing_ok=True)
        mres["attacks"] = attacks
        result["models"][spec.label] = mres
        lossless[spec.label] = wm_path
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if viewer and lossless:
        log("building side-by-side")
        side_by_side([src, *lossless.values()], vdir / "side_by_side.mp4")
        result["side_by_side_order"] = ["original", *lossless.keys()]

    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(result, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    log(f"wrote {res_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
