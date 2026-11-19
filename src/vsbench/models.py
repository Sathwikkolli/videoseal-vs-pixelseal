"""Load Video Seal / Pixel Seal and run streaming embed / extract on video files.

Model specs are strings: "videoseal" (Video Seal 1.0), "pixelseal", optionally with a
step size override, e.g. "pixelseal@k8". Without an override the benchmark-wide
--step-size is used so both models propagate the watermark the same way.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .metrics import QualityAccumulator
from .videoio import VideoWriter, iter_frames, probe


@dataclass
class ModelSpec:
    spec: str
    card: str
    step_size: int

    @property
    def label(self) -> str:
        return self.spec


def parse_spec(spec: str, default_step: int) -> ModelSpec:
    card, _, opt = spec.partition("@")
    step = int(opt[1:]) if opt.startswith("k") else default_step
    return ModelSpec(spec=spec if opt else card, card=card, step_size=step)


@contextlib.contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_model(spec: ModelSpec, device: str, ckpt_dir: str | Path):
    """videoseal.load() downloads checkpoints into ./ckpts and reads ./configs/attenuation.yaml
    (relative to the working directory), so run it from ckpt_dir with the configs copied in."""
    import shutil

    import videoseal

    ckpt_dir = Path(ckpt_dir).resolve()
    cfg_src = Path(videoseal.__file__).resolve().parents[1] / "configs"
    if cfg_src.is_dir() and not (ckpt_dir / "configs").is_dir():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cfg_src, ckpt_dir / "configs")
    with _cwd(ckpt_dir):
        model = videoseal.load(spec.card)
    model.eval().to(device)
    model.step_size = spec.step_size
    return model


def message_for(video_id: str, nbits: int) -> torch.Tensor:
    """Deterministic per-video message, identical for every model (1, nbits) in {0,1}."""
    seed = int.from_bytes(video_id.encode(), "little") % (2**31 - 1)
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (1, nbits), generator=g)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _to_tensor(frames: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return x.clamp(0, 1).mul(255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def warmup(model, device: str, size=(1080, 1920)) -> None:
    """One small embed + detect so cuDNN autotuning is not counted in the timings."""
    x = torch.rand(model.step_size * 2, 3, *size, device=device)
    msg = torch.randint(0, 2, (1, nbits_of(model)), device=device)
    with torch.no_grad():
        model.embed(x, msgs=msg, is_video=True)
        model.detect(x[:2], is_video=True)
    _sync(device)


def nbits_of(model) -> int:
    return int(model.get_random_msg().shape[-1])


@dataclass
class EmbedResult:
    model_seconds: float
    wall_seconds: float
    nframes: int
    quality: dict
    snapshot_frame: int
    extra: dict = field(default_factory=dict)


def embed_video(model, src: str | Path, dst_lossless: str | Path, msg: torch.Tensor, device: str,
                segment: int = 32, diff_path: str | Path | None = None, diff_gain: float = 20.0,
                snapshot_dir: str | Path | None = None, quality_every: int = 10) -> EmbedResult:
    """Watermark `src` segment by segment and write a lossless watermarked file.

    `segment` must be a multiple of the model's step size so key frames land exactly where
    they would if the whole video were embedded at once. Also writes an amplified
    difference video (|wm - orig| * gain) and PNG snapshots of the middle frame.
    """
    if segment % model.step_size:
        raise ValueError(f"segment ({segment}) must be a multiple of step_size ({model.step_size})")
    info = probe(src)
    mid = info.nframes // 2
    qa = QualityAccumulator(device=device, sample_every=quality_every)
    writer = VideoWriter(dst_lossless, info.width, info.height, info.fps, lossless=True)
    diff_writer = VideoWriter(diff_path, info.width, info.height, info.fps, crf=18) if diff_path else None
    msg = msg.to(device)

    model_s, n = 0.0, 0
    t_wall = time.perf_counter()
    for frames in iter_frames(src, batch=segment):
        x = _to_tensor(frames, device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.embed(x, msgs=msg, is_video=True)
        _sync(device)
        model_s += time.perf_counter() - t0
        wm = _to_uint8(out["imgs_w"])
        writer.write(wm)
        qa.update(frames, wm)
        diff = None
        if diff_writer is not None or (snapshot_dir and n <= mid < n + len(frames)):
            diff = np.clip(np.abs(wm.astype(np.int16) - frames.astype(np.int16)) * diff_gain, 0, 255).astype(np.uint8)
        if diff_writer is not None:
            diff_writer.write(diff)
        if snapshot_dir and n <= mid < n + len(frames):
            _save_snapshots(Path(snapshot_dir), frames[mid - n], wm[mid - n], diff[mid - n])
        n += len(frames)
        del x, out
    writer.close()
    if diff_writer is not None:
        diff_writer.close()
    wall = time.perf_counter() - t_wall
    return EmbedResult(model_seconds=model_s, wall_seconds=wall, nframes=n, quality=qa.result(), snapshot_frame=mid)


def _save_snapshots(d: Path, orig: np.ndarray, wm: np.ndarray, diff: np.ndarray) -> None:
    from PIL import Image

    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(orig).save(d / "frame_original.png")
    Image.fromarray(wm).save(d / "frame_watermarked.png")
    Image.fromarray(diff).save(d / "frame_diff_x20.png")


def extract_message(model, path: str | Path, device: str, batch: int = 32,
                    max_frames: int | None = None) -> tuple[np.ndarray, dict]:
    """Average per-frame bit logits over the video (paper Eq. 3) and threshold at 0."""
    total, n = None, 0
    t0 = time.perf_counter()
    for frames in iter_frames(path, batch=batch, max_frames=max_frames):
        x = _to_tensor(frames, device)
        with torch.no_grad():
            preds = model.detect(x, is_video=True)["preds"]
        bits = preds[:, 1:].double()
        if bits.ndim > 2:  # pixelwise models return (f, k, h, w): average over space
            bits = bits.flatten(2).mean(-1)
        bits = bits.sum(0)
        total = bits if total is None else total + bits
        n += len(frames)
    _sync(device)
    avg = (total / n).cpu().numpy()
    return (avg > 0).astype(np.int64), {"frames_used": n, "extract_seconds": time.perf_counter() - t0}
