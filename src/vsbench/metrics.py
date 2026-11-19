"""Motion/texture measures (ITU-T P.910 SI/TI), watermark decoding scores and video quality."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .videoio import FFMPEG, iter_frames


# ---------------------------------------------------------------- SI / TI

def _luma(frames: np.ndarray) -> np.ndarray:
    """uint8 RGB (n,h,w,3) -> float32 BT.601 luma (n,h,w) in [0,255]."""
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _sobel_mag(y: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on the interior pixels of (n,h,w)."""
    gx = (y[:, :-2, 2:] + 2 * y[:, 1:-1, 2:] + y[:, 2:, 2:]) - (y[:, :-2, :-2] + 2 * y[:, 1:-1, :-2] + y[:, 2:, :-2])
    gy = (y[:, 2:, :-2] + 2 * y[:, 2:, 1:-1] + y[:, 2:, 2:]) - (y[:, :-2, :-2] + 2 * y[:, :-2, 1:-1] + y[:, :-2, 2:])
    return np.sqrt(gx * gx + gy * gy)


def si_ti(path: str | Path, max_frames: int | None = None, batch: int = 16) -> dict:
    """ITU-T P.910 spatial (SI) and temporal (TI) information, plus their means.

    SI = max_t std(Sobel(Y_t)),  TI = max_t std(Y_t - Y_{t-1}).
    The means are more stable against a single scene cut, so we report both and
    select tiers on ti_mean by default.
    """
    si_vals, ti_vals = [], []
    prev = None
    for frames in iter_frames(path, batch=batch, max_frames=max_frames):
        y = _luma(frames)
        si_vals.extend(_sobel_mag(y).reshape(len(y), -1).std(axis=1).tolist())
        seq = y if prev is None else np.concatenate([prev[None], y], axis=0)
        if len(seq) > 1:
            ti_vals.extend(np.diff(seq, axis=0).reshape(len(seq) - 1, -1).std(axis=1).tolist())
        prev = y[-1]
    return {
        "si": float(max(si_vals)) if si_vals else float("nan"),
        "ti": float(max(ti_vals)) if ti_vals else float("nan"),
        "si_mean": float(np.mean(si_vals)) if si_vals else float("nan"),
        "ti_mean": float(np.mean(ti_vals)) if ti_vals else float("nan"),
        "nframes": len(si_vals),
    }


# ---------------------------------------------------------------- watermark scores

def bit_accuracy(msg: np.ndarray, decoded: np.ndarray) -> float:
    msg, decoded = np.asarray(msg).astype(int).ravel(), np.asarray(decoded).astype(int).ravel()
    return float((msg == decoded).mean())


def log10_pvalue(n_correct: int, n_bits: int) -> float:
    """log10 P(X >= n_correct), X ~ Binomial(n_bits, 0.5): chance of matching this well by luck.

    Computed in log space so 256/256 correct gives about -77 instead of underflowing.
    """
    if n_correct <= 0:
        return 0.0
    terms = [
        math.lgamma(n_bits + 1) - math.lgamma(k + 1) - math.lgamma(n_bits - k + 1) - n_bits * math.log(2)
        for k in range(n_correct, n_bits + 1)
    ]
    m = max(terms)
    return float((m + math.log(sum(math.exp(t - m) for t in terms))) / math.log(10))


def score_message(msg: np.ndarray, decoded: np.ndarray) -> dict:
    msg = np.asarray(msg).astype(int).ravel()
    decoded = np.asarray(decoded).astype(int).ravel()
    n_correct = int((msg == decoded).sum())
    lp = log10_pvalue(n_correct, len(msg))
    return {
        "bit_acc": n_correct / len(msg),
        "n_correct": n_correct,
        "n_bits": len(msg),
        "log10_p": lp,
        "detected_fpr1e-6": lp < -6,
        "detected_fpr1e-3": lp < -3,
    }


# ---------------------------------------------------------------- quality

class QualityAccumulator:
    """Streams (original, watermarked) uint8 frame batches and accumulates quality metrics.

    PSNR uses every frame. SSIM and LPIPS are computed on every `sample_every`-th frame
    (they are expensive at 1080p); LPIPS runs at `lpips_height` to stay within memory.
    """

    def __init__(self, device: str = "cuda", sample_every: int = 10, lpips_height: int = 512):
        self.device = device
        self.sample_every = sample_every
        self.lpips_height = lpips_height
        self.sq_err = 0.0
        self.n_values = 0
        self.ssim_vals: list[float] = []
        self.lpips_vals: list[float] = []
        self.frame_idx = 0
        self._lpips = None

    def _lpips_model(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        return self._lpips

    def update(self, orig: np.ndarray, wm: np.ndarray) -> None:
        import torch
        import torch.nn.functional as F
        from pytorch_msssim import ssim

        d = orig.astype(np.float64) - wm.astype(np.float64)
        self.sq_err += float((d * d).sum())
        self.n_values += d.size

        idx = [i for i in range(len(orig)) if (self.frame_idx + i) % self.sample_every == 0]
        self.frame_idx += len(orig)
        if not idx:
            return
        with torch.no_grad():
            a = torch.from_numpy(orig[idx]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            b = torch.from_numpy(wm[idx]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            self.ssim_vals.extend(ssim(a, b, data_range=1.0, size_average=False).tolist())
            h = self.lpips_height
            w = int(round(a.shape[-1] * h / a.shape[-2] / 2) * 2)
            a_s = F.interpolate(a, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
            b_s = F.interpolate(b, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
            self.lpips_vals.extend(self._lpips_model()(a_s * 2 - 1, b_s * 2 - 1).flatten().tolist())

    def result(self) -> dict:
        mse = self.sq_err / max(self.n_values, 1)
        return {
            "psnr": float(10 * math.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf"),
            "ssim": float(np.mean(self.ssim_vals)) if self.ssim_vals else float("nan"),
            "lpips": float(np.mean(self.lpips_vals)) if self.lpips_vals else float("nan"),
        }


@lru_cache(maxsize=1)
def vmaf_available() -> bool:
    out = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return "libvmaf" in out


def vmaf(distorted: str | Path, reference: str | Path, threads: int = 8) -> float | None:
    """Mean VMAF of distorted vs reference (same size). None if ffmpeg lacks libvmaf."""
    if not vmaf_available():
        return None
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vmaf.json"
        # libvmaf's option parser treats ':' and '\' specially, so pass a relative path
        cmd = [FFMPEG, "-v", "error", "-i", str(Path(distorted).resolve()), "-i", str(Path(reference).resolve()),
               "-lavfi", f"[0:v][1:v]libvmaf=log_fmt=json:log_path=vmaf.json:n_threads={threads}",
               "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
        if proc.returncode != 0 or not log.exists():
            return None
        data = json.loads(log.read_text())
        return float(data["pooled_metrics"]["vmaf"]["mean"])
