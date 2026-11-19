"""Everyday video attacks, applied with real ffmpeg (not differentiable approximations).

Families and strengths:
  compress   H.264 re-encode at CRF 23 / 28 / 33         (platform upload)
  crop       centre crop keeping 90 / 75 / 50 % of area   (trim / reframe)
  downscale  720p / 480p / 360p                           (platform serves a smaller rendition)
  combined   crop 75 % -> 480p -> CRF 28                  (all three at once)

Crop and downscale outputs are encoded losslessly so each family is measured on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .videoio import run_ffmpeg


@dataclass(frozen=True)
class Attack:
    name: str
    family: str
    strength: str
    vf: str | None
    crf: int | None  # None = lossless (qp 0)


def _crop(area: float) -> str:
    s = math.sqrt(area)
    return f"crop=trunc(iw*{s:.4f}/2)*2:trunc(ih*{s:.4f}/2)*2"


ATTACKS: dict[str, Attack] = {a.name: a for a in [
    Attack("compress_crf23", "compress", "CRF 23", None, 23),
    Attack("compress_crf28", "compress", "CRF 28", None, 28),
    Attack("compress_crf33", "compress", "CRF 33", None, 33),
    Attack("crop_area90", "crop", "90% area", _crop(0.90), None),
    Attack("crop_area75", "crop", "75% area", _crop(0.75), None),
    Attack("crop_area50", "crop", "50% area", _crop(0.50), None),
    Attack("downscale_720p", "downscale", "720p", "scale=-2:720:flags=bicubic", None),
    Attack("downscale_480p", "downscale", "480p", "scale=-2:480:flags=bicubic", None),
    Attack("downscale_360p", "downscale", "360p", "scale=-2:360:flags=bicubic", None),
    Attack("combined", "combined", "crop75+480p+CRF28", f"{_crop(0.75)},scale=-2:480:flags=bicubic", 28),
]}

DEFAULT_ATTACKS = list(ATTACKS)


def apply_attack(src: str | Path, dst: str | Path, attack: Attack | str, seconds: float | None = None) -> Path:
    """Apply `attack` to `src`. `seconds` limits the output to the first N seconds (saves time
    when only those frames are decoded afterwards)."""
    if isinstance(attack, str):
        attack = ATTACKS[attack]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src)]
    if seconds:
        args += ["-t", f"{seconds}"]
    if attack.vf:
        args += ["-vf", attack.vf]
    if attack.crf is None:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0"]
    else:
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(attack.crf)]
    args += ["-pix_fmt", "yuv420p", "-an", str(dst)]
    run_ffmpeg(args)
    return dst
