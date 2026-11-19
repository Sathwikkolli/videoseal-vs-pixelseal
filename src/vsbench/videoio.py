"""Minimal ffmpeg-pipe video I/O: stream RGB frames in and out without loading whole videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    nframes: int


def probe(path: str | Path) -> VideoInfo:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(path),
    ]
    out = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    st = out["streams"][0]
    fps = float(Fraction(st.get("r_frame_rate", "30/1")))
    duration = float(st.get("duration") or out.get("format", {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    nframes = int(nb) if nb and nb.isdigit() else int(round(duration * fps))
    return VideoInfo(int(st["width"]), int(st["height"]), fps, duration, nframes)


def iter_frames(path: str | Path, batch: int = 32, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield uint8 arrays of shape (n, H, W, 3), n <= batch, in RGB order."""
    info = probe(path)
    w, h = info.width, info.height
    cmd = [FFMPEG, "-v", "error", "-i", str(path)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes * batch)
            if not buf:
                break
            n = len(buf) // frame_bytes
            if n == 0:
                break
            yield np.frombuffer(buf[: n * frame_bytes], dtype=np.uint8).reshape(n, h, w, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
        if rc not in (0, None) and not err.strip().endswith("Broken pipe"):
            # a non-zero code after we stopped reading early is fine; otherwise surface it
            if max_frames is None:
                raise RuntimeError(f"ffmpeg decode failed for {path}: {err[-2000:]}")


class VideoWriter:
    """Write uint8 RGB frames to an H.264 file through an ffmpeg pipe.

    lossless=True uses x264 qp 0 (bit-exact for yuv420p input); the result is not
    playable in most browsers, so make a separate preview for viewing.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 crf: int = 18, lossless: bool = False, preset: str = "medium"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        enc = ["-c:v", "libx264", "-preset", "veryfast" if lossless else preset]
        enc += ["-qp", "0"] if lossless else ["-crf", str(crf)]
        cmd = [
            FFMPEG, "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-", *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frames: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        self.proc.stderr.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed for {self.path}: {err[-2000:]}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run([FFMPEG, "-v", "error", "-y", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr[-2000:]}")


def make_preview(src: str | Path, dst: str | Path, crf: int = 18, height: int | None = None) -> None:
    """Browser-playable H.264 (High profile, yuv420p) copy for viewing."""
    vf = ["-vf", f"scale=-2:{height}"] if height else []
    run_ffmpeg(["-i", str(src), *vf, "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)])
