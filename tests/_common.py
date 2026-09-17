"""
Shared helpers for the sample-rate tests.

Importing this module puts <repo>/src first on sys.path and checks that `aware` really comes from this
repository, so an older AWARE installed in the environment can never be tested by mistake.
"""
import os
import sys
from functools import lru_cache

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if sys.path[0] != SRC:
    sys.path.insert(0, SRC)

import aware  # noqa: E402

if not os.path.abspath(aware.__file__).startswith(os.path.abspath(SRC)):
    raise ImportError(f"`aware` was imported from {aware.__file__}, expected {SRC}. "
                      f"Run the tests from a fresh interpreter (python -m unittest discover -s tests).")

ALL_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
# Rates whose STFT grid is an exact scaling of 16 kHz
EXACT_RATES = (8000, 12000, 16000, 24000, 32000, 48000)
# 44.1 kHz family: frame/hop lengths are rounded
ROUNDED_RATES = (11025, 22050, 44100)

BASE_SAMPLE_RATE = 16000
BASE_FRAME_LENGTH = 1024
BASE_HOP_LENGTH = 256
EMBEDDING_BANDS = (1000, 4000)


@lru_cache(maxsize=None)
def get_models(name: str = "AWARE"):
    """Load a model once per test process"""
    from aware.utils.models import load
    models = load(name)
    if models is None:
        raise RuntimeError(f"load('{name}') failed, see the log above")
    return models


def synthetic_voice(sample_rate: int, seconds: float = 2.0, extra_samples: int = 37, seed: int = 0) -> np.ndarray:
    """
    Voiced-speech-like test signal: harmonics of a slowly varying f0 with syllable-rate amplitude modulation
    plus a little white noise (so rates above 16 kHz also have content above 8 kHz).
    `extra_samples` makes the length a non-multiple of every hop length.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate) + extra_samples
    t = np.arange(n) / sample_rate

    f0 = 140.0 + 15.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    x = np.zeros(n)
    k = 1
    while k * 160.0 < sample_rate / 2:
        x += np.sin(k * phase) / k
        k += 1

    x *= 0.5 * (1.0 + 0.8 * np.sin(2 * np.pi * 4.0 * t))
    x += 0.01 * rng.standard_normal(n)
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def smooth_spectrogram(freqs: np.ndarray, n_frames: int, seed: int = 0) -> np.ndarray:
    """A positive magnitude spectrogram defined in Hz, so it can be sampled on the bin grid of any sample rate"""
    rng = np.random.default_rng(seed)
    knots = np.linspace(0.0, 24000.0, 300)
    values = rng.random((len(knots), n_frames)) + 0.1
    return np.stack([np.interp(freqs, knots, values[:, t]) for t in range(n_frames)], axis=1).astype(np.float32)


def high_band_change_db(original: np.ndarray, processed: np.ndarray, sample_rate: int, cutoff: float = 8500.0) -> float:
    """Energy of (processed - gain * original) above `cutoff` relative to the original energy there, in dB"""
    n = min(len(original), len(processed))
    x, y = original[:n].astype(np.float64), processed[:n].astype(np.float64)
    gain = np.dot(x, y) / (np.dot(x, x) + 1e-12)
    band = np.fft.rfftfreq(n, 1 / sample_rate) > cutoff
    ref = np.sum(np.abs(np.fft.rfft(x)[band]) ** 2)
    diff = np.sum(np.abs(np.fft.rfft(y - gain * x)[band]) ** 2)
    return float(10 * np.log10(diff / (ref + 1e-20) + 1e-20))
