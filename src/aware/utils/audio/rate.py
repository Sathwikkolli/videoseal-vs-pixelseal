"""
Sample-rate dependent STFT parameters.

AWARE was designed and tuned at 16 kHz (frame 1024, hop 256, 128 mel bands over 0-8000 Hz).
To embed/detect at another sample rate without resampling the audio, the STFT is scaled with
the sample rate so that every rate "looks the same" to the detection network:

    - bin width stays ~15.6 Hz      (frame_length scales with sr / 16000)
    - hop duration stays ~16 ms      (hop_length scales with sr / 16000)
    - mel bands cover the same Hz    (mel fmax is fixed, see AWAREDetectorNet)

Rates that are integer/simple multiples of 16 kHz (8k, 12k, 24k, 32k, 48k) match exactly,
the 44.1 kHz family (11.025k, 22.05k, 44.1k) matches up to rounding (~0.06 %).
"""
from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np

from aware.utils.logger import logger

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000

# If the upper embedding band edge is above this fraction of Nyquist the watermark still works,
# but the top of the band is usually attenuated by the anti-aliasing filter of real recordings.
_NYQUIST_WARNING_RATIO = 0.9


@dataclass(frozen=True)
class RateParams:
    """STFT parameters and embedding band bin indices for one sample rate"""
    sample_rate: int
    frame_length: int
    win_length: int
    hop_length: int
    bin_width: float
    band_indices: np.ndarray
    out_of_band_indices: np.ndarray


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _round_to_even(x: float) -> int:
    return 2 * _round_half_up(x / 2)


def validate_sample_rate(sample_rate: int, embedding_bands: tuple[int, int]) -> None:
    """Raise ValueError if the sample rate is not supported for the given embedding band"""
    if isinstance(sample_rate, bool) or not float(sample_rate).is_integer():
        raise ValueError(f"Invalid sample rate. Expected an integer number of Hz, got {sample_rate}.")

    sample_rate = int(sample_rate)
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError(f"Invalid sample rate. Expected {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}Hz, got {sample_rate}Hz.")

    nyquist = sample_rate / 2
    band_high = embedding_bands[1]
    if band_high > nyquist:
        raise ValueError(f"Sample rate {sample_rate}Hz is too low for embedding band {tuple(embedding_bands)}Hz "
                         f"(Nyquist {nyquist:.0f}Hz < {band_high}Hz).")

    if band_high > _NYQUIST_WARNING_RATIO * nyquist:
        logger.warning(f"Embedding band upper edge {band_high}Hz is close to Nyquist ({nyquist:.0f}Hz) "
                       f"at {sample_rate}Hz, robustness may be reduced.")


@lru_cache(maxsize=None)
def _compute_rate_params(sample_rate: int, base_sample_rate: int, base_frame_length: int, base_hop_length: int,
                         base_win_length: int, embedding_bands: tuple[int, int]) -> RateParams:
    scale = sample_rate / base_sample_rate

    frame_length = _round_to_even(base_frame_length * scale)
    win_length = min(_round_to_even(base_win_length * scale), frame_length)
    hop_length = _round_half_up(base_hop_length * scale)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    bin_width = sample_rate / frame_length
    # Half-bin tolerance at the band edges: when rounding shifts the bin grid (e.g. 22.05 kHz has a bin at 999.4 Hz)
    # the band keeps the same bins as at 16 kHz. For exact multiples of 16 kHz the selection is unchanged.
    tolerance = 0.5 * bin_width
    mask = (freqs >= embedding_bands[0] - tolerance) & (freqs <= embedding_bands[1] + tolerance)

    # Cached and shared between calls: treat as read-only
    band_indices = np.where(mask)[0]
    out_of_band_indices = np.where(~mask)[0]

    return RateParams(
        sample_rate=sample_rate,
        frame_length=frame_length,
        win_length=win_length,
        hop_length=hop_length,
        bin_width=bin_width,
        band_indices=band_indices,
        out_of_band_indices=out_of_band_indices,
    )


def get_rate_params(sample_rate: int, base_sample_rate: int, base_frame_length: int, base_hop_length: int,
                    base_win_length: int, embedding_bands: tuple[int, int]) -> RateParams:
    """
    Compute (cached) STFT parameters for `sample_rate`, scaled from the base (reference) configuration.

    At `sample_rate == base_sample_rate` the base parameters are returned unchanged.
    """
    return _compute_rate_params(int(sample_rate), int(base_sample_rate), int(base_frame_length),
                                int(base_hop_length), int(base_win_length), tuple(int(b) for b in embedding_bands))
