"""
Sanity checks for multi sample rate support of the default AWARE model.

For every requested sample rate it reports:
    - the derived STFT parameters (frame/hop length, bin width, number of embedding bins)
    - detector equivalence: correlation of the raw detector outputs on clean audio at that rate vs. at 16 kHz
    - embedding at that rate: output length, runtime, BER/confidence when detecting at the same rate
    - cross-rate: BER/confidence after resampling the watermarked audio to 16 kHz
    - clean false positive check: confidence on the un-watermarked audio at that rate
    - high band preservation (rates > 16 kHz): change above 8.5 kHz relative to the original, in dB

Usage:
    python scripts/check_sample_rates.py --audio path/to/speech.wav
    python scripts/check_sample_rates.py --audio speech.wav --rates 16000,22050,44100 --seconds 5
    python scripts/check_sample_rates.py --params-only
"""
import argparse
import time
from math import gcd

import librosa
import numpy as np
from scipy.signal import resample_poly

from aware.utils.audio import get_rate_params
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

DEFAULT_RATES = "8000,11025,12000,16000,22050,24000,32000,44100,48000"
REFERENCE_RATE = 16000


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.copy()
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def ber(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def high_band_change_db(original: np.ndarray, watermarked: np.ndarray, sample_rate: int, cutoff: float = 8500.0) -> str:
    """Energy of (watermarked - gain * original) above `cutoff`, relative to the original energy there"""
    n = min(len(original), len(watermarked))
    x, y = original[:n].astype(np.float64), watermarked[:n].astype(np.float64)
    gain = np.dot(x, y) / (np.dot(x, x) + 1e-12)  # embedding rescales the whole signal
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    band = freqs > cutoff

    ref_energy = np.sum(np.abs(np.fft.rfft(x)[band]) ** 2)
    if ref_energy < 1e-8:
        return "n/a (no content)"
    diff_energy = np.sum(np.abs(np.fft.rfft(y - gain * x)[band]) ** 2)
    return f"{10 * np.log10(diff_energy / ref_energy + 1e-12):.1f} dB"


def print_params(embedder, rates):
    print(f"\n{'rate':>7} {'frame':>6} {'hop':>5} {'bin Hz':>8} {'hop ms':>7} {'band bins':>10}")
    for sr in rates:
        p = get_rate_params(sr, embedder.base_sample_rate, embedder.frame_length, embedder.hop_length,
                            embedder.win_length, embedder.embedding_bands)
        print(f"{sr:>7} {p.frame_length:>6} {p.hop_length:>5} {p.bin_width:>8.3f} "
              f"{1000 * p.hop_length / sr:>7.2f} {len(p.band_indices):>10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", help="speech file (any sample rate, mono or stereo)")
    parser.add_argument("--rates", default=DEFAULT_RATES, help=f"comma separated sample rates (default: {DEFAULT_RATES})")
    parser.add_argument("--seconds", type=float, default=5.0, help="crop the audio to this many seconds (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the watermark bits")
    parser.add_argument("--params-only", action="store_true", help="only print the derived STFT parameters")
    args = parser.parse_args()

    rates = [int(r) for r in args.rates.split(",") if r.strip()]
    embedder, detector = load("AWARE")

    print_params(embedder, rates)
    if args.params_only:
        return
    if args.audio is None:
        parser.error("--audio is required unless --params-only is given")

    audio, native_sr = librosa.load(args.audio, sr=None, mono=True)
    audio = audio[: int(args.seconds * native_sr)]
    print(f"\nInput: {args.audio} ({native_sr}Hz, {len(audio) / native_sr:.2f}s)")
    if native_sr < max(rates):
        print(f"Note: rates above {native_sr}Hz are upsampled, they have no content above {native_sr // 2}Hz.")

    watermark_bits = np.random.default_rng(args.seed).integers(0, 2, size=20, dtype=np.int32)

    # Reference detector output on clean audio at 16 kHz
    reference_clean = detector.detect(resample(audio, native_sr, REFERENCE_RATE), REFERENCE_RATE)

    header = (f"{'rate':>7} | {'clean corr':>10} | {'clean conf':>10} | {'len ok':>6} | {'embed s':>7} | "
              f"{'BER same':>8} | {'conf same':>9} | {'BER ->16k':>9} | {'conf ->16k':>10} | high band change")
    print("\n" + header)
    print("-" * len(header))

    for sr in rates:
        x = resample(audio, native_sr, sr)

        clean_values = detector.detect(x, sr)
        clean_corr = np.corrcoef(clean_values, reference_clean)[0, 1]
        _, clean_conf = detect_watermark(x, sr, detector)

        try:
            start = time.time()
            wm = embed_watermark(x, sr, watermark_bits, embedder)
            embed_seconds = time.time() - start
        except ValueError as e:
            print(f"{sr:>7} | embedding rejected: {e}")
            continue

        bits_same, conf_same = detect_watermark(wm, sr, detector)
        bits_16k, conf_16k = detect_watermark(resample(wm, sr, REFERENCE_RATE), REFERENCE_RATE, detector)
        high_band = high_band_change_db(x, wm, sr) if sr > REFERENCE_RATE else "-"

        print(f"{sr:>7} | {clean_corr:>10.3f} | {clean_conf:>10.3f} | {str(len(wm) == len(x)):>6} | {embed_seconds:>7.1f} | "
              f"{ber(watermark_bits, bits_same):>8.3f} | {conf_same:>9.3f} | {ber(watermark_bits, bits_16k):>9.3f} | "
              f"{conf_16k:>10.3f} | {high_band}")

    print("\nExpected: clean corr close to 1, clean conf low, len ok True, BER close to 0 and conf high "
          "for both same-rate and ->16k, high band change very negative.")


if __name__ == "__main__":
    main()
