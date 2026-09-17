"""
Light end-to-end check of multi sample rate AWARE on real Emilia speech (not a performance evaluation).

For a few Emilia clips, each clip is resampled to every test rate and run through the public API
(embed_watermark / detect_watermark, including the speech check):

    clean_conf        detector confidence on the un-watermarked audio (negative control)
    len_ok / finite   watermarked audio has the input length and no NaN/Inf
    ber_same          BER when detecting at the embedding rate
    ber_16k           BER after resampling the watermarked audio to 16 kHz (cross-rate)
    ber_native        BER after resampling back to the clip's native rate (cross-rate)
    snr_db            SNR of the watermark after gain compensation
    high_band_db      change above 8.5 kHz relative to the original (rates > 16 kHz, should be very negative)

Pass criteria per rate (8 kHz only warns, its band edge sits at Nyquist):
    all outputs len_ok and finite, mean ber_same <= 0.05, mean ber_16k <= 0.10

Usage:
    python scripts/emilia_smoke.py
    python scripts/emilia_smoke.py --n-clips 4 --seconds 6 --rates 8000,16000,22050,44100,48000
    EMILIA_CSV=/path/to/manifest.csv python scripts/emilia_smoke.py --out results/emilia_smoke
"""
import argparse
import csv
import os
import random
import sys
import time
from math import gcd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import aware  # noqa: E402
from aware.service import embed_watermark, detect_watermark  # noqa: E402
from aware.utils.models import load  # noqa: E402

DEFAULT_CSV = os.environ.get("EMILIA_CSV", "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv")
DEFAULT_RATES = "8000,11025,16000,22050,32000,44100,48000"
PATH_KEYS = ("path", "audio_path", "wav", "filepath", "file", "audio", "filename")
DUR_KEYS = ("duration", "dur", "length", "seconds", "duration_s")
MARGINAL_RATES = {8000}
MAX_MEAN_BER_SAME = 0.05
MAX_MEAN_BER_16K = 0.10


def read_manifest(csv_path, min_duration):
    if not os.path.exists(csv_path):
        sys.exit(f"Emilia manifest not found: {csv_path} (set EMILIA_CSV or --csv)")
    paths = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        path_col = next((columns[k] for k in PATH_KEYS if k in columns), None)
        dur_col = next((columns[k] for k in DUR_KEYS if k in columns), None)
        if path_col is None:
            sys.exit(f"no path column in {csv_path}, columns: {reader.fieldnames}")
        for row in reader:
            path = (row.get(path_col) or "").strip()
            if not path:
                continue
            if dur_col:
                try:
                    if float(row.get(dur_col) or "nan") < min_duration:
                        continue
                except ValueError:
                    pass
            paths.append(path)
    return paths


def pick_clips(paths, n_clips, seconds, seed):
    random.Random(seed).shuffle(paths)
    clips = []
    for path in paths:
        if len(clips) == n_clips:
            break
        if not os.path.exists(path):
            continue
        try:
            audio, sample_rate = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        n = int(seconds * sample_rate)
        if len(audio) < n:
            continue
        clips.append((path, audio[:n].astype(np.float32), int(sample_rate)))
    return clips


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.copy()
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def ber(expected, detected):
    return float(np.mean(np.asarray(expected) != np.asarray(detected)))


def gain_compensated(original, processed):
    n = min(len(original), len(processed))
    x, y = original[:n].astype(np.float64), processed[:n].astype(np.float64)
    return x, y, np.dot(x, y) / (np.dot(x, x) + 1e-12)


def snr_db(original, processed):
    x, y, gain = gain_compensated(original, processed)
    return float(10 * np.log10(np.sum((gain * x) ** 2) / (np.sum((y - gain * x) ** 2) + 1e-20)))


def high_band_db(original, processed, sample_rate, cutoff=8500.0):
    x, y, gain = gain_compensated(original, processed)
    band = np.fft.rfftfreq(len(x), 1 / sample_rate) > cutoff
    ref = np.sum(np.abs(np.fft.rfft(x)[band]) ** 2)
    if ref < 1e-8:
        return float("nan")
    return float(10 * np.log10(np.sum(np.abs(np.fft.rfft(y - gain * x)[band]) ** 2) / ref + 1e-20))


def fmt(value, spec=".3f"):
    return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Emilia manifest CSV (default: $EMILIA_CSV or the turbo path)")
    parser.add_argument("--n-clips", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=6.0, help="crop length per clip")
    parser.add_argument("--rates", default=DEFAULT_RATES, help="comma separated test rates, the native rate is always added")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "emilia_smoke"))
    args = parser.parse_args()

    print(f"aware imported from: {aware.__file__}")
    if not os.path.abspath(aware.__file__).startswith(SRC):
        sys.exit(f"expected aware from {SRC}")

    clips = pick_clips(read_manifest(args.csv, args.seconds), args.n_clips, args.seconds, args.seed)
    if len(clips) < args.n_clips:
        sys.exit(f"only found {len(clips)} usable clips")

    embedder, detector = load("AWARE")
    rng = np.random.default_rng(args.seed)
    test_rates = [int(r) for r in args.rates.split(",") if r.strip()]
    rows = []

    for clip_index, (path, native_audio, native_sr) in enumerate(clips):
        bits = rng.integers(0, 2, size=20, dtype=np.int32)
        print(f"\n[{clip_index + 1}/{len(clips)}] {os.path.basename(path)} (native {native_sr}Hz)")

        for sample_rate in sorted(set(test_rates) | {native_sr}):
            audio = resample(native_audio, native_sr, sample_rate)
            row = {"clip": os.path.basename(path), "native_sr": native_sr, "sample_rate": sample_rate, "status": "ok"}

            _, row["clean_conf"] = detect_watermark(audio, sample_rate, detector)

            try:
                start = time.time()
                watermarked = embed_watermark(audio, sample_rate, bits, embedder)
                row["embed_s"] = time.time() - start
            except ValueError as e:
                row["status"] = f"rejected: {e}"
                rows.append(row)
                print(f"  {sample_rate:>5}Hz  {row['status']}")
                continue

            row["len_ok"] = len(watermarked) == len(audio)
            row["finite"] = bool(np.all(np.isfinite(watermarked)))

            detected, row["conf_same"] = detect_watermark(watermarked, sample_rate, detector)
            row["ber_same"] = ber(bits, detected)
            detected, row["conf_16k"] = detect_watermark(resample(watermarked, sample_rate, 16000), 16000, detector)
            row["ber_16k"] = ber(bits, detected)
            detected, row["conf_native"] = detect_watermark(resample(watermarked, sample_rate, native_sr), native_sr, detector)
            row["ber_native"] = ber(bits, detected)
            row["snr_db"] = snr_db(audio, watermarked)
            row["high_band_db"] = high_band_db(audio, watermarked, sample_rate) if sample_rate > 16000 else float("nan")
            rows.append(row)

            print(f"  {sample_rate:>5}Hz  ber same/16k/native={row['ber_same']:.2f}/{row['ber_16k']:.2f}/{row['ber_native']:.2f}  "
                  f"conf={row['conf_same']:.3f} clean_conf={row['clean_conf']:.3f}  snr={row['snr_db']:.1f}dB  "
                  f"high_band={fmt(row['high_band_db'], '.1f')}dB  len_ok={row['len_ok']}  {row['embed_s']:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    columns = ["clip", "native_sr", "sample_rate", "status", "clean_conf", "len_ok", "finite", "ber_same", "conf_same",
               "ber_16k", "conf_16k", "ber_native", "conf_native", "snr_db", "high_band_db", "embed_s"]
    csv_path = os.path.join(args.out, "emilia_smoke.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer rate summary ({len(clips)} clips, {args.seconds:g}s each)")
    print(f"{'rate':>6} {'embedded':>8} {'BER same':>9} {'BER 16k':>8} {'BER nat':>8} {'conf':>6} {'clean conf':>10} "
          f"{'SNR dB':>7} {'>8.5k dB':>9} {'embed s':>8}  result")
    failures = []
    for sample_rate in sorted({row["sample_rate"] for row in rows}):
        rate_rows = [r for r in rows if r["sample_rate"] == sample_rate]
        done = [r for r in rate_rows if r["status"] == "ok"]

        def mean(key):
            values = [r[key] for r in done if not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.mean(values)) if values else float("nan")

        problems = []
        if len(done) < len(rate_rows):
            problems.append(f"{len(rate_rows) - len(done)} rejected")
        if not all(r["len_ok"] and r["finite"] for r in done):
            problems.append("length/finite")
        if done and mean("ber_same") > MAX_MEAN_BER_SAME:
            problems.append(f"BER same {mean('ber_same'):.3f} > {MAX_MEAN_BER_SAME}")
        if done and mean("ber_16k") > MAX_MEAN_BER_16K:
            problems.append(f"BER 16k {mean('ber_16k'):.3f} > {MAX_MEAN_BER_16K}")

        if not problems:
            result = "PASS"
        elif sample_rate in MARGINAL_RATES:
            result = "WARN (" + "; ".join(problems) + ")"
        else:
            result = "FAIL (" + "; ".join(problems) + ")"
            failures.append(sample_rate)

        clean = float(np.mean([r["clean_conf"] for r in rate_rows]))
        print(f"{sample_rate:>6} {len(done):>4}/{len(rate_rows):<3} {fmt(mean('ber_same')):>9} {fmt(mean('ber_16k')):>8} "
              f"{fmt(mean('ber_native')):>8} {fmt(mean('conf_same')):>6} {fmt(clean):>10} {fmt(mean('snr_db'), '.1f'):>7} "
              f"{fmt(mean('high_band_db'), '.1f'):>9} {fmt(mean('embed_s'), '.1f'):>8}  {result}")

    print(f"\nrows written to {csv_path}")
    if failures:
        print(f"FAILED rates: {failures}")
        sys.exit(1)
    print("all rates passed")


if __name__ == "__main__":
    main()
