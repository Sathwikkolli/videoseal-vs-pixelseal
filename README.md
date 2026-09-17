# AWARE: Audio Watermarking via Adversarial Resistance to Edits

## Installation
```bash
git clone https://github.com/deepmarkpy/aware.git
cd ./aware
python -m pip install -e .
```

## Basic Usage
```python
import numpy as np
import librosa
import soundfile
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from aware.metrics.audio import BER, PESQ

# 1. load model
# `name` selects the configuration profile:
#   - "AWARE" (default): embedding is computed over the full sequence
#   - "AWARE(20bps)": embedding is computed per second of audio
embedder, detector = load(name="AWARE")

# 2.create 20-bit watermark
watermark_bits = np.random.randint(0, 2, size=20, dtype=np.int32)


# 3.read host audio
# "AWARE" accepts 8000-48000Hz audio at its native rate, "AWARE(20bps)" requires 16kHz.
# Keep the file's own rate with sr=None:
signal, sample_rate = librosa.load("example.wav", sr=None, mono=True)


# 4.embed watermark
watermarked_signal = embed_watermark(signal, sample_rate, watermark_bits, embedder)
# you can save it as a new one:
# soundfile.write("output.wav", watermarked_signal, sample_rate)


# 5. detect watermark
# Detect the watermark from the audio. The function returns:
#   - detected_pattern: the recovered watermark bits
#   - confidence: a value in [0, 1] representing the estimated probability;
#                 that the signal contains a watermark.
# If confidence >= tau (default tau = 0.5), the signal is considered watermarked;
# otherwise, it is treated as unwatermarked.
detected_pattern, confidence = detect_watermark(watermarked_signal, sample_rate, detector)


# 6.check accuracy and perceptual quality
ber_metric = BER()
ber = ber_metric(watermark_bits, detected_pattern)
print(f"BER: {ber:.2f}")

pesq_metric = PESQ()
pesq = pesq_metric(watermarked_signal, signal, sample_rate)
print(f"PESQ: {pesq:.2f}")
```

## Sample rates

The default `AWARE` model embeds and detects at the audio's own sample rate, from 8 kHz to 48 kHz, without resampling it.
AWARE was designed at 16 kHz, so the STFT is scaled with the sample rate so that every rate looks the same to the detector:

| | formula | 16 kHz | 22.05 kHz | 44.1 kHz | 48 kHz |
|---|---|---|---|---|---|
| frame / window length | even(1024 · sr / 16000) | 1024 | 1412 | 2822 | 3072 |
| hop length | round(256 · sr / 16000) | 256 | 353 | 706 | 768 |
| bin width | sr / frame length | 15.63 Hz | 15.62 Hz | 15.63 Hz | 15.63 Hz |

The mel filter bank of the detector always covers 0-8000 Hz (`fmax` in `config_full_length.yaml`), and the watermark is
embedded in the same 1000-4000 Hz band at every rate. Content above 8 kHz is not modified.

- Detection uses the rate the audio has *now*: a file watermarked at 44.1 kHz and later converted to 16 kHz is detected with `sample_rate=16000`.
- Rates that are simple multiples of 16 kHz (8k, 12k, 24k, 32k, 48k) match the 16 kHz detector exactly, the 44.1 kHz family matches up to rounding (~0.06 %).
- At 8 kHz the 4 kHz band edge equals Nyquist, a warning is logged and robustness may be lower.
- The speech check (WebRTC VAD) only supports 8/16/32/48 kHz; for other rates it runs on a temporary 16 kHz copy.
- `AWARE(20bps)` still requires 16 kHz.

Check a setup with `python scripts/check_sample_rates.py --audio speech.wav`, or evaluate attacks at a given rate with
`python scripts/test.py --audio-dir libri --sample-rate 44100`.

## License

This project is licensed under the terms of the MIT license.

## Citation
If you find this repository useful, we’d greatly appreciate it if you could give it a star ⭐.

Please cite our work if you use it in your research.

```
@article{pavlović2025aware,
      title={AWARE: Audio Watermarking with Adversarial Resistance to Edits}, 
      author={Kosta Pavlović and Lazar Stanarević and Petar Nedić and Slavko Kovačević and Igor Djurović},
      year={2025},
      eprint={2510.17512},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.17512}, 
}
```
