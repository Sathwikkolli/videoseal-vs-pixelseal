# Multi sample rate tests

Correctness checks for embedding/detecting at 8-48 kHz. They confirm the implementation works, they are not a
performance evaluation.

## Run on Great Lakes

```bash
sbatch $HOME/aware-multi-sr/scripts/sample_rate_tests.sbatch     # unit tests + Emilia smoke test, ~1 h CPU
```

or interactively (e.g. in an `salloc` session):

```bash
conda activate wmcompare
cd $HOME/aware-multi-sr
python -m unittest discover -s tests -v                          # unit tests only
python -m unittest discover -s tests -p test_rate_params.py -v   # one file
python scripts/emilia_smoke.py --n-clips 2 --seconds 5           # quick smoke test
```

No `pip install` needed: the tests put `src/` of this repo first on the path and fail if `aware` is imported from
anywhere else, so an AWARE installed in the environment is never tested by mistake.

## What is checked

| File | Checks | Time |
|---|---|---|
| `test_rate_params.py` | frame/hop per rate, 193 band bins at every rate, 16 kHz bins identical to original, validation errors, 8 kHz warning | seconds |
| `test_detector_net.py` | reference mel bank unchanged, exact-multiple rates give **identical** network output to 16 kHz, 44.1 kHz family close, wrong STFT size raises, per-rate mel cache | seconds |
| `test_silence_checker.py` | WebRTC VAD no longer crashes at 11.025/12/22.05/24/44.1 kHz, resamples only those, input untouched | seconds |
| `test_service.py` | unsupported rates rejected, silence gives the normal `ValueError` at every rate, detection at every rate, `AWARE(20bps)` still 16 kHz only | ~1 min |
| `test_embed_detect.py` | embed + detect at 7 rates on synthetic voiced audio: length kept, BER <= 0.05, content above 8 kHz untouched | few min |
| `test_backward_compat.py` | at 16 kHz, runs the original AWARE (commit `9f0fe88` from git history) and this version in separate processes on the same input: clean detection identical, watermarked audio identical up to the kept tail samples | ~2 min |
| `scripts/emilia_smoke.py` | 4 Emilia clips x (8 rates + native) through `embed_watermark`/`detect_watermark`: BER at same rate, after resampling to 16 kHz and to native rate, clean confidence, SNR, change above 8.5 kHz | ~30-60 min |

Emilia smoke pass criteria per rate: every output has the input length and is finite, mean BER at the embedding
rate <= 0.05, mean BER after resampling to 16 kHz <= 0.10. 8 kHz only warns (its 4 kHz band edge is at Nyquist).
Results: `results/emilia_smoke_<jobid>/emilia_smoke.csv`.
