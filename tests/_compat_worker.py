"""
Runs AWARE from a given source tree at 16 kHz and saves the results (used by test_backward_compat.py).

    python _compat_worker.py <src_dir> <input.npz> <output.npz>

Each code version runs in its own process so the two `aware` packages never share an interpreter.
"""
import os
import sys

src_dir, input_path, output_path = sys.argv[1:4]
sys.path.insert(0, os.path.abspath(src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # deterministic CPU kernels, identical in both processes

import aware  # noqa: E402

if not os.path.abspath(aware.__file__).startswith(os.path.abspath(src_dir)):
    sys.exit(f"aware imported from {aware.__file__}, expected {src_dir}")

from aware.utils.models import load  # noqa: E402

models = load("AWARE")
if models is None:
    sys.exit("load('AWARE') failed")
embedder, detector = models

data = np.load(input_path)
audio, pattern = data["audio"], data["pattern"]

clean_values = detector.detect(audio, 16000)
watermarked = embedder.embed(audio, 16000, pattern)
watermarked_values = detector.detect(watermarked, 16000)

np.savez(output_path, clean_values=clean_values, watermarked=watermarked, watermarked_values=watermarked_values)
