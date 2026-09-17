"""
At 16 kHz the multi-rate code must reproduce the original AWARE.

The original source is taken from git history (commit before the multi-rate change, override with
AWARE_BASELINE_REF) and both versions run in separate processes on the same input:

    - detector output on clean audio:          identical
    - watermarked audio:                       identical, except that the new version keeps the trailing
                                               samples the original ISTFT dropped (and therefore a possibly
                                               different peak normalisation, compared after a gain fit)
    - detector output on the watermarked audio: equal up to the effect of those trailing samples
"""
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import _common
import numpy as np

from aware.utils.watermark import PatternEncoder

BASELINE_REF = os.environ.get("AWARE_BASELINE_REF", "9f0fe88")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_compat_worker.py")


def extract_baseline(ref: str, destination: str) -> str:
    """Extract src/ of `ref` into `destination` and return the src path"""
    archive = os.path.join(destination, "baseline.tar")
    with open(archive, "wb") as fh:
        subprocess.run(["git", "-C", _common.REPO_ROOT, "archive", "--format=tar", ref, "src"],
                       stdout=fh, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination)
    return os.path.join(destination, "src")


def run_worker(src_dir: str, input_path: str, output_path: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run([sys.executable, WORKER, src_dir, input_path, output_path],
                            env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"worker failed for {src_dir}:\n{result.stdout}\n{result.stderr}")
    return dict(np.load(output_path))


class TestBackwardCompat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if shutil.which("git") is None:
            raise unittest.SkipTest("git not available")
        check = subprocess.run(["git", "-C", _common.REPO_ROOT, "cat-file", "-e", f"{BASELINE_REF}^{{commit}}"],
                               capture_output=True)
        if check.returncode != 0:
            raise unittest.SkipTest(f"baseline commit {BASELINE_REF} not in this clone")

        cls.tmp = tempfile.mkdtemp(prefix="aware_compat_")
        baseline_src = extract_baseline(BASELINE_REF, cls.tmp)

        # length 32037 is not a multiple of the 256 hop, so the ISTFT length fix is exercised
        cls.audio = _common.synthetic_voice(16000, seconds=2.0, extra_samples=37)
        bits = np.random.default_rng(2).integers(0, 2, size=20, dtype=np.int32)
        pattern = PatternEncoder(mode="bits2bipolar")(bits)
        input_path = os.path.join(cls.tmp, "input.npz")
        np.savez(input_path, audio=cls.audio, pattern=pattern)

        cls.baseline = run_worker(baseline_src, input_path, os.path.join(cls.tmp, "baseline.npz"))
        cls.new = run_worker(_common.SRC, input_path, os.path.join(cls.tmp, "new.npz"))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_clean_detection_identical(self):
        np.testing.assert_allclose(self.new["clean_values"], self.baseline["clean_values"], rtol=0, atol=1e-6)

    def test_output_length(self):
        hop = 256
        self.assertEqual(len(self.baseline["watermarked"]), hop * (len(self.audio) // hop))  # original drops the tail
        self.assertEqual(len(self.new["watermarked"]), len(self.audio))

    def test_watermarked_audio_matches(self):
        base = self.baseline["watermarked"].astype(np.float64)
        new = self.new["watermarked"][:len(base)].astype(np.float64)
        gain = np.dot(base, new) / np.dot(new, new)
        relative_error = np.linalg.norm(gain * new - base) / np.linalg.norm(base)
        print(f"\n  16 kHz watermarked audio vs original: gain={gain:.6f}, relative error={relative_error:.2e}")
        self.assertLess(relative_error, 1e-3)

    def test_watermarked_detection_matches(self):
        base_values, new_values = self.baseline["watermarked_values"], self.new["watermarked_values"]
        print(f"\n  16 kHz detection on watermarked audio: max|diff|={np.max(np.abs(new_values - base_values)):.2e}")
        np.testing.assert_array_equal(np.sign(new_values), np.sign(base_values))
        np.testing.assert_allclose(new_values, base_values, rtol=0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
