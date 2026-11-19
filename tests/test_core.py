"""Model-free checks: video I/O round trip, SI/TI ordering, attacks, p-values.

    python -m unittest discover -s tests -v
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsbench.attacks import ATTACKS, apply_attack  # noqa: E402
from vsbench.metrics import log10_pvalue, score_message, si_ti  # noqa: E402
from vsbench.videoio import VideoWriter, iter_frames, probe, run_ffmpeg  # noqa: E402


def synth(path: Path, motion: str, w=320, h=180, n=30) -> Path:
    """static: same frame repeated; moving: testsrc2 with scrolling content."""
    if motion == "static":
        src = f"testsrc2=size={w}x{h}:rate=30,trim=end_frame=1,loop=loop={n}:size=1:start=0"
    else:
        src = f"testsrc2=size={w}x{h}:rate=30,scroll=horizontal=0.05"
    run_ffmpeg(["-f", "lavfi", "-i", src, "-frames:v", str(n), "-c:v", "libx264", "-crf", "10",
                "-pix_fmt", "yuv420p", str(path)])
    return path


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.static = synth(cls.d / "static.mp4", "static")
        cls.moving = synth(cls.d / "moving.mp4", "moving")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_probe_and_read(self):
        info = probe(self.moving)
        self.assertEqual((info.width, info.height), (320, 180))
        frames = np.concatenate(list(iter_frames(self.moving, batch=7)))
        self.assertEqual(frames.shape, (30, 180, 320, 3))

    def test_lossless_roundtrip(self):
        frames = np.concatenate(list(iter_frames(self.moving)))
        out = self.d / "rt.mp4"
        with VideoWriter(out, 320, 180, 30, lossless=True) as w:
            w.write(frames)
        back = np.concatenate(list(iter_frames(out)))
        # rgb->yuv420->rgb is not exact, but qp0 adds no coding loss on top of that
        self.assertLess(np.abs(back.astype(int) - frames.astype(int)).mean(), 3.0)

    def test_si_ti_orders_motion(self):
        s, m = si_ti(self.static), si_ti(self.moving)
        self.assertLess(s["ti_mean"], 0.5)
        self.assertGreater(m["ti_mean"], s["ti_mean"] + 5)
        self.assertGreater(s["si_mean"], 1)

    def test_attacks_produce_expected_sizes(self):
        big = synth(self.d / "big.mp4", "moving", w=1920, h=1080, n=8)
        expect = {"crop_area50": (1356, 762), "downscale_480p": (854, 480), "compress_crf33": (1920, 1080),
                  "combined": (854, 480)}
        for name, (w, h) in expect.items():
            out = apply_attack(big, self.d / f"{name}.mp4", ATTACKS[name])
            info = probe(out)
            self.assertEqual((info.width, info.height), (w, h), name)

    def test_pvalue(self):
        self.assertAlmostEqual(log10_pvalue(256, 256), -256 * math.log10(2), places=6)
        self.assertAlmostEqual(log10_pvalue(0, 256), 0.0)
        self.assertGreater(log10_pvalue(128, 256), -0.5)  # chance level
        self.assertLess(log10_pvalue(200, 256), -15)
        s = score_message(np.ones(256), np.ones(256))
        self.assertEqual(s["bit_acc"], 1.0)
        self.assertTrue(s["detected_fpr1e-6"])


if __name__ == "__main__":
    unittest.main()
