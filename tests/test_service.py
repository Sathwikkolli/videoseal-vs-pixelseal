"""Sample rate validation in embed_watermark / detect_watermark"""
import unittest

import _common
import numpy as np

from aware.service import embed_watermark, detect_watermark


class TestService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedder, cls.detector = _common.get_models("AWARE")
        cls.bits = np.random.default_rng(0).integers(0, 2, size=20, dtype=np.int32)

    def test_unsupported_rates_rejected(self):
        audio = _common.synthetic_voice(16000, seconds=1.0)
        for sample_rate in (7999, 48001, 96000, 22050.5):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    embed_watermark(audio, sample_rate, self.bits, self.embedder)
                with self.assertRaises(ValueError):
                    detect_watermark(audio, sample_rate, self.detector)

    def test_silence_rejected_with_value_error_at_all_rates(self):
        # Before the VAD fix, unsupported rates raised a webrtcvad error instead of the ValueError
        for sample_rate in _common.ALL_RATES:
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaisesRegex(ValueError, "speach"):
                    embed_watermark(np.zeros(sample_rate, dtype=np.float32), sample_rate, self.bits, self.embedder)

    def test_detect_returns_bits_and_confidence_at_all_rates(self):
        for sample_rate in _common.ALL_RATES:
            with self.subTest(sample_rate=sample_rate):
                bits, confidence = detect_watermark(_common.synthetic_voice(sample_rate, seconds=1.0), sample_rate, self.detector)
                self.assertEqual(len(bits), 20)
                self.assertTrue(set(np.unique(bits)) <= {0, 1})
                self.assertTrue(0.0 <= confidence <= 1.0)

    def test_20bps_requires_16k(self):
        embedder, detector = _common.get_models("AWARE(20bps)")
        audio = _common.synthetic_voice(22050, seconds=2.0)
        with self.assertRaisesRegex(ValueError, "expects 16000Hz"):
            embed_watermark(audio, 22050, self.bits, embedder)
        with self.assertRaisesRegex(ValueError, "expects 16000Hz"):
            detect_watermark(audio, 22050, detector)
        # still runs at 16 kHz
        result = detect_watermark(_common.synthetic_voice(16000, seconds=2.0), 16000, detector)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
