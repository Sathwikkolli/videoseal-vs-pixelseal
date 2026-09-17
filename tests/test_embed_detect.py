"""
Embedding + detection at the native sample rate on a synthetic voiced signal.

Calls the embedder/detector directly (the service-level speech check is covered by the Emilia smoke test,
because WebRTC VAD does not reliably accept synthetic signals). Takes a few minutes on CPU.
"""
import unittest

import _common
import numpy as np

from aware.utils.watermark import PatternEncoder

RATES = (8000, 11025, 16000, 22050, 24000, 44100, 48000)


class TestEmbedDetect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedder, cls.detector = _common.get_models()
        cls.results = {}
        bits = np.random.default_rng(1).integers(0, 2, size=20, dtype=np.int32)
        cls.bits = bits
        pattern = PatternEncoder(mode=cls.embedder.pattern_mode)(bits)

        print()
        for sample_rate in RATES:
            audio = _common.synthetic_voice(sample_rate)
            watermarked = cls.embedder.embed(audio, sample_rate, pattern)
            values = cls.detector.detect(watermarked, sample_rate)
            clean_values = cls.detector.detect(audio, sample_rate)
            cls.results[sample_rate] = (audio, watermarked, values, clean_values)
            ber = np.mean((values > 0).astype(np.int32) != bits)
            print(f"  {sample_rate:>5}Hz: BER={ber:.3f} mean|act| wm={np.mean(np.abs(values)):.4f} "
                  f"clean={np.mean(np.abs(clean_values)):.4f}")

    def test_output_shape_and_values(self):
        for sample_rate, (audio, watermarked, _, _) in self.results.items():
            with self.subTest(sample_rate=sample_rate):
                self.assertEqual(watermarked.shape, audio.shape)
                self.assertTrue(np.all(np.isfinite(watermarked)))
                self.assertGreater(np.max(np.abs(watermarked)), 0.0)

    def test_watermark_recovered_at_same_rate(self):
        for sample_rate, (_, _, values, _) in self.results.items():
            with self.subTest(sample_rate=sample_rate):
                ber = np.mean((values > 0).astype(np.int32) != self.bits)
                self.assertLessEqual(ber, 0.05)

    def test_watermarked_activation_above_clean(self):
        for sample_rate, (_, _, values, clean_values) in self.results.items():
            with self.subTest(sample_rate=sample_rate):
                self.assertGreater(np.mean(np.abs(values)), np.mean(np.abs(clean_values)))

    def test_content_above_8khz_preserved(self):
        for sample_rate, (audio, watermarked, _, _) in self.results.items():
            if sample_rate <= 16000:
                continue
            with self.subTest(sample_rate=sample_rate):
                change_db = _common.high_band_change_db(audio, watermarked, sample_rate)
                print(f"\n  {sample_rate}Hz change above 8.5 kHz: {change_db:.1f} dB")
                self.assertLess(change_db, -20.0)


if __name__ == "__main__":
    unittest.main()
