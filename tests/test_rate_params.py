"""Derived STFT parameters, embedding band bins and sample rate validation (fast, no model needed)"""
import unittest
from unittest import mock

import _common
import librosa
import numpy as np

from aware.utils.audio import get_rate_params, validate_sample_rate
from aware.utils.audio import rate as rate_module

# sample_rate -> (frame_length, hop_length) for the default AWARE config
EXPECTED = {
    8000: (512, 128),
    11025: (706, 176),
    12000: (768, 192),
    16000: (1024, 256),
    22050: (1412, 353),
    24000: (1536, 384),
    32000: (2048, 512),
    44100: (2822, 706),
    48000: (3072, 768),
}


def params(sample_rate, frame_length=_common.BASE_FRAME_LENGTH, hop_length=_common.BASE_HOP_LENGTH,
           bands=_common.EMBEDDING_BANDS):
    return get_rate_params(sample_rate, _common.BASE_SAMPLE_RATE, frame_length, hop_length, frame_length, bands)


class TestRateParams(unittest.TestCase):

    def test_base_rate_matches_original_selection(self):
        # Original AWARE: bins with lo <= f <= hi at 16 kHz, for both shipped configs
        for frame_length, hop_length, bands in [(1024, 256, (1000, 4000)), (512, 128, (1000, 5000))]:
            with self.subTest(frame_length=frame_length):
                p = params(16000, frame_length, hop_length, bands)
                freqs = librosa.fft_frequencies(sr=16000, n_fft=frame_length)
                mask = (freqs >= bands[0]) & (freqs <= bands[1])
                self.assertEqual((p.frame_length, p.win_length, p.hop_length), (frame_length, frame_length, hop_length))
                np.testing.assert_array_equal(p.band_indices, np.where(mask)[0])
                np.testing.assert_array_equal(p.out_of_band_indices, np.where(~mask)[0])

    def test_scaled_lengths(self):
        for sample_rate, (frame_length, hop_length) in EXPECTED.items():
            with self.subTest(sample_rate=sample_rate):
                p = params(sample_rate)
                self.assertEqual(p.frame_length, frame_length)
                self.assertEqual(p.win_length, frame_length)
                self.assertEqual(p.hop_length, hop_length)
                self.assertLessEqual(p.hop_length, p.frame_length // 2)
                # ~15.6 Hz bins and ~16 ms hops at every rate
                self.assertAlmostEqual(p.bin_width, 15.625, delta=0.02)
                self.assertAlmostEqual(1000 * p.hop_length / sample_rate, 16.0, delta=0.05)

    def test_band_bins_same_as_16k(self):
        for sample_rate in _common.ALL_RATES:
            with self.subTest(sample_rate=sample_rate):
                p = params(sample_rate)
                n_bins = p.frame_length // 2 + 1
                self.assertEqual(len(p.band_indices), 193)
                # band and out-of-band bins partition all bins
                self.assertEqual(len(p.band_indices) + len(p.out_of_band_indices), n_bins)
                self.assertEqual(len(np.intersect1d(p.band_indices, p.out_of_band_indices)), 0)
                # band edges within half a bin of 1000 / 4000 Hz
                self.assertLessEqual(abs(p.band_indices[0] * p.bin_width - 1000), p.bin_width / 2)
                self.assertLessEqual(abs(p.band_indices[-1] * p.bin_width - 4000), p.bin_width / 2)
                # contiguous
                np.testing.assert_array_equal(np.diff(p.band_indices), 1)

    def test_cached(self):
        self.assertIs(params(44100), get_rate_params(44100.0, 16000, 1024, 256, 1024, [1000, 4000]))

    def test_validation_errors(self):
        for sample_rate in (7999, 48001, 96000, 22050.5, True, 0, -16000):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    validate_sample_rate(sample_rate, (1000, 4000))
        # 20bps band does not fit at 8 kHz
        with self.assertRaises(ValueError):
            validate_sample_rate(8000, (1000, 5000))

    def test_validation_accepts_supported_rates(self):
        for sample_rate in _common.ALL_RATES + (np.int64(22050), 44100.0):
            with self.subTest(sample_rate=sample_rate):
                validate_sample_rate(sample_rate, (1000, 4000))

    def test_warning_only_near_nyquist(self):
        with mock.patch.object(rate_module.logger, "warning") as warning:
            validate_sample_rate(8000, (1000, 4000))
            self.assertEqual(warning.call_count, 1)
        with mock.patch.object(rate_module.logger, "warning") as warning:
            for sample_rate in _common.ALL_RATES[1:]:
                validate_sample_rate(sample_rate, (1000, 4000))
            warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
