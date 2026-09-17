"""WebRTC VAD speech check works at every sample rate"""
import unittest
from unittest import mock

import _common
import numpy as np

from aware.utils.audio import SilenceChecker
from aware.utils.audio import waveform as waveform_module


class TestSilenceChecker(unittest.TestCase):

    def test_silence_detected_at_all_rates(self):
        for sample_rate in _common.ALL_RATES:
            with self.subTest(sample_rate=sample_rate):
                # used to raise inside webrtcvad for 11.025k, 12k, 22.05k, 24k, 44.1k
                self.assertTrue(SilenceChecker(sample_rate=sample_rate)(np.zeros(2 * sample_rate, dtype=np.float32)))

    def test_resamples_only_unsupported_rates(self):
        for sample_rate in _common.ALL_RATES:
            with self.subTest(sample_rate=sample_rate):
                with mock.patch.object(waveform_module.resampy, "resample", wraps=waveform_module.resampy.resample) as resample:
                    SilenceChecker(sample_rate=sample_rate)(np.zeros(sample_rate, dtype=np.float32))
                expected = sample_rate not in SilenceChecker.VAD_SAMPLE_RATES
                self.assertEqual(resample.called, expected)

    def test_does_not_modify_input(self):
        audio = _common.synthetic_voice(44100, seconds=1.0)
        original = audio.copy()
        SilenceChecker(sample_rate=44100)(audio)
        np.testing.assert_array_equal(audio, original)

    def test_out_of_range_samples_do_not_raise(self):
        audio = np.full(16000, 1.5, dtype=np.float32)
        self.assertIsInstance(SilenceChecker(sample_rate=16000)(audio), (bool, np.bool_))


if __name__ == "__main__":
    unittest.main()
