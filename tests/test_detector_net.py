"""The detection network sees the same features at every sample rate"""
import unittest

import _common
import librosa
import numpy as np
import torch

N_FRAMES = 60


def spectrogram_for_rate(net_device, sample_rate):
    """Same Hz-defined spectrogram sampled on this rate's bin grid, out-of-band bins zeroed like the detector does"""
    embedder, _ = _common.get_models()
    p = embedder.get_rate_params(sample_rate)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=p.frame_length)
    magnitude = _common.smooth_spectrogram(freqs, N_FRAMES)
    magnitude[p.out_of_band_indices] = 0.0
    return torch.from_numpy(magnitude).unsqueeze(0).to(net_device)


class TestDetectorNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        embedder, _ = _common.get_models()
        cls.net = embedder.detection_net
        cls.device = next(cls.net.parameters()).device

    def forward(self, x, sample_rate=None):
        with torch.no_grad():
            return self.net(x, sample_rate).squeeze().cpu().numpy()

    def test_reference_mel_bank_unchanged(self):
        # Explicit fmax=8000 in the config gives the same filter bank as the original default (sr / 2)
        from aware.detection.modules.mel import get_mel_filter_bank
        original = get_mel_filter_bank(16000, 1024, 128, 0.0, None)
        np.testing.assert_array_equal(self.net.mel_layer.mel_filter_bank.cpu().numpy(), original.astype(np.float32))
        self.assertEqual(self.net.fmax, 8000.0)

    def test_default_rate_equals_explicit_base_rate(self):
        x = spectrogram_for_rate(self.device, 16000)
        np.testing.assert_array_equal(self.forward(x), self.forward(x, 16000))

    def test_wrong_number_of_bins_raises(self):
        x = torch.rand(1, 707, N_FRAMES, device=self.device)  # 22.05 kHz sized STFT
        with self.assertRaises(ValueError):
            self.forward(x)
        with self.assertRaises(ValueError):
            self.forward(x, 16000)

    def test_exact_rates_match_16k(self):
        reference = self.forward(spectrogram_for_rate(self.device, 16000), 16000)
        for sample_rate in _common.EXACT_RATES:
            with self.subTest(sample_rate=sample_rate):
                values = self.forward(spectrogram_for_rate(self.device, sample_rate), sample_rate)
                np.testing.assert_allclose(values, reference, atol=1e-5)

    def test_rounded_rates_close_to_16k(self):
        reference = self.forward(spectrogram_for_rate(self.device, 16000), 16000)
        for sample_rate in _common.ROUNDED_RATES:
            with self.subTest(sample_rate=sample_rate):
                values = self.forward(spectrogram_for_rate(self.device, sample_rate), sample_rate)
                cosine = np.dot(values, reference) / (np.linalg.norm(values) * np.linalg.norm(reference) + 1e-12)
                print(f"\n  {sample_rate}Hz vs 16kHz: cosine={cosine:.5f}, max|diff|={np.max(np.abs(values - reference)):.5f}")
                self.assertGreater(cosine, 0.9)

    def test_mel_bank_cache(self):
        x = spectrogram_for_rate(self.device, 22050)
        self.forward(x, 22050)
        self.forward(x, 22050)
        keys = [key for key in self.net._rate_mel_layers if key[0] == 22050]
        self.assertEqual(keys, [(22050, 1412)])
        layer = self.net._rate_mel_layers[(22050, 1412)]
        self.assertEqual(layer.mel_filter_bank.device, x.device)
        self.assertEqual(tuple(layer.mel_filter_bank.shape), (128, 707))
        self.assertIn(22050, self.net.get_model_info()["cached_sample_rates"])


if __name__ == "__main__":
    unittest.main()
