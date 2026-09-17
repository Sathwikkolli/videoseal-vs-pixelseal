import torch
import numpy as np
from aware.interfaces.detection import BaseDetectorNet, BaseDetector
from aware.utils.utils import to_tensor
from aware.utils.audio import WaveformNormalizer, STFTDecomposer, STFT, RateParams, get_rate_params


class AWAREDetector(BaseDetector):
    """
    `frame_length`, `hop_length` and `win_length` are the base values for `model.sample_rate`
    (16 kHz by default). For other sample rates they are scaled by `get_rate_params`.
    """
    def __init__(self, model: BaseDetectorNet, threshold: float = 0.0, frame_length: int = 1024, hop_length: int = 256, window: str = "hann", win_length: int = 1024, pattern_mode: str = "bits2bipolar", embedding_bands: tuple[int, int] = (500, 4000), mode_name: str = "full_length"):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_length = model.output_length
        self.pattern_mode = pattern_mode

        self.embedding_bands = embedding_bands

        self.win_length = win_length
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window

        self.mode_name = mode_name

        self.detection_net = model
        self.detection_net.eval().to(self.device)
        self.base_sample_rate = model.sample_rate

        # Base-rate pipeline, per-rate pipelines are built in `_get_pipeline`
        self.audio_preprocess_pipeline = [WaveformNormalizer(), STFT(frame_length, hop_length, window, win_length), STFTDecomposer()]
        self._rate_pipelines = {self.base_sample_rate: self.audio_preprocess_pipeline}

    def get_rate_params(self, sample_rate: int) -> RateParams:
        """STFT parameters and embedding band indices for `sample_rate`"""
        return get_rate_params(sample_rate, self.base_sample_rate, self.frame_length, self.hop_length,
                               self.win_length, self.embedding_bands)

    def _get_pipeline(self, params: RateParams) -> list:
        pipeline = self._rate_pipelines.get(params.sample_rate)
        if pipeline is None:
            pipeline = [WaveformNormalizer(), STFT(params.frame_length, params.hop_length, self.window, params.win_length), STFTDecomposer()]
            self._rate_pipelines[params.sample_rate] = pipeline
        return pipeline

    def detect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | bytes:
        params = self.get_rate_params(sample_rate)

        x = to_tensor(audio).to(self.device)
        for processor in self._get_pipeline(params):
            x = processor(x)
        magnitude, _ = x

        magnitude[params.out_of_band_indices] = 0.0

        magnitude_tensor = magnitude.unsqueeze(0)
        detected_values = self.detection_net(magnitude_tensor, params.sample_rate).squeeze().detach().cpu().numpy()

        return detected_values

