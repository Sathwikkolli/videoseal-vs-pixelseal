from aware.interfaces.audio import BaseAudioProcessor
import torch
import numpy as np
import webrtcvad
import resampy
from typing import Any, List, Union

class WaveformNormalizer(BaseAudioProcessor):
    """
    Normalize waveform to [-1, 1]
    
    Args:
        data: torch.Tensor of shape (batch_size, channels, time)

    Returns:
        torch.Tensor of shape (batch_size, channels, time)
    """
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data/torch.max(torch.abs(data)+1e-8)
    

class SilenceChecker(BaseAudioProcessor):

    # webrtcvad only accepts these sample rates
    VAD_SAMPLE_RATES = (8000, 16000, 32000, 48000)
    VAD_FALLBACK_SAMPLE_RATE = 16000

    def __init__(self, sample_rate = 16000, aggr = 3, frame_ms = 30.0, min_speech_seconds = 0.01):
        self.sample_rate = sample_rate
        self.aggr = aggr
        self.frame_ms = frame_ms
        self.min_speech_seconds = min_speech_seconds


    def __call__(self, data: np.ndarray) -> bool:
        audio = data
        sample_rate = self.sample_rate

        if sample_rate not in self.VAD_SAMPLE_RATES:
            # Only this temporary copy is resampled for the speech decision, the watermarked audio keeps its rate.
            # WebRTC VAD analyses content below 4 kHz, which survives resampling to 16 kHz.
            audio = resampy.resample(np.asarray(audio, dtype=np.float64), sample_rate, self.VAD_FALLBACK_SAMPLE_RATE, filter='kaiser_fast')
            sample_rate = self.VAD_FALLBACK_SAMPLE_RATE

        # Clip first: values outside [-1, 1] would wrap around in int16
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        vad = webrtcvad.Vad(self.aggr)

        bytes_per_frame = int(sample_rate * self.frame_ms / 1000) * 2
        frames = [pcm[i:i+bytes_per_frame] for i in range(0, len(pcm), bytes_per_frame)
                if len(pcm[i:i+bytes_per_frame]) == bytes_per_frame]

        voiced_flags = [vad.is_speech(f, sample_rate) for f in frames]
        voiced_count = sum(voiced_flags)
        speech_seconds = voiced_count * (self.frame_ms / 1000.0)
        
        return speech_seconds < self.min_speech_seconds
