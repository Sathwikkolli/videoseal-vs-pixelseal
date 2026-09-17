from aware.embedding.multibit_embedder import AWAREEmbedder
from aware.utils.audio import SilenceChecker, validate_sample_rate
from aware.utils.watermark import PatternEncoder
from aware.utils.logger import logger
import numpy as np


def check_sample_rate(sample_rate: int, mode_name: str, base_sample_rate: int, embedding_bands: tuple[int, int]) -> int:
    """Validate the sample rate for a model and return it as int (raises ValueError)"""
    try:
        validate_sample_rate(sample_rate, embedding_bands)
    except ValueError as e:
        logger.error(str(e))
        raise

    sample_rate = int(sample_rate)
    # Only the default AWARE (full_length) model supports sample rates other than the base rate
    if mode_name != "full_length" and sample_rate != base_sample_rate:
        logger.error(f"Invalid sample rate. Model '{mode_name}' expects {base_sample_rate}Hz, got {sample_rate}Hz.")
        raise ValueError(f"Invalid sample rate. Model '{mode_name}' expects {base_sample_rate}Hz.")

    return sample_rate


def embed_watermark(audio: np.ndarray, sample_rate: int, watermark_bits:bytes | np.ndarray, model: AWAREEmbedder)->np.ndarray:

    """
    Embeds a given watermark in audio data and returns the watermarked audio data.

    Args:
        audio (np.ndarray): The audio data.
        sampe_rate (int): Sampling rate of the audio. The default AWARE model supports 8000-48000Hz,
            AWARE(20bps) supports 16000Hz only. The watermark is embedded at this rate (no resampling).
        watermark_bits (buytes | np.ndarray): The watermark bits (0/1) to embed.
        model: The embeder model.

    Returns:
        watermarked_audio (np.ndarray): The watermarked audio data.
    """
    sample_rate = check_sample_rate(sample_rate, model.mode_name, model.base_sample_rate, model.embedding_bands)

    pattern_preprocess_pipeline = [PatternEncoder(mode=model.pattern_mode)]
    silence_checker_pipeline = [SilenceChecker(sample_rate=sample_rate)]

    if audio.ndim != 1:
        logger.error(f"Invalid audio shape. Expected mono (1D) signal, got shape {audio.shape}.")
        raise ValueError(f"Invalid audio shape. Expected mono (1D) signal, got shape {audio.shape}.")
    

    watermark = watermark_bits
    for processor in pattern_preprocess_pipeline:
        watermark = processor(watermark)

    if len(watermark) != model.detection_net.output_length:
        logger.error(f"Invalid watermark length. Expected {model.detection_net.output_length}, got {len(watermark)}.")
        raise ValueError(f"Invalid watermark length.")
    
    if model.mode_name == "full_length":
        for process in silence_checker_pipeline:
            is_silent = process(audio)

        if is_silent == True:
            logger.error(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")
            raise ValueError(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")

        audio_mx = np.max(audio)
        
        watermarked_audio = model.embed(audio, sample_rate, watermark)

        watermarked_audio = audio_mx * watermarked_audio
        
    elif model.mode_name == "segments":        
        for process in silence_checker_pipeline:
            is_silent = process(audio)

        if is_silent == True:
            logger.error(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")
            raise ValueError(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")

        start = 0
        step = sample_rate+128
        wm_audio = []
        while start < len(audio):
            audio_ = audio[start:start+step]
            start += step

            if len(audio_) < step:
                audio_ = np.pad(audio_, (0, step - len(audio_)), mode='constant')

            audio_mx = np.max(audio_)
            
            watermarked_audio = model.embed(audio_, sample_rate, watermark)

            watermarked_audio = audio_mx * watermarked_audio

            wm_audio.append(watermarked_audio)

        wm_audio = np.concatenate(wm_audio)
        return wm_audio[:len(audio)]


    else:
        logger.error("Invalid audio shape. Expected 1D or 2D numpy array.")
        raise ValueError("Invalid audio shape. Expected 1D or 2D numpy array.")
    

    return watermarked_audio
