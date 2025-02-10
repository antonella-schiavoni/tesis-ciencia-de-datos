from pathlib import Path
from typing import Dict

import librosa
import numpy as np

from ml_project.src.interfaces import FeatureExtractor


class MelFeatureExtractor(FeatureExtractor):
    """Mel spectrogram feature extractor extending base class"""
    
    def __init__(self, sample_rate: int = 16000, 
                 max_duration: float = 3.0,
                 hop_length: int = 512):
        """Initialize MelFeatureExtractor"""
        super().__init__(sample_rate)
        self.max_duration = max_duration
        self.hop_length = hop_length

    def extract_features(self, file_path: Path) -> Dict:
        """Extract Mel spectrogram features"""
        features = {}
        audio_data = self._safe_load_audio(file_path)
        
        if audio_data:
            y, sr = audio_data
            features.update(self._compute_mel_features(y, sr))
            
        return features

    def _compute_mel_features(self, y: np.ndarray, sr: int) -> Dict:
        """Compute Mel-specific features"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Estimate the desired length of the spectrogram
            length = int(self.max_duration * sr / self.hop_length)
            
            return {
                "mel_spectrogram": self._resize_spectrogram(mel_db, length), # Put mel spectrogram into the right shape
                "mel_strength": np.mean(mel_db, axis=1), # Compute mean strength per frequency for mel spectrogram
            }
        except Exception as e:
            self.logger.error(f"Mel feature error: {str(e)}")
            return {}

    def _resize_spectrogram(self, spec: np.ndarray, length: int, factor: float = -80.0) -> np.ndarray:
        """Resize spectrogram to fixed duration"""
        # Create an empty canvas to put spectrogram into
        canvas = np.ones((len(spec), length)) * factor

        if spec.shape[1] <= length:
            canvas[:, : spec.shape[1]] = spec
        else:
            canvas[:, :length] = spec[:, :length]
        return canvas
