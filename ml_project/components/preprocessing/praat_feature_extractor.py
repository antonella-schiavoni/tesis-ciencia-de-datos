import logging
from pathlib import Path
import parselmouth
from parselmouth.praat import call # type: ignore
import pandas as pd
import numpy as np

from typing import Dict

from ml_project.src.interfaces import FeatureExtractor

class PraatFeatureExtractor(FeatureExtractor):
    def __init__(self, exclude_segments: bool = False, features_template: Dict = None):
        self.exclude_segments = exclude_segments
        self.feature_template = features_template

    def extract_features(self, file_path: Path, label: str) -> Dict:
        features = {
            "file": file_path.name,
            "label": label,
            **self.feature_template.copy()
        }
        
        try:
            sound = parselmouth.Sound(str(file_path))
            features.update(self._extract_acoustic_features(sound))
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {str(e)}")
            
        return features

    def _extract_acoustic_features(self, sound) -> Dict:
        features = {}
        
        # Pitch analysis
        try:
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            features["mean_pitch"] = call(pitch, "Get mean", 0, 0, "Hertz")
        except Exception as e:
            print(f"Pitch extraction error: {str(e)}")

        # Formant analysis
        try:
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            for i in range(1, 4):
                features[f"mean_f{i}"] = call(formant, "Get mean", i, 0, 0, "Hertz")
        except Exception as e:
            print(f"Formant extraction error: {str(e)}")

        # Intensity analysis
        try:
            intensity = call(sound, "To Intensity", 75, 0.0)
            features["mean_intensity"] = call(intensity, "Get mean", 0, 0, "dB")
        except Exception as e:
            print(f"Intensity extraction error: {str(e)}")

        # Jitter/Shimmer analysis
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            features["local_jitter"] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features["local_shimmer"] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception as e:
            print(f"Jitter/Shimmer extraction error: {str(e)}")

        return features
