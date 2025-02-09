from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

@dataclass
class ModelConfig:
    pipeline_steps: list
    param_grid: dict
    train_size: float = 0.8
    random_state: int = 0

@dataclass
class PreprocessingConfig:
    target_column: str = "Age_category"

@dataclass
class LoggingConfig:
    experiment_name: str
    tracking_uri: str

@dataclass
class SentenceConfig:
    sentences_path: Path
    participant_info_path: Path
    output_path: Path
    sample_rate: int = 16000  # default sample rate
    dataset_name: str = "audio_features_librosa_dataset"
    experiment_name: str = "sentence-dataset-creation"
    tracking_uri: str = "file:///Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/mlruns"
    mlflow_experiment: str = "sentence-dataset-creation"
    features_template = {
        "f0_mean": np.nan,
        "f0_median": np.nan,
        "f0_std": np.nan,
        "f0_5perc": np.nan,
        "f0_95perc": np.nan,
        "f1_mean": np.nan,
        "f1_median": np.nan,
        "f1_std": np.nan,
        "f1_5perc": np.nan,
        "f1_95perc": np.nan,
        "f2_mean": np.nan,
        "f2_median": np.nan,
        "f2_std": np.nan,
        "f2_5perc": np.nan,
        "f2_95perc": np.nan,
        "f3_mean": np.nan,
        "f3_median": np.nan,
        "f3_std": np.nan,
        "f3_5perc": np.nan,
        "f3_95perc": np.nan,
        "intensity_mean": np.nan,
        "jitter_local": np.nan,
        "shimmer_local": np.nan
    }

@dataclass
class VowelConfig:
    base_dir: Path
    eval_path: Path
    participant_path: Path
    output_dir: Path
    exclude_segments: bool = True # This config is used to exclude segments to be included in the dataset. By segments, i mean the audio segment to pronounce i, a, o.
    mlflow_tracking_uri: str = "file:///Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/mlruns"
    mlflow_experiment: str = "vowel-feature-extraction"
    features_template = {
        "f0_mean": np.nan,
        "f1_mean": np.nan,
        "f2_mean": np.nan,
        "f3_mean": np.nan,
        "intensity_mean": np.nan,
        "jitter_local": np.nan,
        "shimmer_local": np.nan
        }