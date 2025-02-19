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
    base_dir: Path
    eval_path: Path
    participant_path: Path
    output_dir: Path
    sample_rate: int = 16000  # default sample rate
    dataset_name: str = "sentence_features_librosa_dataset"
    experiment_name: str = "sentence-dataset-creation"
    mlflow_tracking_uri: str = "file:///Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/mlruns"
    mlflow_experiment: str = "sentence-dataset-creation"
    exclude_segments: bool = True
    features_template = {
        "f0_mean": np.nan,
        "f1_mean": np.nan,
        "f2_mean": np.nan,
        "f3_mean": np.nan,
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