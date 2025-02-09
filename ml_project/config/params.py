from dataclasses import dataclass
from pathlib import Path

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
class DataConfig:
    sentences_path: Path
    participant_info_path: Path
    output_path: Path
    sample_rate: int = 16000  # default sample rate
    dataset_name: str = "audio_features_librosa_dataset"
    experiment_name: str = "dataset-creation"
    tracking_uri: str = "file:///Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/mlruns"
    mlflow_experiment: str = "dataset-creation"

@dataclass
class VowelConfig:
    base_dir: Path
    eval_path: Path
    participant_path: Path
    output_dir: Path
    exclude_segments: bool = True # This config is used to exclude segments to be included in the dataset. By segments, i mean the audio segment to pronounce i, a, o.
    mlflow_tracking_uri: str = "file:///Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/mlruns"
    mlflow_experiment: str = "vowel-feature-extraction"