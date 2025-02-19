from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

class DataPreprocessor(ABC):
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        pass

class ModelTrainer(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> GridSearchCV:
        pass

class ExperimentLogger(ABC):
    @abstractmethod
    def log_training_metadata(self, model: GridSearchCV, cv_results: pd.DataFrame):
        pass

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, file_path: Path, label: str) -> Dict:
        pass

class DatasetCreator(ABC):
    @abstractmethod
    def create_dataset(self, base_dir: Path) -> pd.DataFrame:
        pass

class DataLogger(ABC):
    @abstractmethod
    def log_dataset(self, dataset: pd.DataFrame, dataset_name: str):
        pass