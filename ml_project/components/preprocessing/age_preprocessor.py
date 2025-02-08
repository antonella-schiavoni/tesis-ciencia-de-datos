import pandas as pd
import numpy as np
from src.interfaces import DataPreprocessor

class AgePredictionPreprocessor(DataPreprocessor):
    def __init__(self, target_column: str = "Age_category"):
        self.target_column = target_column
        
    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        features = self._extract_features(df)
        y = self._extract_target(df)
        return self._prepare_feature_matrix(features), y
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation details
        return pd.DataFrame(np.random.rand(len(df), 5))
        
    def _extract_target(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.target_column].values
        
    def _prepare_feature_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.astype(np.float64)
        features.columns = features.columns.astype(str)
        return features
