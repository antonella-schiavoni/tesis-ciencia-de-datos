import mlflow
import pandas as pd
from ml_project.components.preprocessing.praat_feature_extractor import PraatFeatureExtractor
from ml_project.components.preprocessing.vowel_dataset_creator import VowelDatasetCreator

from ml_project.config.params import VowelConfig
from typing import Optional

from ml_project.logging.mlflow_logger import MLflowLogger

class VowelFeaturePipeline:
    def __init__(self, config: VowelConfig, logger: Optional[MLflowLogger] = None):
        self.config = config
        self.logger = logger or MLflowLogger(
            config.mlflow_tracking_uri,
            config.mlflow_experiment
        )
        self.feature_extractor = PraatFeatureExtractor(
            exclude_segments=config.exclude_segments,
            features_template=config.features_template
        )
        self.dataset_creator = VowelDatasetCreator(
            feature_extractor=self.feature_extractor,
            eval_path=config.eval_path,
            participant_path=config.participant_path,
            output_dir=config.output_dir
        )

    def run(self) -> pd.DataFrame:
        with mlflow.start_run():
            df = self.dataset_creator.create_dataset(self.config.base_dir)
            
            if self.logger:
                self._log_metadata(df)
                
            self._save_dataset(df)
            return df

    def _log_metadata(self, df: pd.DataFrame):
        self.logger.log_dataset(df, "vowel_features")
        mlflow.log_params({
            "exclude_segments": self.config.exclude_segments,
            "input_dir": str(self.config.base_dir),
            "output_dir": str(self.config.output_dir)
        })

    def _save_dataset(self, df: pd.DataFrame):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.config.output_dir / f"vowel_features_{timestamp}.csv"
        mlflow.log_param("output_path", str(output_path))
        df.to_csv(output_path, index=False)
