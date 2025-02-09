import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from datetime import datetime
from pathlib import Path
from re import I

from ml_project.config.params import SentenceConfig
from ml_project.pipelines.sentence_pipeline import run_data_pipeline
from ml_project.logging.mlflow_logger import MLflowLogger

def run_dataset_creation_pipeline():
    """
    Run the dataset creation pipeline.
    """
    # Initialize MLflow logger
    logger = MLflowLogger(
        experiment_name="dataset_creation",
        tracking_uri="file:./mlruns"  # Using local filesystem for MLflow tracking
    )
    
    config = SentenceConfig(
        sentences_path=Path("data/processed/voices_sentences/2.Hour/"),
        participant_info_path=Path("data/raw/participant-information/DATA-GEFAV-Participant Information.csv"),
        output_path=Path(f"data/processed/datasets/audio_features_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"),
        sample_rate=16000
    )

    # Pass both config and logger to the pipeline
    dataset = run_data_pipeline(config, logger=logger)

if __name__ == "__main__":
    run_dataset_creation_pipeline()