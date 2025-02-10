import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from datetime import datetime
from pathlib import Path
from re import I
from ml_project.pipelines.sentence_pipeline import SentencePipeline
from ml_project.config.params import SentenceConfig
from ml_project.logging.mlflow_logger import MLflowLogger

def run_sentence_dataset_creation_pipeline():
    """
    Run the dataset creation pipeline.
    """
    # Initialize MLflow logger
    logger = MLflowLogger(
        experiment_name="sentence_dataset_creation",
        tracking_uri="file:./mlruns"  # Using local filesystem for MLflow tracking
    )
    
    config = SentenceConfig(
        base_dir=Path("data/processed/voices_sentences/2.Hour/"),
        eval_path=Path("data/raw/evaluation/DATA_GEFAV_EVAL.CSV"),
        participant_path=Path("data/raw/participant-information/DATA-GEFAV-Participant Information.csv"),
        output_dir=Path(f"data/processed/datasets/sentence_features/sentence_features_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"),
        sample_rate=16000
    )

    pipeline = SentencePipeline(config=config, logger=logger)
    df = pipeline.run()
    print(f"Created dataset with {len(df)} samples at {config.output_dir}")

if __name__ == "__main__":
    run_sentence_dataset_creation_pipeline()
