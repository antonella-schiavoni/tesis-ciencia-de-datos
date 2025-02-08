# ml_project/pipelines/data_pipeline.py
from pathlib import Path
import mlflow
import pandas as pd
from components.preprocessing.audio_dataset_creator import AudioDatasetCreator
from logging.mlflow_logger import MLflowLogger
from config.params import DataConfig

def run_data_pipeline(config: DataConfig):
    """End-to-end pipeline for dataset creation and registration"""
    # Initialize components
    dataset_creator = AudioDatasetCreator(
        participant_info_path=config.participant_info_path,
        sample_rate=config.sample_rate
    )
    
    logger = MLflowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=config.tracking_uri
    )

    # Execute pipeline
    with mlflow.start_run(run_name="Dataset Creation"):
        # Create dataset
        dataset = dataset_creator.prepare_data(config.sentences_path)
        
        # Log dataset and metadata
        logger.log_dataset(dataset, config.dataset_name)
        logger.log_params({
            "audio_sample_rate": config.sample_rate,
            "participant_info_path": config.participant_info_path
        })
        
        # Save dataset locally
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(config.output_path, index=False)
        
    return dataset
