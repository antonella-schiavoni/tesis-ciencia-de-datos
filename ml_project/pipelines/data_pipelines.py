from ml_project.components.preprocessing.librosa_feature_extractor import AudioDatasetCreator
from ml_project.logging.mlflow_logger import MLflowLogger
from ml_project.config.params import SentenceConfig

def run_data_pipeline(config: SentenceConfig, logger: MLflowLogger = None):
    """Run the data pipeline to create the dataset
    
    Args:
        config (DataConfig): Configuration for dataset creation
        logger (MLflowLogger, optional): MLflow logger instance
    
    Returns:
        pd.DataFrame: The created dataset
    """
    # Create dataset
    dataset_creator = AudioDatasetCreator(
        participant_info_path=config.participant_info_path,
        sample_rate=config.sample_rate
    )
    dataset = dataset_creator.prepare_data(config.sentences_path)
    
    # Log dataset if logger is provided
    if logger:
        logger.log_dataset(
            dataset=dataset,
            dataset_name="audio_features",
            description="Dataset containing audio features extracted from voice recordings"
        )
        
        # Log configuration parameters
        logger.log_params({
            "sentences_path": str(config.sentences_path),
            "participant_info_path": str(config.participant_info_path),
            "output_path": str(config.output_path),
            "num_samples": len(dataset),
            "num_features": len(dataset.columns)
        })
    
    # Save dataset locally as well
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.output_path, index=False)
    
    return dataset