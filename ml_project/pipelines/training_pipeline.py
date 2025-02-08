import pandas as pd
from sklearn.model_selection import train_test_split
from components.preprocessing.age_preprocessor import AgePredictionPreprocessor
from components.models.logreg_trainer import LogisticRegressionTrainer
from logging.mlflow_logger import MLflowLogger
from config.params import ModelConfig, PreprocessingConfig, LoggingConfig

def run_pipeline(df: pd.DataFrame, 
                model_config: ModelConfig,
                preprocessing_config: PreprocessingConfig,
                logging_config: LoggingConfig):
    
    # Initialize components
    preprocessor = AgePredictionPreprocessor(
        target_column=preprocessing_config.target_column
    )
    
    model_trainer = LogisticRegressionTrainer(
        pipeline_steps=model_config.pipeline_steps,
        param_grid=model_config.param_grid
    )
    
    logger = MLflowLogger(
        experiment_name=logging_config.experiment_name,
        tracking_uri=logging_config.tracking_uri
    )

    # Execute pipeline
    X, y = preprocessor.prepare_data(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, 
        train_size=model_config.train_size,
        shuffle=True,
        stratify=y,
        random_state=model_config.random_state
    )
    
    trained_model = model_trainer.train(X_train, y_train)
    logger.log_training_metadata(trained_model, pd.DataFrame(trained_model.cv_results_))
