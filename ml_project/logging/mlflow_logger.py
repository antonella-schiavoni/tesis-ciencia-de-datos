from pathlib import Path
import mlflow
import pandas as pd
from ml_project.src.interfaces import ExperimentLogger
from sklearn.model_selection import GridSearchCV

class MLflowLogger(ExperimentLogger):
    def __init__(self, experiment_name: str, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = mlflow.tracking.MlflowClient()
        
    def log_training_metadata(self, model: GridSearchCV, cv_results: pd.DataFrame):
        with mlflow.start_run():
            self._log_model(model)
            self._log_cv_results(cv_results)
            self._log_best_params(model)
    
    def _log_model(self, model: GridSearchCV):
        mlflow.sklearn.log_model(
            sk_model=model.best_estimator_,
            artifact_path="model",
            input_example=model.best_estimator_._final_estimator.input_example
        )
    
    def _log_cv_results(self, cv_results: pd.DataFrame):
        for i, params in enumerate(cv_results['params']):
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_test_score", cv_results['mean_test_score'][i])
    
    def _log_best_params(self, model: GridSearchCV):
        mlflow.log_metric("best_score", model.best_score_)
        for param, value in model.best_params_.items():
            mlflow.log_param(param, value)
    
    def log_dataset(self, dataset: pd.DataFrame, dataset_name: str, 
                   description: str = "Processed audio features dataset"):
        """Log dataset as MLflow artifact"""
        with mlflow.start_run(nested=True):
            mlflow.log_text(description, f"{dataset_name}-description.txt")
            mlflow.log_param("num_samples", len(dataset))
            mlflow.log_param("num_features", len(dataset.columns))
            mlflow.log_param("features_names", dataset.columns.tolist())
            
            # Log summary statistics
            stats = dataset.describe().to_dict()
            mlflow.log_dict(stats, f"{dataset_name}-stats.json")
            
            # Log actual dataset
            temp_path = Path("temp") / f"{dataset_name}.parquet"
            temp_path.parent.mkdir(exist_ok=True)
            dataset.to_parquet(temp_path)
            mlflow.log_artifact(temp_path)
            temp_path.unlink()

    def log_params(self, params):
        """Log parameters to MLflow
        
        Args:
            params (dict): Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics):
        """Log metrics to MLflow
        
        Args:
            metrics (dict): Dictionary of metrics to log
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            
    def log_artifact(self, local_path):
        """Log an artifact (file) to MLflow
        
        Args:
            local_path (str): Path to the file to log
        """
        mlflow.log_artifact(local_path)
