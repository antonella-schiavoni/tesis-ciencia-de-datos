import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.interfaces import ModelTrainer

class LogisticRegressionTrainer(ModelTrainer):
    def __init__(self, pipeline_steps: list, param_grid: dict):
        self.pipeline = Pipeline(pipeline_steps)
        self.param_grid = param_grid
        
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> GridSearchCV:
        grid_cv = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=4,
            return_train_score=True,
            verbose=1
        )
        return grid_cv.fit(X_train, y_train)
