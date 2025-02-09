"""
Purpose:
This script runs the vowel dataset creation pipeline. It reads raw input files, processes them to extract vowel features, and saves the resulting dataset. During execution, it also logs various aspects of the pipeline using MLflow.

Inputs:
Configuration via VowelConfig:
base_dir: The base directory for processed vowel segmentation (e.g., "data/processed/vowel_segmentation_v2_2024_06_01").
eval_path: Path to the evaluation file (e.g., "data/raw/evaluation/DATA_GEFAV_EVAL.CSV").
participant_path: Path to the participant information file (e.g., "data/raw/participant-information/DATA-GEFAV-Participant Information.csv").
output_dir: Directory where the final processed vowel features dataset will be stored (e.g., "data/processed/datasets/vowel_features").

Pipeline Module:
The script instantiates the VowelFeaturePipeline using the above configuration, which internally handles the data loading, processing, and feature extraction.

MLflow Logger:
It initializes an MLflowLogger to track the experiment (named "vowel_dataset_creation") and record run details using a local MLflow tracking server.

Outputs:
Processed Dataset CSV:
After execution, the pipeline produces a CSV file containing the extracted vowel features. This file is saved into the directory indicated by output_dir (with a filename that includes a timestamp).

Console Output:
The script prints a message to the console indicating the number of samples created and the output directory.
"""

import sys
import os

from ml_project.logging.mlflow_logger import MLflowLogger
from ml_project.pipelines.vowel_pipeline import VowelFeaturePipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pathlib import Path
from ml_project.config.params import VowelConfig


def main():

    # Initialize MLflow logger
    logger = MLflowLogger(
        experiment_name="vowel_dataset_creation",
        tracking_uri="file:./mlruns"  # Using local filesystem for MLflow tracking
    )

    config = VowelConfig(
        base_dir=Path("data/processed/vowel_segmentation_v2_2024_06_01"),
        eval_path=Path("data/raw/evaluation/DATA_GEFAV_EVAL.CSV"),
        participant_path=Path("data/raw/participant-information/DATA-GEFAV-Participant Information.csv"),
        output_dir=Path("data/processed/datasets/vowel_features")
    )

    pipeline = VowelFeaturePipeline(config=config, logger=logger)
    df = pipeline.run()
    print(f"Created dataset with {len(df)} samples at {config.output_dir}")

if __name__ == "__main__":
    main()
