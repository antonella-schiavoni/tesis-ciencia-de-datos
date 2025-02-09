import sys
import os

from ml_project.pipelines.vowel_pipeline import VowelFeaturePipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pathlib import Path
from ml_project.config.params import VowelConfig


def main():
    config = VowelConfig(
        base_dir=Path("data/processed/vowel_segmentation_v2_2024_06_01"),
        eval_path=Path("data/raw/evaluation/DATA_GEFAV_EVAL.CSV"),
        participant_path=Path("data/raw/participant-information/DATA-GEFAV-Participant Information.csv"),
        output_dir=Path("data/processed/datasets/vowel_features")
    )

    pipeline = VowelFeaturePipeline(config)
    df = pipeline.run()
    print(f"Created dataset with {len(df)} samples at {config.output_dir}")

if __name__ == "__main__":
    main()
