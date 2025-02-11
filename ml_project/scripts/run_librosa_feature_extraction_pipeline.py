"""
Use LibrosaFeatureExtractor to extract features from audio files.
Audio files are stored in the /Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/data/processed/voices_sentences/2.Hour.
store the features in a csv file in the folder /Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/data/processed/datasets/librosa_features.
"""

from pathlib import Path
from ml_project.components.preprocessing.librosa_feature_extractor import LibrosaFeatureExtractor

def run_librosa_feature_extraction_pipeline():

    participant_info_path = Path('/Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/data/processed/participant_info.csv')
    sentences_path = Path('/Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/data/processed/voices_sentences/2.Hour')
    # Initialize the feature extractor
    feature_extractor = LibrosaFeatureExtractor(participant_info_path)

    # Extract features from audio files
    feature_extractor.prepare_data(sentences_path)

if __name__ == "__main__":
    run_librosa_feature_extraction_pipeline()
