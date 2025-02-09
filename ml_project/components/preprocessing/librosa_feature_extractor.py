# ml_project/components/preprocessing/sentence_dataset_creator.py
import logging
from pathlib import Path
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from ml_project.src.interfaces import DataPreprocessor

class LibrosaFeatureExtractor(DataPreprocessor):
    """Class to create and enrich the sentence dataset"""
    def __init__(self, participant_info_path: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.participant_info_path = participant_info_path
        self.required_columns = [
            'file', 'gender', 'duration', 'words_per_second', 'tempo',
            'f0_mean', 'f0_median', 'f0_std', 'f0_5perc', 'f0_95perc'
        ]

    def prepare_data(self, sentences_path: Path) -> pd.DataFrame:
        """Main method to create and enrich the dataset"""

        logging.info("Processing audio files...")
        raw_data = self._process_audio_files(sentences_path)
        logging.info("Creating base dataframe...")
        df = self._create_base_dataframe(raw_data)
        logging.info("Enriching with participant info...")
        return self._enrich_with_participant_info(df)

    def _process_audio_files(self, sentences_path: Path) -> Dict[str, list]:
        """Process all audio files and extract features"""

        # Initialize data dictionary with required columns
        data = {col: [] for col in self.required_columns}

        # Get all audio files in the sentences path
        files = [f for f in sentences_path.glob('*.wav') if f.is_file()]

        for file in tqdm(files, desc="Processing audio files"):
            try:
                features = self._extract_features(file)
                self._store_features(data, file, features)
            except Exception as e:
                logging.error(f"Error processing {file.name}: {str(e)}")
        
        return data

    def _extract_features(self, file_path: Path) -> Dict[str, Any]:
        """Extract audio features using Librosa"""
        logging.info(f"Extracting features from {file_path.name}")
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        duration = len(y) / sr
        words_per_second = 6 / duration  # Fixed sentence structure. Speakers always pronounce 6 words in the audio
        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        f0 = librosa.pyin(y, sr=sr, fmin=10, fmax=8000, frame_length=1024)[0]

        return {
            'duration': duration,
            'words_per_second': words_per_second,
            'tempo': tempo,
            'f0_mean': np.nanmean(f0),
            'f0_median': np.nanmedian(f0),
            'f0_std': np.nanstd(f0),
            'f0_5perc': np.nanpercentile(f0, 5),
            'f0_95perc': np.nanpercentile(f0, 95)
        }

    def _store_features(self, data: Dict[str, list], file: Path, features: Dict[str, Any]):
        """Store extracted features in the data dictionary
        
        Args:
            data (Dict[str, list]): Dictionary to store the features
            file (Path): Path to the audio file
            features (Dict[str, Any]): Extracted features
        """
        logging.info(f"Storing features for {file.name}")
        data['file'].append(str(file))

        # Extract gender from the file name
        data['gender'].append(file.name.split('_')[0].split('-')[0])

        # Store the features
        for key in features:
            data[key].append(features[key])

    def _create_base_dataframe(self, data: Dict[str, list]) -> pd.DataFrame:
        """Create and validate base dataframe
        
        Args:
            data (Dict[str, list]): Dictionary with the features
        
        Returns:
            pd.DataFrame: Base dataframe
        """
        df = pd.DataFrame(data)
        self._validate_dataframe(df)
        logging.info("Base dataframe created and validated")
        return df

    def _validate_dataframe(self, df: pd.DataFrame):
        """Ensure data consistency
        
        Args:
            df (pd.DataFrame): Dataframe to validate
        """
        if len(df) == 0:
            raise ValueError("No valid audio files processed")
        if df.isna().sum().sum() > 0:
            print("Warning: Missing values detected in dataset")

    def _enrich_with_participant_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge with participant metadata
        
        Args:
            df (pd.DataFrame): Dataframe to enrich
        
        Returns:
            pd.DataFrame: Enriched dataframe
        """
        df['Participant'] = df['file'].str.split('/').str[-1].str.split('_').str[0]
        participant_df = pd.read_csv(self.participant_info_path)
        
        merged_df = df.merge(
            participant_df,
            on='Participant',
            how='left',
            validate='one_to_one'
        )
        
        return merged_df.drop(columns=[
            'Participant', 'Sex', 'CollectionDate', 
            'Experimenter', 'RESTRICTION OF USE'
        ])
