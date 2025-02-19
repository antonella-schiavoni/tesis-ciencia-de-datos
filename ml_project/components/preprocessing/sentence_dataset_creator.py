from pathlib import Path
import pandas as pd
import re

from typing import List, Dict

from ml_project.src.interfaces import DatasetCreator, FeatureExtractor


class SentenceDatasetCreator(DatasetCreator):
    def __init__(self, 
                 feature_extractor: FeatureExtractor,
                 eval_path: Path,
                 participant_path: Path,
                 output_dir: Path):
        self.feature_extractor = feature_extractor
        self.eval_path = eval_path
        self.participant_path = participant_path
        self.output_dir = output_dir
        self.file_pattern = re.compile(r'^[FM]-\d+_VoiceSentence2(Hour).wav')

    def create_dataset(self, base_dir: Path) -> pd.DataFrame:
        raw_data = self._process_audio_files_in_directory(base_dir)
        df = self._create_base_df(raw_data)
        df = self._enrich_with_eval_data(df)
        df = self._enrich_with_participant_data(df)
        return self._finalize_dataset(df)
        
    def _process_audio_files_in_directory(self, base_dir: Path) -> List[Dict]:
        """
        The audio files are in a directory with the following structure:
        - base_dir
            - wav_file.wav
        """
        data = []
        for wav_file in self._get_valid_files(base_dir):
            if wav_file.suffix == '.wav':
                label = wav_file.stem.split('_')[0]
                features = self.feature_extractor.extract_features(file_path=wav_file, label=label)
                data.append(features)
        return data

    def _get_valid_files(self, folder: Path):
        return [
            f for f in folder.glob('*.wav') 
            if not (self.feature_extractor.exclude_segments and 'segment' in f.name)
        ]

    def _create_base_df(self, data: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df['sample_name'] = df['file'].str.replace('_VoiceSentence2(Hour).wav', '', regex=False)
        return df

    def _enrich_with_eval_data(self, df: pd.DataFrame) -> pd.DataFrame:
        eval_df = pd.read_csv(self.eval_path, sep='\t')
        eval_df['sample_name'] = eval_df['SEX'] + '-' + eval_df['DONOR'].astype(str)
        eval_df['sample_name'] = eval_df['sample_name'].replace({r'^FO': 'F', r'^H': 'M'}, regex=True)
        return pd.merge(df, eval_df, on='sample_name', how='left')

    def _enrich_with_participant_data(self, df: pd.DataFrame) -> pd.DataFrame:
        participant_df = pd.read_csv(self.participant_path)[['Participant', 'Age', 'Sex']]
        # Make sure participant age is numeric
        participant_df['Age'] = pd.to_numeric(participant_df['Age'], errors='coerce')
        participant_df['label'] = participant_df['Participant']
        return pd.merge(df, participant_df, on='label', how='left')

    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cleanup operations
        df = df.drop(columns=['SEX', 'DONOR', 'stimulussex', 'Participant'])
        df = df[[c for c in df.columns if 'Face' not in c and 'Video' not in c]]
        df.columns = df.columns.str.lower()
        return df
