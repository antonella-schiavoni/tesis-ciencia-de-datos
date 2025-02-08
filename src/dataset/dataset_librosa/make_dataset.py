"""
This script makes a dataset with the features extracted from the sentences and the participant information.

Based on the following link: https://miykael.github.io/blog/2022/audio_eda_and_modeling/
"""

from pathlib import Path
import sys
import os
from tqdm import tqdm  # Import tqdm for the progress bar

# Add the project root directory to the Python path
sys.path.append('/Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/')

from feature_extraction.librosa.feature_extraction import extract_features_librosa
import librosa
import pandas as pd

def make_dataset(sentences_path: Path):
    """
    Make a dataset with the features extracted from the sentences and the participant information

    args:
        sentences_path: path to the sentences folder

    returns:
        df: dataframe with the features extracted from the sentences and the participant information
    """

    data = {
        "file": [],
        "gender": [],
        "duration": [],
        "words_per_second": [],
        "tempo": [],
        "f0_mean": [],
        "f0_median": [],
        "f0_std": [],
        "f0_5perc": [],
        "f0_95perc": []
    }

    # filter the files to only include the ones with the extension .wav
    files = [file for file in os.listdir(sentences_path) if file.endswith('.wav')]

    # Wrap os.listdir with tqdm for a progress bar
    for file in tqdm(files, desc="Processing files"):
        file_path = sentences_path / file
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=16000)

            # Extract the features
            result = extract_features_librosa(y, sr)

            # Get the gender of the participant
            gender = file.split('_')[0].split('-')[0]

            # Store the data
            data['file'].extend([file_path])
            data['gender'].extend([gender])
            data['duration'].extend([result['duration']])
            data['words_per_second'].extend([result['words_per_second']])
            data['tempo'].extend([result['tempo'][0]])
            data['f0_mean'].extend([result['f0_mean']])
            data['f0_median'].extend([result['f0_median']])
            data['f0_std'].extend([result['f0_std']])
            data['f0_5perc'].extend([result['f0_5perc']])
            data['f0_95perc'].extend([result['f0_95perc']])
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Debugging: Check if all lists have the same length
    for key, value in data.items():
        print(f"{key} has {len(value)} items")

    df = pd.DataFrame(data)

    # obtener el id del participante para poder hacer el join con el dataframe de los datos de los participantes
    df['file'] = df['file'].astype(str)  # Convert to string
    df['Participant'] = df['file'].str.split('/').str[-1].str.split('_').str[0]

    df_participants = pd.read_csv("data/raw/participant-information/DATA-GEFAV-Participant Information.csv")

    # hacer el join
    df = df.merge(df_participants, on='Participant', how='left')

    df.drop(columns=['Participant', 'Sex', 'CollectionDate', 'Experimenter', 'RESTRICTION OF USE'], inplace=True)

    return df

if __name__ == "__main__":
    sentences_path = Path('data/processed/voices_sentences/2.Hour/')
    df = make_dataset(sentences_path)
    df.to_csv('data/processed/voices_sentences/2.Hour/dataset_librosa.csv', index=False)
    print(df.head())


