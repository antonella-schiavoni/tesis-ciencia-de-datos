# This code will be used for the dataset build for modelo_predictor_edad_3.ipynb

import librosa
import numpy as np
from numpy import ndarray


def extract_features_librosa(y: ndarray, sr: int):

    # Audio duration as duration
    duration = len(y) / sr

    # Il est deux heures moins dix
    # words_per_second. The audio is nomalized and all speakers pronounce the same sentence which contains 6 wods
    words_per_second = 6 / duration

    # Tempo as tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # F0 as f0_mean, f0_median, f0_std, f0_5perc, f0_95perc
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=10, fmax=8000, frame_length=1024)
    # Computes mean, median, 5%- and 95%-percentile value of fundamental frequency

    result = {
        "duration": duration,
        "words_per_second": words_per_second,
        "tempo": tempo,
        "f0_mean": np.nanmean(f0),
        "f0_median": np.nanmedian(f0),
        "f0_std": np.nanstd(f0),
        "f0_5perc": np.nanpercentile(f0, 5),
        "f0_95perc": np.nanpercentile(f0, 95)
    }

    return result