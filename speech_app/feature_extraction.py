import librosa
import numpy as np

def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    return np.hstack([mfccs, chroma, mel])
