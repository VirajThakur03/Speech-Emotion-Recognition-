# feature_visualizer.py

import librosa
import numpy as np
import os


def display_waveform_ascii(file_path, width=80):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Normalize waveform for better visual contrast
    y = y / np.max(np.abs(y))
    
    # Resample to fixed width
    resampled = librosa.util.utils.frame(y, frame_length=len(y) // width, hop_length=len(y) // width)
    waveform = np.mean(np.abs(resampled), axis=0)
    
    # Print ASCII waveform
    print("\nASCII Waveform:\n")
    for amp in waveform:
        bar = "█" * int(amp * 50)
        print(f"{bar}")

# Example usage
file_path = r"C:\Users\ROHAN\Desktop\ravdess\Actor_01\03-01-01-01-01-01-01.wav"
display_waveform_ascii(file_path)


def ascii_visualize(feature, title, max_width=80):
    print(f"\n=== {title} ===")
    norm_feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-6)
    step = max(1, feature.shape[1] // max_width)
    sampled = norm_feature[:, ::step]

    for row in sampled[::-1]:
        line = ''.join(['█' if val > 0.75 else
                        '▓' if val > 0.5 else
                        '▒' if val > 0.25 else
                        '░' if val > 0.1 else
                        ' ' for val in row])
        print(line[:max_width])

def extract_and_visualize(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    ascii_visualize(mfccs, "MFCC")
    ascii_visualize(mel, "Mel Spectrogram")
    ascii_visualize(chroma, "Chroma")

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\ROHAN\Desktop\ravdess\Actor_01\03-01-01-01-01-02-01.wav"

    if os.path.exists(file_path):
        extract_and_visualize(file_path)
    else:
        print(f"❌ File not found: {file_path}")
