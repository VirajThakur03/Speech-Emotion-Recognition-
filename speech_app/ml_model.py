import librosa
import numpy as np
import subprocess

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def convert_audio(file_path):
    converted_path = file_path.replace(".wav", "_converted.wav")
    try:
        command = f'ffmpeg -y -i "{file_path}" -acodec pcm_s16le -ar 22050 -ac 1 "{converted_path}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg stderr:", result.stderr)  # <-- log error output
            raise ValueError(f"FFmpeg conversion failed: {result.stderr}")
        return converted_path
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error converting audio file: {file_path}. Reason: {str(e)}")


def extract_features(file_path, max_len=228):
    file_path = convert_audio(file_path)
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10).T

    def fix_length(feat):
        if feat.shape[0] < max_len:
            pad_width = max_len - feat.shape[0]
            return np.pad(feat, ((0, pad_width), (0, 0)), mode='constant')
        return feat[:max_len, :]

    mfcc = fix_length(mfcc)
    chroma = fix_length(chroma[:, :10])
    mel = fix_length(mel)

    combined = np.concatenate((mfcc, chroma, mel), axis=1)  # Currently less than 260

    # Pad to 260 features per timestep
    current_features = combined.shape[1]
    if current_features < 260:
        pad_width = 260 - current_features
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')

    print(f"Padded feature shape: {combined.shape}")  # Should be (228, 260)
    return np.expand_dims(combined, axis=0)


def predict_emotion(file_path, model):
    features = extract_features(file_path)
    if features is not None:
        prediction = model.predict(features)
        return EMOTION_LABELS[np.argmax(prediction)]
    return "Could not predict"
