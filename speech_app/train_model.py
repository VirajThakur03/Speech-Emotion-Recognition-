import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Bidirectional, Dense, Dropout,
                                     Attention, Conv1D, MaxPooling1D, BatchNormalization,
                                     GlobalAveragePooling1D, Add)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Emotion mapping
emotions = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
    'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
}

# Dataset paths
dataset_paths = {
    'ravdess': r"C:\Users\ROHAN\Desktop\ravdess",
    'crema-d': r"C:\Users\ROHAN\Desktop\crema-d"
}

# Feature extractor
def extract_features(file_path):
    try:
        X, sample_rate = librosa.load(file_path, sr=22050)
        stft = np.abs(librosa.stft(X))

        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        combined_mfcc = np.vstack((mfcc, delta_mfcc, delta2_mfcc))

        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)

        min_time = min(combined_mfcc.shape[1], chroma.shape[1], mel.shape[1])
        combined = np.vstack((
            combined_mfcc[:, :min_time],
            chroma[:, :min_time],
            mel[:, :min_time]
        ))
        return combined.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Dataset loader functions
def load_ravdess():
    X, y = [], []
    for file in glob.glob(os.path.join(dataset_paths['ravdess'], "Actor_*", "*.wav")):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        emotion = emotion_map.get(emotion_code)
        if emotion:
            features = extract_features(file)
            if features is not None:
                X.append(features)
                y.append(emotions[emotion])
    return X, y

def load_crema_d():
    X, y = [], []
    for file in glob.glob(os.path.join(dataset_paths['crema-d'], "*.wav")):
        file_name = os.path.basename(file)
        parts = file_name.split('_')
        emotion_code = parts[2]
        crema_map = {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        }
        emotion = crema_map.get(emotion_code)
        if emotion:
            features = extract_features(file)
            if features is not None:
                X.append(features)
                y.append(emotions[emotion])
    return X, y

# Load and combine datasets
X_ravdess, y_ravdess = load_ravdess()
X_crema, y_crema = load_crema_d()

X = X_ravdess + X_crema
y = y_ravdess + y_crema

# Pad sequences
max_len = max(f.shape[0] for f in X)
feature_dim = X[0].shape[1]
X = [np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='constant') for f in X]
X = np.array(X)
y = to_categorical(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = Conv1D(128, kernel_size=5, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

shortcut = x  # Save for residual

x = Bidirectional(LSTM(64, return_sequences=True))(x)
shortcut = Dense(128)(shortcut)  # Project shortcut to match shape (128)
x = Add()([x, shortcut])  # Residual connection
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
attention = Attention()([x, x])
x = GlobalAveragePooling1D()(attention)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint("speech_app/best_model_ravdess_cremad.h5", monitor='val_accuracy', save_best_only=True)
]

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=callbacks)

# Evaluate
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_labels, y_pred_labels)
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
cm = confusion_matrix(y_true_labels, y_pred_labels)

print("\nEvaluation Metrics:")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", cm)

# Save model
os.makedirs("speech_app", exist_ok=True)
model.save("speech_app/bilstm_attention_model_ravdess_cremad.h5")
print("✅ Combined model (RAVDESS + CREMA-D) saved to speech_app/bilstm_attention_model_ravdess_cremad.h5")
