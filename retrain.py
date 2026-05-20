import librosa
import soundfile as sf
import numpy as np
import os
import glob
import warnings
import urllib.request
import zipfile
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

warnings.filterwarnings('ignore')

print("Downloading dataset...")
if not os.path.exists("Audio_Speech_Actors_01-24.zip"):
    subprocess.run(["wget", "-q", "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"])

print("Extracting dataset...")
if not os.path.exists("ravdess_data"):
    os.makedirs("ravdess_data")
    subprocess.run(["unzip", "-q", "Audio_Speech_Actors_01-24.zip", "-d", "ravdess_data/"])

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))
    return result

def load_data(test_size=0.2):
    x, y = [], []
    audio_files = glob.glob("ravdess_data/**/*.wav", recursive=True)
    for file in audio_files:
        file_name = os.path.basename(file)
        try:
            emotion_code = file_name.split("-")[2]
            emotion = emotions.get(emotion_code, None)
        except IndexError:
            continue
        if emotion not in observed_emotions:
            continue
        try:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
        except Exception as e:
            continue
    return np.array(x), y

print("Extracting features from audio files...")
x, y_str = load_data()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print("Training model...")
# Keep it fast and exactly what app.py expects
model = MLPClassifier(
    hidden_layer_sizes=(300, 200, 100),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

model.fit(x_scaled, y)

print("Saving models...")
joblib.dump(model, 'Emotion_Model.pkl')
joblib.dump(scaler, 'DhwaniCore_Scaler.pkl')
joblib.dump(label_encoder, 'Label_Encoder.pkl')

print("Done! Retrained models are saved.")
