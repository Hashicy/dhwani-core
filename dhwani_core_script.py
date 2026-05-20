import librosa
import soundfile as sf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
!pip install gdown
import gdown

# Download sample RAVDESS data
!wget -q --show-progress \
  "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
!unzip -q Audio_Speech_Actors_01-24.zip -d ravdess_data/
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

print("🎭 Emotion Categories:")
for code, emotion in emotions.items():
    marker = "✅" if emotion in observed_emotions else "⬜"
    print(f"  {marker} Code {code}: {emotion}")

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):

    # Open and read the sound file
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        # Initialize result array
        result = np.array([])

        # 1. MFCC - Mel Frequency Cepstral Coefficients
        # Captures timbral texture of sound
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(
                    y=X,
                    sr=sample_rate,
                    n_mfcc=40
                ).T,
                axis=0
            )
            result = np.hstack((result, mfccs))

        # 2. Chroma - Represents 12 different pitch classes
        # Captures harmonic and melodic characteristics
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_feat = np.mean(
                librosa.feature.chroma_stft(
                    S=stft,
                    sr=sample_rate
                ).T,
                axis=0
            )
            result = np.hstack((result, chroma_feat))

        # 3. Mel Spectrogram - Frequency spectrum of sound
        # Maps power of signal in frequency domain
        if mel:
            mel_feat = np.mean(
                librosa.feature.melspectrogram(
                    y=X,
                    sr=sample_rate
                ).T,
                axis=0
            )
            result = np.hstack((result, mel_feat))

    return result
# Test the feature extraction
print("🧪 Testing Feature Extraction...")

# Get first audio file for testing
test_files = glob.glob("ravdess_data/**/*.wav", recursive=True)

if test_files:
    test_file = test_files[0]
    features = extract_feature(test_file)
    print(f"✅ Feature extraction works!")
    print(f"📊 Feature vector shape: {features.shape}")
    print(f"📁 Test file: {os.path.basename(test_file)}")
else:
    print("⚠️ No audio files found. Check dataset path!")
def load_data(test_size=0.2):
    x, y = [], []
    files_processed = 0
    files_skipped = 0

    audio_files = glob.glob("ravdess_data/**/*.wav", recursive=True)
    total_files = len(audio_files)

    for idx, file in enumerate(audio_files):

        # Get filename
        file_name = os.path.basename(file)

        # Extract emotion from filename (3rd identifier)
        # Format: 03-01-06-01-02-01-12.wav
        try:
            emotion_code = file_name.split("-")[2]
            emotion = emotions.get(emotion_code, None)
        except IndexError:
            files_skipped += 1
            continue

        # Skip if emotion not in our observed list
        if emotion not in observed_emotions:
            continue

        # Extract features
        try:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            files_processed += 1

            # Progress update every 50 files
            if files_processed % 50 == 0:
                print(f"  ✅ Processed {files_processed} files...")

        except Exception as e:
            files_skipped += 1
            continue

    print(f"\n📊 Summary:")
    print(f"  ✅ Files processed: {files_processed}")
    print(f"  ⏭️  Files skipped: {files_skipped}")

    # Split into train and test sets
    return train_test_split(
        np.array(x),
        y,
        test_size=test_size,
        random_state=9
    )


# Load the data
print("🚀 Loading data...")
x_train, x_test, y_train_str, y_test_str = load_data(test_size=0.25)

# Encode string labels to numerical labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_str)
y_test = label_encoder.transform(y_test_str)

print(f"\n📐 Dataset Shapes:")
print(f"  Training set:   {x_train.shape}")
print(f"  Testing set:    {x_test.shape}")
print(f"  Features count: {x_train.shape[1]}")
print(f"  Encoded Labels: {label_encoder.classes_}")
# Cell 6: Explore and Visualize Data

# --- Plot 1: Emotion Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training set distribution
from collections import Counter
train_counts = Counter(y_train)
test_counts = Counter(y_test)

axes[0].bar(
    train_counts.keys(),
    train_counts.values(),
    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
    edgecolor='black'
)
axes[0].set_title('Training Set - Emotion Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Emotion')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Testing set distribution
axes[1].bar(
    test_counts.keys(),
    test_counts.values(),
    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
    edgecolor='black'
)
axes[1].set_title('Testing Set - Emotion Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Emotion')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('emotion_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Emotion distribution plotted!")
# Cell 7: Visualize Audio Features

# Load a sample file for visualization
sample_files = {}
for file in glob.glob("ravdess_data/**/*.wav", recursive=True)[:50]:
    emotion_code = os.path.basename(file).split("-")[2]
    emotion = emotions.get(emotion_code)
    if emotion in observed_emotions and emotion not in sample_files:
        sample_files[emotion] = file

print(f"Sample files for visualization: {list(sample_files.keys())}")

fig, axes = plt.subplots(len(sample_files), 3, figsize=(18, 4*len(sample_files)))

for idx, (emotion, file_path) in enumerate(sample_files.items()):

    # Load audio
    y_audio, sr = librosa.load(file_path, duration=3)

    # Plot 1: Waveform
    axes[idx, 0].set_title(f'{emotion.upper()} - Waveform', fontweight='bold')
    librosa.display.waveshow(y_audio, sr=sr, ax=axes[idx, 0], color='#2196F3')
    axes[idx, 0].set_xlabel('Time (s)')
    axes[idx, 0].set_ylabel('Amplitude')

    # Plot 2: MFCC
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    img2 = librosa.display.specshow(
        mfccs, x_axis='time', ax=axes[idx, 1], cmap='viridis'
    )
    axes[idx, 1].set_title(f'{emotion.upper()} - MFCC', fontweight='bold')
    fig.colorbar(img2, ax=axes[idx, 1])

    # Plot 3: Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img3 = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel',
        ax=axes[idx, 2], cmap='magma'
    )
    axes[idx, 2].set_title(f'{emotion.upper()} - Mel Spectrogram', fontweight='bold')
    fig.colorbar(img3, ax=axes[idx, 2], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('audio_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Audio features visualized!")
# Cell 8: Scale Features

# Normalize features for better MLP performance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("✅ Features scaled successfully!")
print(f"Training features - Mean: {x_train_scaled.mean():.4f}, Std: {x_train_scaled.std():.4f}")
print(f"Testing features  - Mean: {x_test_scaled.mean():.4f}, Std: {x_test_scaled.std():.4f}")

# Cell 9: Build MLP Classifier

# Initialize Multi-Layer Perceptron Classifier
model = MLPClassifier(
    hidden_layer_sizes=(300, 200, 100),  # 3 hidden layers
    activation='relu',                    # ReLU activation function
    solver='adam',                        # Adam optimizer
    alpha=0.0001,                         # L2 regularization
    batch_size='auto',                    # Auto batch size
    learning_rate='adaptive',            # Adaptive learning rate
    max_iter=500,                         # Max iterations
    random_state=42,                      # Reproducibility
    early_stopping=True,                  # Stop when validation improves
    validation_fraction=0.1,             # 10% for validation
    n_iter_no_change=20,                  # Patience for early stopping
    verbose=True                          # Show training progress
)

print("🏗️ MLP Architecture:")
print(f"  Input Layer:    {x_train.shape[1]} features")
print(f"  Hidden Layer 1: 300 neurons (ReLU)")
print(f"  Hidden Layer 2: 200 neurons (ReLU)")
print(f"  Hidden Layer 3: 100 neurons (ReLU)")
print(f"  Output Layer:   {len(label_encoder.classes_)} classes") # Use the length of encoded classes
print(f"\n⚙️  Configuration:")
print(f"  Optimizer:      Adam")
print(f"  Max Iterations: 500")
print(f"  Early Stopping: Yes (patience=20)")
from sklearn.model_selection import GridSearchCV

# Cell 9.1: Hyperparameter Tuning with GridSearchCV

print("🧪 Starting GridSearchCV for MLPClassifier...")

# Define the parameter grid to search
param_grid = {
    'hidden_layer_sizes': [(200, 100), (300, 200, 100)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [300, 500],
    'early_stopping': [True],
    'validation_fraction': [0.1],
    'n_iter_no_change': [20],
    'random_state': [42]
}

# Initialize GridSearchCV
# verbose=3 provides detailed output during fitting
# n_jobs=-1 uses all available CPU cores
grid_search = GridSearchCV(estimator=MLPClassifier(),
                         param_grid=param_grid,
                         cv=5, # 5-fold cross-validation
                         scoring='accuracy',
                         verbose=3,
                         n_jobs=-1)

# Fit GridSearchCV to the scaled training data
grid_search.fit(x_train_scaled, y_train)

print("\n✅ GridSearchCV complete!")

# Get the best parameters and best model
best_params = grid_search.best_params_
best_mlp = grid_search.best_estimator_

print("\n✨ Best Hyperparameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n🏆 Best Cross-validation Accuracy: {grid_search.best_score_:.4f}")
# Cell 10: Train the Model

# Train the model
best_mlp.fit(x_train_scaled, y_train)

print("\n✅ Training Complete!")
print(f"  Iterations completed: {best_mlp.n_iter_}")
print(f"  Best validation score: {best_mlp.best_validation_score_:.4f}")
# Cell 11: Make Predictions and Evaluate

# Predict on test set
y_pred = best_mlp.predict(x_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("="*50)
print("📊 MODEL EVALUATION RESULTS")
print("="*50)
print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%")
print(f"\n📋 Detailed Classification Report:")
print("-"*50)
# Use label_encoder.classes_ for target_names to get readable emotion names
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Confusion Matrix ---
# Use label_encoder.classes_ for labels, xticklabels, and yticklabels
# Fixed: Removed the 'labels' argument from confusion_matrix as y_test and y_pred are numerical.
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ax=axes[0],
    linewidths=0.5
)
axes[0].set_title('Confusion Matrix\n(Absolute Values)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Emotion', fontsize=12)
axes[0].set_ylabel('True Emotion', fontsize=12)

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Greens',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ax=axes[1],
    linewidths=0.5
)
axes[1].set_title('Confusion Matrix\n(Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Emotion', fontsize=12)
axes[1].set_ylabel('True Emotion', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Confusion matrix saved!")
# Cell 13: Training Loss Curve

plt.figure(figsize=(10, 5))
plt.plot(best_mlp.loss_curve_, color='#2196F3', linewidth=2, label='Training Loss')
if hasattr(best_mlp, 'validation_scores_'):
    plt.plot(
        [1-s for s in best_mlp.validation_scores_],
        color='#FF5722',
        linewidth=2,
        label='Validation Loss',
        linestyle='--'
    )
plt.title('MLP Training Loss Curve', fontsize=14, fontweight='bold')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training loss curve plotted!")
# Cell 14: Predict on New Audio File

def predict_emotion(audio_file_path, model, scaler, label_encoder):
    """
    Predict emotion from a new audio file

    Parameters:
    -----------
    audio_file_path : str - path to audio file
    model           : trained MLPClassifier
    scaler          : fitted StandardScaler
    label_encoder   : fitted LabelEncoder

    Returns:
    --------
    predicted emotion and probabilities
    """

    print(f"🎵 Analyzing: {os.path.basename(audio_file_path)}")

    # Extract features
    features = extract_feature(
        audio_file_path,
        mfcc=True,
        chroma=True,
        mel=True
    )

    # Reshape and scale
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict numerical label
    numerical_prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # Inverse transform to get string emotion
    prediction = label_encoder.inverse_transform([numerical_prediction])[0]

    # Display results
    print("\n" + "="*40)
    print(f"🎭 Predicted Emotion: {prediction.upper()}")
    print("="*40)
    print("\n📊 Confidence Scores:")

    # Map numerical class probabilities to string emotion names
    emotion_probs = dict(zip(label_encoder.classes_, probabilities))
    for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 20)
        print(f"  {emotion:<12}: {bar:<20} {prob*100:.1f}%")

    return prediction, emotion_probs


# Test with a sample file
if test_files:
    sample_file = test_files[0]
    # Pass label_encoder to the prediction function
    emotion_pred, probs = predict_emotion(sample_file, best_mlp, scaler, label_encoder)

    # Visualize the audio
    y_audio, sr = librosa.load(sample_file, duration=3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Waveform
    librosa.display.waveshow(y_audio, sr=sr, ax=axes[0], color='#2196F3')
    axes[0].set_title(f'Audio Waveform\nPredicted: {emotion_pred.upper()}', fontweight='bold')

    # Probability bar chart
    emotions_list = list(probs.keys())
    probs_list = list(probs.values())
    colors = ['#4CAF50' if e == emotion_pred else '#90CAF9' for e in emotions_list]

    axes[1].barh(emotions_list, probs_list, color=colors, edgecolor='black')
    axes[1].set_title('Emotion Probability Distribution', fontweight='bold')
    axes[1].set_xlabel('Probability')
    axes[1].set_xlim(0, 1)

    for i, (prob, emotion) in enumerate(zip(probs_list, emotions_list)):
        axes[1].text(prob + 0.01, i, f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
# Cell 15: Save the Model

import pickle
import joblib

# Save model, scaler, and label_encoder
joblib.dump(best_mlp, 'emotion_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl') # Save the label encoder

print("✅ Model saved successfully!")
print("  📁 emotion_model.pkl - Trained MLP model")
print("  📁 scaler.pkl - Feature scaler")
print("  📁 label_encoder.pkl - Label encoder") # Print for label encoder

# Download the model (for Google Colab)
from google.colab import files
files.download('emotion_model.pkl')
files.download('scaler.pkl')
files.download('label_encoder.pkl') # Download label encoder
