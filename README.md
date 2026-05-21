<div align="center">
  <h1>DhwaniCore</h1>
  <p><b>Speech Emotion Intelligence</b></p>
</div>

<br/>

**DhwaniCore** is an end-to-end Machine Learning web application designed to detect and classify human emotions from speech audio. Built with Streamlit and deployed on the cloud, it provides real-time acoustic feature extraction, waveform generation, and AI-driven emotion prediction through an elegant, glassmorphism-styled UI.

## Features
- **Real-Time Emotion Detection:** Classifies audio into 8 core emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised).
- **Acoustic Feature Extraction:** Leverages `librosa` to compute Mel-Frequency Cepstral Coefficients (MFCCs), Chroma Short-Time Fourier Transforms (STFT), and Mel Spectrograms.
- **Dynamic Visualizations:** Automatically generates and displays time-domain waveforms and Mel Spectrograms for uploaded audio.
- **Dark Mode UI:** A fully custom, sleek dark-themed interface built using Streamlit's native theming engine.
- **Broad Format Support:** Accepts `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, and `.aac` via `ffmpeg` and `libsndfile`.

## Tech Stack
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Audio Processing:** [Librosa](https://librosa.org/), Soundfile
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Model & Scaler pipelines), Joblib
- **Data Visualization:** Matplotlib, NumPy

## How It Works
1. **Input:** The user uploads an audio clip or selects one of the pre-loaded sample files.
2. **Signal Processing:** The audio is read as a mono time-series array. 
3. **Feature Extraction:** 
   - 40-band **MFCC** (captures vocal tract shape)
   - **Chroma STFT** (captures pitch classes)
   - **Mel Spectrogram** (captures frequencies converted to the Mel scale)
4. **Prediction:** Extracted features are flattened, scaled using the pre-fitted `StandardScaler` (`DhwaniCore_Scaler.pkl`), and fed into the trained Scikit-Learn classifier (`Emotion_Model.pkl`).
5. **Output:** The app returns the detected emotion, confidence percentage, and visual acoustic plots.

## Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hashicy/dhwani-core.git
   cd dhwani-core
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may need system-level audio dependencies like `ffmpeg` installed on your machine to process formats like `.mp3` or `.m4a`)*

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Deployment (Streamlit Cloud)
This project is configured for seamless deployment on **Streamlit Cloud**. 
- Python dependencies are managed via `requirements.txt`.
- OS-level dependencies (like `libsndfile1` and `ffmpeg`) required by Librosa are automatically installed by Streamlit Cloud via the included `packages.txt` file.
- UI theming is enforced via `.streamlit/config.toml` rather than brittle CSS hacks.

## Project Structure
```text
dhwani-core/
├── .streamlit/
│   └── config.toml          # Native dark theme configuration
├── samples/                 # Sample audio files (Happy, Calm, etc.)
├── app.py                   # Main Streamlit application
├── Emotion_Model.pkl        # Pre-trained Scikit-Learn classifier
├── DhwaniCore_Scaler.pkl    # Pre-fitted feature scaler
├── Label_Encoder.pkl        # Target label encoder mapping
├── requirements.txt         # Python dependencies
├── packages.txt             # Apt-get system dependencies for cloud
└── README.md
```
