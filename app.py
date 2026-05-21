import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DhwaniCore · Emotion AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Emotion metadata ───────────────────────────────────────────────────────────
EMOTIONS = {
    "calm":     {"emoji": "😌", "hex": "#34D399", "rgb": (0.204, 0.831, 0.600)},
    "happy":    {"emoji": "😊", "hex": "#FBBF24", "rgb": (0.984, 0.749, 0.141)},
    "fearful":  {"emoji": "😨", "hex": "#818CF8", "rgb": (0.506, 0.549, 0.973)},
    "disgust":  {"emoji": "🤢", "hex": "#F87171", "rgb": (0.973, 0.443, 0.443)},
    "angry":    {"emoji": "😠", "hex": "#EF4444", "rgb": (0.937, 0.267, 0.267)},
    "sad":      {"emoji": "😢", "hex": "#60A5FA", "rgb": (0.376, 0.647, 0.980)},
    "neutral":  {"emoji": "😐", "hex": "#94A3B8", "rgb": (0.580, 0.639, 0.722)},
    "surprised":{"emoji": "😲", "hex": "#F472B6", "rgb": (0.957, 0.447, 0.714)},
}
DEFAULT = {"emoji": "🎵", "hex": "#38BDF8", "rgb": (0.220, 0.741, 0.973)}

SAMPLES = {
    "😊  Happy":   "samples/happy.wav",
    "😌  Calm":    "samples/calm.wav",
    "😨  Fearful": "samples/fearful.wav",
    "🤢  Disgust": "samples/disgust.wav",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

*, *::before, *::after {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}
[data-testid="stAppViewContainer"] { background: #070b18 !important; }
[data-testid="stHeader"]           { background: transparent !important; }
section.main > div.block-container {
    padding-top: 1.5rem !important;
    max-width: 960px !important;
}
footer { visibility: hidden; }

/* ── Hero ── */
.dw-hero       { text-align: center; padding: 1.8rem 0 1rem; }
.dw-logo       {
    font-size: 3.4rem; font-weight: 900; letter-spacing: -2px; line-height: 1.05;
    background: linear-gradient(135deg, #38BDF8, #818CF8 55%, #C084FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.dw-sub        { font-size: .75rem; letter-spacing: 4px; text-transform: uppercase; color: #2d3f58; font-weight: 600; margin-top: .3rem; }
.dw-rule       { width: 48px; height: 2px; margin: 1rem auto 2rem; background: linear-gradient(90deg,#38BDF8,#818CF8); border: none; border-radius: 2px; }

/* ── Emotion result card ── */
.dw-result {
    border-radius: 20px;
    padding: 2.2rem 1.5rem 1.8rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,.07);
    background: linear-gradient(160deg, rgba(12,18,40,.98) 0%, rgba(18,28,58,.85) 100%);
    margin-bottom: 1.5rem;
}
.dw-emoji  { font-size: 3.8rem; line-height: 1; display: block; margin-bottom: .7rem; }
.dw-elabel { font-size: .65rem; letter-spacing: 4px; text-transform: uppercase; color: #334155; margin-bottom: .35rem; font-weight: 600; }
.dw-ename  { font-size: 2.5rem; font-weight: 900; letter-spacing: -1px; margin-bottom: .3rem; }
.dw-econf  { font-size: .85rem; color: #3d5068; font-weight: 500; }

/* ── Section labels ── */
.dw-section-label {
    font-size: .65rem; letter-spacing: 3px; text-transform: uppercase;
    color: #2d3f58; font-weight: 600; margin-bottom: .6rem;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(20,30,55,.7) !important;
    border: 1px solid rgba(100,120,200,.25) !important;
    color: #8899bb !important;
    border-radius: 10px !important;
    font-size: .84rem !important;
    font-weight: 500 !important;
    padding: .55rem 1rem !important;
    transition: all .18s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(100,120,200,.18) !important;
    border-color: rgba(129,140,248,.6) !important;
    color: #e2e8f0 !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(12,18,40,.7) !important;
    border: 1.5px dashed rgba(56,189,248,.2) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(56,189,248,.6) !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #8899bb !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(20,30,55,.7) !important;
    border: 1px solid rgba(100,120,200,.25) !important;
    color: #8899bb !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(100,120,200,.18) !important;
    border-color: rgba(129,140,248,.6) !important;
    color: #e2e8f0 !important;
}

/* ── Streamlit native pyplot container – kill white bg ── */
[data-testid="stImage"] img { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        m = joblib.load("Emotion_Model.pkl")
        s = joblib.load("DhwaniCore_Scaler.pkl")
        e = joblib.load("Label_Encoder.pkl")
        return m, s, e
    except Exception as err:
        st.error(f"Model load failed: {err}")
        return None, None, None

model, scaler, encoder = load_models()


# ── Feature extraction (librosa handles wav/mp3/flac/ogg/m4a/aac) ──────────────
def extract_feature(path):
    X, sr = librosa.load(path, sr=None, mono=True)
    return np.hstack([
        np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0),
        np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(X)), sr=sr).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0),
    ])


# ── Chart helpers ──────────────────────────────────────────────────────────────
def make_waveform(y, sr, rgb):
    r, g, b = rgb
    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor("#0a1020")
    ax.set_facecolor("#0a1020")
    t = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(t, y, alpha=0.30, color=(r, g, b))
    ax.plot(t, y, color=(r, g, b), linewidth=0.9, alpha=0.95)
    ax.axhline(0, color=(1, 1, 1, 0.05), linewidth=0.7)
    ax.set_xlabel("Time (s)", color="#2d3f58", fontsize=8, labelpad=4)
    ax.tick_params(colors="#2d3f58", labelsize=7, length=3)
    for sp in ax.spines.values():
        sp.set_color((1, 1, 1, 0.06))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    return fig


def make_spectrogram(path):
    y, sr = librosa.load(path, sr=None)
    S_db  = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max
    )
    fig, ax = plt.subplots(figsize=(9, 2.4))
    fig.patch.set_facecolor("#0a1020")
    ax.set_facecolor("#0a1020")
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             ax=ax, cmap="magma")
    ax.set_xlabel("Time (s)", color="#2d3f58", fontsize=8, labelpad=4)
    ax.set_ylabel("Hz",       color="#2d3f58", fontsize=8, labelpad=4)
    ax.tick_params(colors="#2d3f58", labelsize=7, length=3)
    for sp in ax.spines.values():
        sp.set_color((1, 1, 1, 0.06))
    fig.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════ LAYOUT ════════════════════════════════════════

# Hero
st.markdown("""
<div class="dw-hero">
  <div class="dw-logo">DhwaniCore</div>
  <div class="dw-sub">Speech Emotion Intelligence</div>
  <hr class="dw-rule">
</div>
""", unsafe_allow_html=True)

# ── Upload + samples ───────────────────────────────────────────────────────────
_, mid, _ = st.columns([1, 2.2, 1])
with mid:
    uploaded_file = st.file_uploader(
        "Drop an audio file here or click to browse",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"],
        label_visibility="collapsed"
    )
    st.markdown(
        "<p style='text-align:center;color:#1e2d42;font-size:.78rem;"
        "letter-spacing:2px;margin:.8rem 0 .5rem;'>— OR TRY A SAMPLE —</p>",
        unsafe_allow_html=True
    )
    samp_keys = list(SAMPLES.keys())
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    chosen_sample = None
    for col, label in zip([r1c1, r1c2, r2c1, r2c2], samp_keys):
        if col.button(label, key=f"s_{label}"):
            chosen_sample = label

# ── Resolve file ───────────────────────────────────────────────────────────────
file_path = None
audio_src  = None
is_tmp     = False

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[-1].lower() or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    file_path = tmp.name
    audio_src = uploaded_file
    is_tmp    = True
elif chosen_sample is not None:
    p = SAMPLES[chosen_sample]
    if os.path.exists(p):
        file_path = p
        audio_src = p
    else:
        st.error(f"Sample not found: {p}")

# ── Inference + display ────────────────────────────────────────────────────────
if file_path and model:
    try:
        with st.spinner("Analysing audio…"):
            y_wav, sr_wav = librosa.load(file_path, sr=None, duration=5)
            feats         = extract_feature(file_path).reshape(1, -1)
            feats_sc      = scaler.transform(feats)
            raw_pred      = model.predict(feats_sc)[0]
            probs         = model.predict_proba(feats_sc)[0]
            pred_name     = encoder.inverse_transform([raw_pred])[0]
            confidence    = probs.max() * 100

        cfg     = EMOTIONS.get(pred_name, DEFAULT)
        hex_col = cfg["hex"]
        rgb     = cfg["rgb"]
        emoji   = cfg["emoji"]
        r, g, b = rgb

        # ── Emotion hero card (pure HTML — no st widgets inside)
        _, hc, _ = st.columns([0.5, 2, 0.5])
        with hc:
            st.markdown(f"""
            <div class="dw-result"
                 style="box-shadow:0 0 70px rgba({int(r*255)},{int(g*255)},{int(b*255)},0.18);">
              <span class="dw-emoji">{emoji}</span>
              <div class="dw-elabel">Detected Emotion</div>
              <div class="dw-ename"
                   style="color:{hex_col};
                          text-shadow:0 0 28px rgba({int(r*255)},{int(g*255)},{int(b*255)},0.45);">
                {pred_name.upper()}
              </div>
              <div class="dw-econf">Confidence — {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Audio player (standalone — no wrapper div)
        st.markdown('<p class="dw-section-label">Audio Playback</p>',
                    unsafe_allow_html=True)
        st.audio(audio_src)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Waveform (standalone — no wrapper div)
        st.markdown('<p class="dw-section-label">Acoustic Waveform</p>',
                    unsafe_allow_html=True)
        fig_w = make_waveform(y_wav, sr_wav, rgb)
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Spectrogram (standalone — no wrapper div)
        st.markdown('<p class="dw-section-label">Mel Spectrogram</p>',
                    unsafe_allow_html=True)
        fig_s = make_spectrogram(file_path)
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

    except Exception as err:
        st.error(f"Processing error: {err}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        if is_tmp and os.path.exists(file_path):
            os.remove(file_path)
