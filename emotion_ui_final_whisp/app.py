import io
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import soundfile as sf
import torch
import torch.nn as nn

# Optional live recording
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_RECORDER = True
except:
    HAS_RECORDER = False

# ========= BASIC CONFIG ========= #
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
}
SR = 16000

MODEL_PATH = "models/ser_finetuned.pt"

# ======= LOAD CSS ======= #
with open("styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ======= DECORATIONS ======= #
decor_html = """
<div class="page-decor"></div>
<div class="floating-emoji" style="top: 8%; left: 4%;">🎤</div>
<div class="floating-emoji" style="top: 70%; left: 3%;">🎧</div>
<div class="floating-emoji" style="top: 20%; right: 5%;">🔊</div>
<div class="floating-emoji" style="bottom: 10%; right: 4%;">🎵</div>
<div class="side-lines left"></div>
<div class="side-lines right"></div>
"""
st.markdown(decor_html, unsafe_allow_html=True)

# ============================================
#       LOAD WHISPER PROCESSOR + MODEL
# ============================================
from transformers import WhisperProcessor, WhisperModel

@st.cache_resource
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper = WhisperModel.from_pretrained("openai/whisper-small")
    whisper.eval()
    return processor, whisper

processor, whisper = load_whisper()

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper = whisper.to(device)

# ============================================
#             MIXER-SER MODEL
# ============================================
class MixerSER(nn.Module):
    def __init__(self, dim, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, dim),
        )
        self.final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.final(x)

# ============================================
#          LOAD FINETUNED MODEL (.pt)
# ============================================
@st.cache_resource
def load_classifier():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False   # allow pickle loading
    )

    classes = checkpoint["classes"]

    dim = checkpoint["model"]["block1.1.weight"].shape[1]
    model = MixerSER(dim=dim, num_classes=len(classes))
    model.load_state_dict(checkpoint["model"])   # FIXED HERE
    model.to(device)
    model.eval()

    return model, classes


classifier, classes = load_classifier()

# ============================================
#      AUDIO → WHISPER EMBEDDING EXTRACTOR
# ============================================
def extract_embedding_from_bytes(audio_bytes):
    """Return Whisper embedding + waveform + mel spectrogram."""
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_buffer, dtype="float32")

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Resample
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Loudness normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Remove silence
    y, _ = librosa.effects.trim(y)

    # Whisper inputs
    inputs = processor(y, sampling_rate=SR, return_tensors="pt")
    feats = inputs.input_features.to(device)

    with torch.no_grad():
        out = whisper.encoder(feats)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

    # Mel spectrogram for UI
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return emb.astype(np.float32), y, mel_db

# ============================================
#              PREDICT EMOTION
# ============================================
def predict_emotion(embedding):
    x = torch.tensor(embedding).unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return classes[idx], probs

# ============================================
#                HEADER
# ============================================
st.markdown(
    """
    <div class="app-title">🎤 Speech Emotion Recognition</div>
    <div class="app-subtitle">
        Upload or record speech and let the model estimate the underlying emotion.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="center">
        <div class="team-card">
            <b>Team 20 – Speech Emotion Recognition</b><br/>
            CREMA-D Dataset · Himanshi Aggarwal · Noel Tom · Jnanasagara Srinivasa
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================
#                INPUT CARD
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">1️⃣ Input Audio</div>', unsafe_allow_html=True)
st.markdown('<p class="muted">Upload or record your audio.</p>', unsafe_allow_html=True)

tabs = st.tabs(["📁 Upload Audio", "🎙️ Record Live"])
audio_bytes = None

with tabs[0]:
    uploaded = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format="audio/wav")

with tabs[1]:
    if HAS_RECORDER:
        recorded_audio = audio_recorder(
            text="Tap to record / stop", 
            recording_color="#22c55e",
            neutral_color="#64748b",
            icon_name="microphone",
            icon_size="3x",
        )
        if recorded_audio:
            audio_bytes = recorded_audio
            st.audio(audio_bytes, format="audio/wav")
    else:
        st.info("Install: `pip install audio-recorder-streamlit`")

st.markdown('<div class="analyse-wrapper center">', unsafe_allow_html=True)
analyse_clicked = st.button("🔮 Analyse Emotion", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

prediction_placeholder = st.empty()
chart_placeholder = st.empty()
waveform_placeholder = st.empty()
spec_placeholder = st.empty()

# ============================================
#        PROCESS + PREDICT + VISUALIZE
# ============================================
if analyse_clicked:
    if audio_bytes is None:
        st.warning("Please upload or record an audio clip.")
    else:
        with st.spinner("Extracting Whisper embeddings..."):
            emb, y, mel_db = extract_embedding_from_bytes(audio_bytes)
            pred_label, probs = predict_emotion(emb)

        emoji = EMOJI_MAP[pred_label]
        confidence = float(np.max(probs))

        # Prediction Card
        with prediction_placeholder.container():
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="display:flex; justify-content:space-between;">
                        <div>
                            <span class="tag-pill">PREDICTED EMOTION</span>
                            <div class="emotion-label">{emoji} {pred_label.upper()}</div>
                            <p class="muted">Top confidence: <b>{confidence*100:.1f}%</b></p>
                        </div>
                        <div class="emoji-bubble">{emoji}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Confidence chart
        with chart_placeholder.container():
            st.markdown("### Class-wise confidence")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=classes, y=[float(p) for p in probs]))
            fig.update_layout(
                xaxis_title="Emotion",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1.0]),
                template="plotly_dark",
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)

        # Waveform
        with waveform_placeholder.container():
            st.markdown("### Waveform")
            fig_w, ax_w = plt.subplots(figsize=(6, 2))
            librosa.display.waveshow(y, sr=SR, ax=ax_w)
            st.pyplot(fig_w)

        # Spectrogram
        with spec_placeholder.container():
            st.markdown("### Spectrogram")
            fig_s, ax_s = plt.subplots(figsize=(6, 2.5))
            img = librosa.display.specshow(
                mel_db, x_axis="time", y_axis="mel", sr=SR, fmax=8000, ax=ax_s
            )
            fig_s.colorbar(img, ax=ax_s)
            st.pyplot(fig_s)

# Footer
st.markdown(
    "<div class='footer'>Team 20 · Speech Emotion Recognition · Amrita School of Computing, Bengaluru</div>",
    unsafe_allow_html=True,
)
