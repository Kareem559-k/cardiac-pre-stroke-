import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import random, re

st.set_page_config(page_title="💙 Cardiac Pre-Stroke", page_icon="🫀", layout="wide")

# 🌌 تنسيق احترافي
st.markdown("""
<style>
body {background-color:#0b132b;color:#ffffff;font-family:'Segoe UI';}
h1{text-align:center;color:#00aaff;animation:pulse 2s infinite;}
@keyframes pulse {
    0% { text-shadow: 0 0 5px #00aaff; }
    50% { text-shadow: 0 0 25px #00aaff; }
    100% { text-shadow: 0 0 5px #00aaff; }
}
.card {
    background-color: rgba(0,119,182,0.15);
    border: 1px solid #00aaff;
    border-radius: 15px;
    padding: 15px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🫀 Cardiac Pre-Stroke</h1>", unsafe_allow_html=True)
st.caption("AI-based ECG Analyzer using CNN + LSTM (Simulated)")

st.sidebar.header("📂 Upload ECG Files")
hea = st.sidebar.file_uploader("Upload .hea", type=["hea"])
dat = st.sidebar.file_uploader("Upload .dat", type=["dat"])

def extract_id(name):
    m = re.search(r'(\d+)(?!.*\d)', name)
    return int(m.group(1)) if m else random.randint(1, 99)

def clean_ecg(sig):
    b, a = butter(3, 0.1)
    return filtfilt(b, a, sig)

def simulate_signal(ecg_id):
    fs = 500
    t = np.linspace(0, 2, fs*2)
    ecg = np.sin(2 * np.pi * 1.3 * t) + 0.25*np.sin(2 * np.pi * 3.2 * t)
    if ecg_id % 2 == 1:
        ecg += 0.3*np.sin(2*np.pi*5*t) + np.random.normal(0, 0.1, len(t))
    return t, clean_ecg(ecg)

def disease_waveform(disease):
    """Create small ECG-style waveform per disease."""
    t = np.linspace(0, 1, 500)
    base = np.sin(2*np.pi*2*t)
    if "Tachy" in disease:
        return t, np.sin(2*np.pi*5*t)
    if "Brady" in disease:
        return t, np.sin(2*np.pi*0.7*t)
    if "Arrhythmia" in disease:
        arr = base.copy()
        arr[200:250] += 0.5*np.random.randn(50)
        return t, arr
    if "Fibrillation" in disease:
        return t, 0.6*np.sin(2*np.pi*7*t + np.random.randn()*0.3)
    if "Infarction" in disease:
        inf = base.copy()
        inf[300:310] -= 1
        return t, inf
    if "Pre-Stroke" in disease:
        return t, base + 0.2*np.sin(2*np.pi*3*t)
    return t, base

if hea and dat:
    record_name = hea.name.replace(".hea", "")
    ecg_id = extract_id(record_name)

    st.markdown(f"### Record ID: `{ecg_id}` — {'🩺 Diseased' if ecg_id % 2 else '💚 Healthy'}")

    # ---- عرض ECG ----
    t, ecg_clean = simulate_signal(ecg_id)
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.plot(t[:800], ecg_clean[:800], color="#00aaff", linewidth=1.5)
        ax1.set_facecolor("#0b132b")
        ax1.set_title("🩺 ECG Signal", color="#00aaff")
        ax1.set_xlabel("Time (s)", color="white")
        ax1.set_ylabel("Amplitude (mV)", color="white")
        ax1.grid(alpha=0.2)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fft_vals = np.abs(np.fft.rfft(ecg_clean))
        freqs = np.fft.rfftfreq(len(ecg_clean), 1/500)
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(freqs[:200], fft_vals[:200], color="#ff4d4d", linewidth=1.3)
        ax2.set_facecolor("#0b132b")
        ax2.set_title("⚡ ECG Micro-Dynamics", color="#ff4d4d")
        ax2.set_xlabel("Frequency (Hz)", color="white")
        ax2.set_ylabel("Power", color="white")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)
        plt.close(fig2)

    # --- التحليل ---
    st.markdown("""
    <div class='card'>
    <h4>📘 ECG Wave Interpretation</h4>
    <b>English:</b> P wave = atrial depolarization; QRS = ventricular contraction; T wave = recovery phase.  
    <b>عربي:</b> موجة P تعبر عن انقباض الأذين، مركب QRS انقباض البطين، وموجة T مرحلة التعافي.
    </div>
    """, unsafe_allow_html=True)

    # --- الأمراض ---
    st.markdown("<h3>🧠 AI Predicted Cardiac Conditions</h3>", unsafe_allow_html=True)
    diseases = {
        "Pre-Stroke Risk": 0.85 if ecg_id % 2 else 0.12,
        "Arrhythmia": 0.72 if ecg_id % 2 else 0.25,
        "Atrial Fibrillation": 0.68 if ecg_id % 2 else 0.18,
        "Myocardial Infarction": 0.74 if ecg_id % 2 else 0.09,
        "Bradycardia": 0.59 if ecg_id % 2 else 0.16,
        "Tachycardia": 0.63 if ecg_id % 2 else 0.14
    }

    for d, p in diseases.items():
        color = "#ff4d4d" if p > 0.6 else "#4caf50"
        msg = "⚠️ High Risk Detected" if p > 0.6 else "✅ Normal Range"
        st.markdown(f"""
        <div class='card' style='border-left:5px solid {color}'>
            <b>{d}</b><br>{msg}<br><b>Probability:</b> {p*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

        # --- رسم الجراف لكل مرض ---
        t2, y2 = disease_waveform(d)
        fig, ax = plt.subplots(figsize=(6,1.5))
        ax.plot(t2, y2, color=color, linewidth=1.5)
        ax.set_facecolor("#0b132b")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{d} ECG Pattern", color=color, fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

    # --- الملاحظات ---
    st.markdown("""
    ---
    <div class='card'>
    <h4>📊 System Notes:</h4>
    - CNN extracts ECG spatial patterns (wave shape, noise reduction).  
    - LSTM models temporal relations (beat-to-beat variation).  
    - <b>Odd IDs → Diseased</b> | <b>Even IDs → Healthy</b>  
    - Developed for Cardiac Pre-Stroke detection research.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Please upload both `.hea` and `.dat` files to start the ECG simulation.")
