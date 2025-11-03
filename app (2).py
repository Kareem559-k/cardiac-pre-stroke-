import streamlit as st
import numpy as np
import pandas as pd
import random, re, warnings
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="💙 Cardiac Multi-Disease Analyzer", page_icon="🫀", layout="centered")

# --------------------------------
# Custom Style (Dark Mode)
# --------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0a192f 0%, #112d4e 100%);
    color: #ffffff;
}
[data-testid="stSidebar"] { display: none; }
h1, h2, h3, h4, h5 { color: #ffffff; }
.stButton>button {
    background-color: #1b263b;
    color: #ffffff;
    border-radius: 10px;
    border: 1px solid #00b4d8;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00b4d8;
    color: #0a192f;
    border: 1px solid #0a192f;
}
div.stAlert {
    background-color: rgba(10, 25, 47, 0.3) !important;
    border-left: 4px solid #00b4d8 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Title
# --------------------------------
st.title("🩺 Cardiac Pre-Stroke & Multi-Disease Analyzer")
st.caption("AI-powered simulated ECG diagnostic system — for demonstration purposes only.")

# --------------------------------
# Upload ECG files
# --------------------------------
st.markdown("### 📤 Upload ECG Record (.hea + .dat)")
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# --------------------------------
# Helper functions
# --------------------------------
def extract_numeric_id(name):
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def auto_clean_ecg(signal):
    b, a = butter(3, 0.1)
    return filtfilt(b, a, signal)

def simulate_multi_disease(nid):
    diseases = {
        "Pre-Stroke": 0,
        "Arrhythmia": 0,
        "Myocardial Infarction": 0,
        "Atrial Fibrillation": 0
    }

    if nid is None:
        return diseases

    if nid % 2 == 1:
        diseases["Pre-Stroke"] = random.uniform(0.74, 0.9)
        diseases["Arrhythmia"] = random.uniform(0.65, 0.85)
        diseases["Myocardial Infarction"] = random.uniform(0.5, 0.7)
        diseases["Atrial Fibrillation"] = random.uniform(0.4, 0.6)
    else:
        diseases["Pre-Stroke"] = random.uniform(0.05, 0.15)
        diseases["Arrhythmia"] = random.uniform(0.05, 0.2)
        diseases["Myocardial Infarction"] = random.uniform(0.1, 0.25)
        diseases["Atrial Fibrillation"] = random.uniform(0.05, 0.15)

    return diseases

def make_probability_bar(prob, label):
    fig, ax = plt.subplots(figsize=(6,1.2))
    color = "#ff4d4d" if prob > 0.5 else "#4caf50"
    ax.barh(["Risk"], [prob], color=color, height=0.5)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(prob, 0, f"{prob*100:.1f}%", va='center', fontsize=11, fontweight='bold', color='white')
    buf = BytesIO()
    plt.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def make_donut_chart(data):
    labels = list(data.keys())
    values = [v*100 for v in data.values()]
    colors = ["#ff4d4d", "#ffd166", "#06d6a0", "#118ab2"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, textprops={"color":"white"}, wedgeprops={"width":0.4}
    )
    ax.set_title("🩸 Disease Risk Distribution", color="white", fontsize=13)
    fig.patch.set_alpha(0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --------------------------------
# Main
# --------------------------------
if hea_file and dat_file:
    record_name = hea_file.name.replace(".hea", "")
    with open(hea_file.name, "wb") as f: f.write(hea_file.read())
    with open(dat_file.name, "wb") as f: f.write(dat_file.read())

    st.markdown(f"**📁 Record:** `{record_name}`")

    try:
        rec = rdrecord(record_name)
        sig = rec.p_signal
        y = sig[:,0] if sig.ndim > 1 else sig
        y_clean = auto_clean_ecg(y)

        # ECG Signal
        st.markdown("#### 🩸 ECG Signal (first 2000 samples)")
        fig1, ax1 = plt.subplots(figsize=(8,2.2))
        ax1.plot(y_clean[:2000], color="#00b4d8", linewidth=0.9)
        ax1.set_ylabel("Amplitude", color="white")
        ax1.set_xlabel("Samples", color="white")
        ax1.grid(alpha=0.3)
        fig1.patch.set_alpha(0)
        st.pyplot(fig1)
        plt.close(fig1)

        # Amplitude Histogram
        st.markdown("#### 📊 Amplitude Distribution")
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.hist(y_clean, bins=60, color="#0077b6", alpha=0.9)
        ax2.set_xlabel("Amplitude", color="white")
        ax2.set_ylabel("Count", color="white")
        ax2.grid(alpha=0.3)
        fig2.patch.set_alpha(0)
        st.pyplot(fig2)
        plt.close(fig2)

        # RMS Trend
        st.markdown("#### ⚡ Signal RMS Trend")
        rms = np.sqrt(pd.Series(y_clean).rolling(window=80).mean().fillna(method='bfill').values)
        fig3, ax3 = plt.subplots(figsize=(6,1.2))
        ax3.plot(rms[-200:], color="#90e0ef", linewidth=1)
        ax3.set_yticks([]); ax3.set_xticks([])
        fig3.patch.set_alpha(0)
        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as e:
        st.warning(f"⚠️ Unable to read ECG: {e}")
        y_clean = None

    # Simulated multi-disease result
    nid = extract_numeric_id(record_name)
    results = simulate_multi_disease(nid)

    st.markdown("### 🧠 Diagnostic Results")
    for disease, prob in results.items():
        color = "#ff4d4d" if prob > 0.5 else "#4caf50"
        msg = "⚠️ Possible abnormality detected." if prob > 0.5 else "💚 No risk detected."
        st.markdown(f"""
        <div style='background:{color};padding:14px;border-radius:10px;text-align:center;font-size:17px;color:white'>
            <b>{disease}</b><br>{msg}<br><b>Risk:</b> {prob*100:.1f}%
        </div>
        """, unsafe_allow_html=True)
        img_bytes = make_probability_bar(prob, disease)
        st.image(img_bytes, use_container_width=True)

    # Donut Chart (after results)
    st.markdown("### 📈 Overall Disease Risk Summary")
    donut_img = make_donut_chart(results)
    st.image(donut_img, use_container_width=False)

else:
    st.info("Please upload both `.hea` and `.dat` files to start analysis.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("💙 Cardiac Multi-Disease Analyzer © 2025 — Smart AI Mode — Simulated ML System.")
