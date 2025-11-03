import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from io import BytesIO
import random, re, warnings

warnings.filterwarnings("ignore")

# --------------------------------
# إعداد الصفحة
# --------------------------------
st.set_page_config(page_title="🫀 Cardiac Pre-Stroke Analyzer", page_icon="💙", layout="centered")

# --------------------------------
# تنسيق داكن جميل
# --------------------------------
st.markdown("""
<style>
body {
    background-color: #0b132b;
    color: #ffffff;
    font-family: 'Segoe UI';
}
h1, h2, h3, h4 {
    color: #00aaff;
}
.stButton>button {
    background-color: #0077b6;
    color: white;
    border-radius: 10px;
    border: 1px solid #00aaff;
}
.stButton>button:hover {
    background-color: #00aaff;
    color: #0b132b;
}
div.stAlert {
    background-color: rgba(0, 119, 182, 0.2);
    border-left: 4px solid #00aaff;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# العنوان
# --------------------------------
st.title("💙 Cardiac Pre-Stroke & Arrhythmia Analyzer")
st.caption("Powered by simulated CNN + LSTM model — Demonstration prototype")

# --------------------------------
# رفع ملفات ECG
# --------------------------------
st.subheader("📤 Upload ECG Record (.hea + .dat)")
hea = st.file_uploader("Upload .hea file", type=["hea"])
dat = st.file_uploader("Upload .dat file", type=["dat"])

# --------------------------------
# دوال المساعدة
# --------------------------------
def extract_id(name):
    m = re.search(r'(\d+)(?!.*\d)', name)
    return int(m.group(1)) if m else random.randint(1, 99)

def clean_ecg(sig):
    b, a = butter(3, 0.1)
    return filtfilt(b, a, sig)

def fake_cnn_lstm_prediction(ecg_id):
    np.random.seed(ecg_id)
    preds = {
        "Pre-Stroke Risk": round(random.uniform(0.1, 0.9), 2),
        "Arrhythmia Risk": round(random.uniform(0.1, 0.8), 2),
        "Atrial Fibrillation": round(random.uniform(0.05, 0.7), 2)
    }
    return preds

# --------------------------------
# عرض النتائج
# --------------------------------
if hea and dat:
    record_name = hea.name.replace(".hea", "")
    ecg_id = extract_id(record_name)

    # محاكاة إشارة ECG
    fs = 500
    t = np.linspace(0, 2, fs*2)
    ecg_sim = np.sin(2 * np.pi * 1.7 * t) + 0.3*np.sin(2 * np.pi * 3.4 * t)
    ecg_clean = clean_ecg(ecg_sim)

    # ---- رسم الإشارة ----
    st.subheader("🫀 ECG Signal Visualization")
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.plot(t[:1000], ecg_clean[:1000], color="#00aaff", linewidth=1.2)
    ax.set_facecolor("#0b132b")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Amplitude (mV)", color="white")
    ax.grid(alpha=0.2)
    ax.set_title("Filtered ECG Signal (CNN + LSTM Feature Extraction)", color="#00aaff", fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

    # ---- شرح أجزاء الموجة ----
    st.markdown("""
    **Wave Explanation:**  
    - 🟢 **P Wave**: Atrial depolarization (start of heart contraction)  
    - 🔵 **QRS Complex**: Ventricular depolarization (strongest part of ECG)  
    - 🟣 **T Wave**: Ventricular repolarization (recovery phase)
    """)

    # ---- تحليل باستخدام نموذج محاكى ----
    preds = fake_cnn_lstm_prediction(ecg_id)
    st.subheader("🧠 CNN + LSTM AI Prediction Results")

    for disease, prob in preds.items():
        color = "#ff4d4d" if prob > 0.6 else "#4caf50"
        msg = "⚠️ High Risk Detected!" if prob > 0.6 else "✅ Normal / Low Risk"
        st.markdown(f"""
        <div style='background:{color};padding:15px;border-radius:10px;text-align:center;font-size:17px;color:white'>
            <b>{disease}</b><br>{msg}<br><b>Probability:</b> {prob*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

        # رسم شريط الاحتمال
        fig2, ax2 = plt.subplots(figsize=(6, 1))
        ax2.barh([""], [prob], color=color)
        ax2.set_xlim(0, 1)
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.text(prob - 0.1, 0, f"{prob*100:.1f}%", color="white", fontsize=12, fontweight="bold")
        fig2.patch.set_alpha(0)
        st.pyplot(fig2)
        plt.close(fig2)

    # ---- ملاحظات بعد التحليل ----
    st.markdown("""
    ---
    ### 🧩 Observations:
    - The CNN layer extracts spatial ECG waveform features.  
    - The LSTM layer captures temporal dependencies (beat-to-beat variations).  
    - Combined output predicts potential cardiac abnormalities like **Pre-Stroke** or **Arrhythmia**.  
    - This simulation mimics a real clinical AI diagnostic pipeline.
    """)

else:
    st.info("Please upload both `.hea` and `.dat` files to start the AI ECG analysis.")

# --------------------------------
# التذييل
# --------------------------------
st.markdown("---")
st.caption("💙 2025 | Smart Medical Simulation — AI-Powered ECG Diagnostic Prototype")
