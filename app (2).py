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
# تنسيق داكن احترافي
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
st.title("💙 Cardiac Pre-Stroke & Multi-Disease Analyzer")
st.caption("Simulated CNN + LSTM ECG diagnostic prototype — For demonstration only.")

# --------------------------------
# رفع ملفات ECG
# --------------------------------
st.subheader("📤 Upload ECG Record (.hea + .dat)")
hea = st.file_uploader("Upload .hea file", type=["hea"])
dat = st.file_uploader("Upload .dat file", type=["dat"])

# --------------------------------
# دوال مساعدة
# --------------------------------
def extract_id(name):
    """Extract numeric ID from filename (used to simulate patient condition)."""
    m = re.search(r'(\d+)(?!.*\d)', name)
    return int(m.group(1)) if m else random.randint(1, 99)

def clean_ecg(sig):
    """Apply low-pass Butterworth filter to denoise ECG signal."""
    b, a = butter(3, 0.1)
    return filtfilt(b, a, sig)

def simulate_cnn_lstm_prediction(ecg_id):
    """Simulate disease prediction using odd/even logic."""
    random.seed(ecg_id)
    diseases = [
        "Pre-Stroke Risk",
        "Arrhythmia",
        "Myocardial Infarction",
        "Atrial Fibrillation",
        "Bradycardia",
        "Tachycardia"
    ]

    preds = {}
    if ecg_id % 2 == 1:  # فردي = مريض
        for d in diseases:
            preds[d] = round(random.uniform(0.65, 0.9), 2)
    else:  # زوجي = سليم
        for d in diseases:
            preds[d] = round(random.uniform(0.05, 0.3), 2)

    return preds

# --------------------------------
# Main Logic
# --------------------------------
if hea and dat:
    record_name = hea.name.replace(".hea", "")
    ecg_id = extract_id(record_name)

    st.markdown(f"### 📁 Record ID Detected: `{ecg_id}`")
    st.caption("Odd → Patient  |  Even → Healthy")

    # ---- محاكاة إشارة ECG ----
    fs = 500  # التردد (عدد العينات في الثانية)
    t = np.linspace(0, 2, fs*2)
    ecg_sim = np.sin(2 * np.pi * 1.3 * t) + 0.2*np.sin(2 * np.pi * 3.2 * t)
    ecg_clean = clean_ecg(ecg_sim)

    # ---- رسم إشارة ECG ----
    st.subheader("🫀 ECG Signal Visualization")
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.plot(t[:1000], ecg_clean[:1000], color="#00aaff", linewidth=1.3)
    ax.set_facecolor("#0b132b")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Amplitude (mV)", color="white")
    ax.set_title("Filtered ECG Signal (CNN + LSTM Feature Extraction)", color="#00aaff", fontsize=12)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    # ---- شرح أجزاء الموجة ----
    st.markdown("""
    ### 📘 ECG Wave Explanation:
    - 🟢 **P Wave** → Atrial depolarization (atria start contracting)
    - 🔵 **QRS Complex** → Ventricular depolarization (strong contraction)
    - 🟣 **T Wave** → Ventricular repolarization (recovery phase)
    """)

    # ---- تحليل الأمراض ----
    st.subheader("🧠 Simulated CNN + LSTM AI Predictions")
    preds = simulate_cnn_lstm_prediction(ecg_id)

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

    # ---- ملاحظات ----
    st.markdown("""
    ---
    ### 🧩 Observations:
    - The CNN module simulates spatial ECG feature extraction.  
    - The LSTM module models heartbeat-to-heartbeat changes over time.  
    - For **odd record IDs**, AI detects strong probabilities for multiple cardiac issues.  
    - For **even record IDs**, AI shows stable readings indicating healthy signals.  
    """)

else:
    st.info("Please upload both `.hea` and `.dat` files to start the ECG AI simulation.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("💙 2025 | Smart Medical Simulation — AI-Powered ECG Diagnostic Prototype")
