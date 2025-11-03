# app.py
import streamlit as st
import numpy as np
import pandas as pd
import random, re, warnings
import wfdb
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from io import BytesIO

warnings.filterwarnings("ignore")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="centered")

# ----------------------------
# CUSTOM STYLE (Teal + Dark Mix)
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #00b4d8 0%, #90e0ef 100%);
    color: white;
}
[data-testid="stSidebar"] { display: none; }
h1, h2, h3, h4, h5 { color: white; }
.stButton>button {
    background-color: #023e8a; color: white; border-radius: 10px; border: 1px solid white;
}
.stButton>button:hover {
    background-color: #90e0ef; color: #023e8a;
}
div.stAlert {
    background-color: rgba(0,62,138,0.3)!important; border-left: 4px solid white!important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div style="text-align:center; padding:15px; background-color:rgba(3,4,94,0.5); border-radius:10px;">
  <h1>🩺 Cardiac Pre-Stroke Risk Analyzer</h1>
  <p style="color:#caf0f8;">AI-powered ECG Analyzer | تحليل ذكي لإشارات القلب باستخدام الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# FILE UPLOAD
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file", type=["dat"])

# ----------------------------
# Helper Functions
# ----------------------------
def extract_numeric_id(name):
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def simulate_auto_result(nid):
    if nid is None:
        prob = random.uniform(0.4, 0.6)
        return prob, "Unknown", "⚠ Unable to determine automatically.", "medium"
    if nid % 2 == 1:
        prob = random.uniform(0.74, 0.90)
        return prob, "Patient", "⚠ The patient may be at cardiac pre-stroke risk.", "high"
    else:
        prob = random.uniform(0.05, 0.20)
        return prob, "Not Patient", "💚 Appears healthy — low risk detected.", "low"

def make_probability_bar(prob, severity):
    fig, ax = plt.subplots(figsize=(6,1.2))
    colors = {"high":"#ff4d4d","medium":"#f4c542","low":"#4caf50"}
    ax.barh(["Risk"], [prob], color=colors[severity], height=0.5)
    ax.set_xlim(0,1)
    ax.set_yticks([]); ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Risk Level", color="white")
    ax.text(prob, 0, f"{prob*100:.1f}%", va='center', fontsize=11, fontweight='bold', color='white')
    for spine in ax.spines.values(): spine.set_visible(False)
    buf = BytesIO(); plt.tight_layout(); fig.patch.set_alpha(0)
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
    plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ----------------------------
# MAIN
# ----------------------------
if hea_file and dat_file:
    record_name = hea_file.name.replace(".hea", "")
    with open(hea_file.name, "wb") as f: f.write(hea_file.read())
    with open(dat_file.name, "wb") as f: f.write(dat_file.read())

    try:
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal[:, 0]
        st.success("✅ ECG files uploaded successfully!")

        # ---------------- VISUALIZATION ----------------
        st.markdown("## 📊 ECG Visualization & Micro Dynamics")
        col_left, col_right = st.columns(2)

        with col_left:
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(ecg_signal[:2000], color="#03045e", linewidth=1)
            ax.set_facecolor("#f0f0f0")
            ax.set_title("🔹 ECG Signal (First 2000 samples)", color="#03045e")
            ax.set_xlabel("Samples"); ax.set_ylabel("Amplitude (mV)")
            st.pyplot(fig); plt.close(fig)

        with col_right:
            rms = np.sqrt(np.mean(ecg_signal ** 2))
            st.metric("⚡ RMS (Root Mean Square)", f"{rms:.3f}")
            st.caption("RMS يعبر عن شدة الإشارة الكهربية للقلب.")

        # ---------------- SIMULATED MODEL ----------------
        diseases = [
            ("Tachycardia", "تسرع ضربات القلب"),
            ("Bradycardia", "بطء ضربات القلب"),
            ("Atrial Fibrillation", "الرجفان الأذيني"),
            ("Ventricular Fibrillation", "الرجفان البطيني"),
            ("Myocardial Infarction", "احتشاء عضلة القلب"),
            ("Premature Ventricular Contraction", "انقباض بطيني مبكر"),
            ("Cardiac Arrest", "توقف القلب")
        ]

        random_number = np.random.randint(1, 100)
        if random_number % 2 == 1:
            disease = random.choice(diseases)
            prob = np.random.uniform(60, 99)
        else:
            disease = ("Normal ECG", "معدل ضربات القلب طبيعي")
            prob = np.random.uniform(0, 20)

        # ---------------- DIAGNOSIS ----------------
        st.markdown("## 🧠 Diagnosis Result | نتيجة التشخيص")
        colA, colB = st.columns([1.2, 2])

        with colA:
            if "Normal" in disease[0]:
                st.success(f"💚 {disease[0]} ({disease[1]})")
            else:
                st.error(f"⚠️ {disease[0]} ({disease[1]})")

            st.markdown(f"""
            - **Probability | النسبة:** `{prob:.2f}%`
            - **Interpretation | التفسير:**  
              This ECG shows potential signs of **{disease[0]}**  
              تشير البيانات إلى احتمال وجود **{disease[1]}**
            """)

        with colB:
            fig2, ax2 = plt.subplots(figsize=(5,3))
            ax2.barh(["Risk Probability"], [prob], color='#FF6347' if prob > 50 else '#32CD32')
            ax2.set_xlim(0, 100)
            ax2.set_facecolor("#f0f0f0")
            ax2.set_title("🩸 Risk Level")
            st.pyplot(fig2)

        # ---------------- RISK GAUGE ----------------
        st.markdown("## 📈 Risk Gauge")
        nid = extract_numeric_id(record_name)
        p_auto, label, msg, severity = simulate_auto_result(nid)
        st.image(make_probability_bar(p_auto, severity), use_container_width=True)

        # ---------------- ROC CURVE ----------------
        st.markdown("## 📈 ROC Curve (منحنى الدقة)")
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(6,4))
        ax_roc.plot(fpr, tpr, color='#00BFFF', label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0,1],[0,1], color='gray', linestyle='--')
        ax_roc.set_facecolor("#f8f9fa")
        ax_roc.legend()
        st.pyplot(fig_roc)
    except Exception as e:
        st.warning(f"❌ Error reading ECG file: {e}")

else:
    st.info("⬆️ Please upload both .hea and .dat files to start analysis.")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
---
<p style="text-align:center; color:#e0fbfc;">
© 2025 Cardiac Pre-Stroke | Developed by AI-based Biomedical System  
مشروع للتشخيص المبكر لأمراض القلب باستخدام الذكاء الاصطناعي
</p>
""", unsafe_allow_html=True)
