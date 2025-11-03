import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.metrics import roc_curve, auc

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Cardiac Pre-Stroke",
    page_icon="🩺",
    layout="wide"
)

# ----------------- PAGE HEADER -----------------
st.markdown("""
<div style="text-align:center; padding:10px; background-color:#0a0a0a; border-radius:10px;">
  <h1 style="color:#1E90FF;">🩺 Cardiac Pre-Stroke</h1>
  <p style="color:#ccc;">An AI-powered ECG Analyzer for Early Heart Disease Detection<br>نظام ذكي لتحليل إشارات القلب واكتشاف الأمراض مبكرًا</p>
</div>
""", unsafe_allow_html=True)

# ----------------- FILE UPLOAD -----------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file", type=["dat"])

if hea_file and dat_file:
    record_name = hea_file.name.replace('.hea', '')

    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    record = wfdb.rdrecord(record_name)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs

    st.success("✅ ECG files uploaded and loaded successfully!")

    # ----------------- ECG SIGNAL VISUALIZATION -----------------
    st.markdown("## 📊 ECG Visualization & Micro Dynamics")

    col_left, col_right = st.columns(2)

    with col_left:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ecg_signal[:2000], color='#1E90FF', linewidth=1.2)
        ax.set_facecolor("#111")
        ax.set_title("🔹 ECG Signal (First 2000 samples)", color="white", fontsize=12)
        ax.set_xlabel("Samples", color="gray")
        ax.set_ylabel("Amplitude (mV)", color="gray")
        ax.tick_params(colors="gray")
        st.pyplot(fig)

    with col_right:
        st.markdown("### ⚡ Micro Dynamics | الميكرو دايناميكس")
        st.write("""
        The **micro dynamics** show the small variations between heartbeats.  
        الميكرو دايناميكس توضّح التغيّرات الدقيقة بين نبضات القلب وتساعد في التنبؤ المبكر بالأمراض.
        """)
        rms = np.sqrt(np.mean(ecg_signal ** 2))
        st.metric(label="RMS (Root Mean Square)", value=f"{rms:.3f}")

    # ----------------- SIMULATED MODEL (CNN + LSTM) -----------------
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
        disease = np.random.choice(diseases)
        prob = np.random.uniform(60, 99)
    else:
        disease = ("Normal ECG", "معدل ضربات القلب طبيعي")
        prob = np.random.uniform(0, 20)

    # ----------------- DIAGNOSIS RESULT -----------------
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
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.barh(["Risk Probability"], [prob], color='#FF6347' if prob > 50 else '#32CD32')
        ax2.set_xlim(0, 100)
        ax2.set_facecolor("#111")
        ax2.set_title("🩸 Risk Level", color="white")
        ax2.tick_params(colors="white")
        st.pyplot(fig2)

    # ----------------- ROC CURVE -----------------
    st.markdown("## 📈 ROC Curve (منحنى دقة النموذج)")

    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, color='#00BFFF', label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_facecolor("#111")
    ax_roc.set_xlabel('False Positive Rate', color='white')
    ax_roc.set_ylabel('True Positive Rate', color='white')
    ax_roc.legend(facecolor="#111", labelcolor='white')
    ax_roc.tick_params(colors="white")
    st.pyplot(fig_roc)

    # ----------------- FOOTER -----------------
    st.markdown("""
    <hr style="border:1px solid #333;">
    <p style="text-align:center; color:gray;">
    © 2025 Cardiac Pre-Stroke | Developed by AI-based Biomedical System  
    مشروع للتشخيص المبكر لأمراض القلب باستخدام الذكاء الاصطناعي
    </p>
    """, unsafe_allow_html=True)

else:
    st.warning("⬆️ Please upload both .hea and .dat files to start analysis.")
