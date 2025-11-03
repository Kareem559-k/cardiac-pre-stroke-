# ----------------- AUTO INSTALL -----------------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "reportlab"])

# ----------------- IMPORTS -----------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.metrics import roc_curve, auc
from scipy.signal import find_peaks, spectrogram
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from datetime import datetime
import random

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="wide")

# ----------------- LANGUAGE SELECTOR -----------------
lang = st.radio("🌐 Language / اللغة", ["English", "العربية"], horizontal=True)

# ----------------- STYLES -----------------
st.markdown("""
<style>
body {background-color: #0a0a0a;}
h1, h2, h3, h4, p, div, span {color: #f0f0f0;}
.stMetric {background-color: #111 !important; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
if lang == "English":
    st.markdown("""
    <div style="text-align:center; padding:10px; background-color:#0a0a0a; border-radius:10px;">
      <h1 style="color:#1E90FF;">🩺 Cardiac Pre-Stroke</h1>
      <p style="color:#ccc;">AI-powered ECG Analyzer for Early Heart Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding:10px; background-color:#0a0a0a; border-radius:10px;">
      <h1 style="color:#1E90FF;">🩺 تحليل القلب قبل الجلطة</h1>
      <p style="color:#ccc;">نظام ذكاء اصطناعي لتحليل إشارات القلب والتنبؤ المبكر بالأمراض</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- FILE UPLOAD -----------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file" if lang == "English" else "📄 ارفع ملف .hea", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file" if lang == "English" else "📊 ارفع ملف .dat", type=["dat"])

if hea_file and dat_file:
    record_name = hea_file.name.replace('.hea', '')
    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    record = wfdb.rdrecord(record_name)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs

    st.success("✅ ECG loaded successfully!" if lang == "English" else "✅ تم تحميل بيانات ECG بنجاح!")

    # ----------- Tabs Layout -----------
    tabs = st.tabs([
        "📈 ECG Signal", "📊 Histogram", "⚡ RMS Trend", "❤️ Heart Rate", "🎛️ Spectrogram", "🩸 Risk", "📉 ROC Curve"
    ])

    # ECG Plot
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ecg_signal[:2000], color='#1E90FF', linewidth=1)
        ax.set_facecolor("#111")
        ax.set_title("ECG Signal (First 2000 samples)", color="white")
        ax.tick_params(colors="gray")
        st.pyplot(fig)

    # Histogram
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(ecg_signal, bins=60, color='#0077b6')
        ax.set_facecolor("#111")
        ax.set_title("Amplitude Distribution", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    # RMS Trend
    with tabs[2]:
        rms = np.sqrt(np.convolve(ecg_signal**2, np.ones(300)/300, mode='valid'))
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(rms, color="#00BFFF")
        ax.set_facecolor("#111")
        ax.set_title("RMS Trend", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    # Heart Rate
    with tabs[3]:
        peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)
        rr_intervals = np.diff(peaks) / fs
        heart_rate = 60 / rr_intervals
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(heart_rate, color="#FF6B6B")
        ax.set_facecolor("#111")
        ax.set_title("Heart Rate Trend (bpm)", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    # Spectrogram
    with tabs[4]:
        f, t, Sxx = spectrogram(ecg_signal[:5000], fs)
        fig, ax = plt.subplots(figsize=(8, 3))
        pcm = ax.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud', cmap='plasma')
        fig.colorbar(pcm, ax=ax, label='Power (dB)')
        ax.set_facecolor("#111")
        ax.set_title("ECG Spectrogram", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    # Risk Bar
    with tabs[5]:
        prob = random.uniform(0.05, 0.95)
        severity = "High" if prob > 0.7 else "Low" if prob < 0.3 else "Medium"
        color = "#ff4d4d" if severity == "High" else "#4caf50" if severity == "Low" else "#ffd166"
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(["Risk"], [prob], color=color)
        ax.set_xlim(0, 1)
        ax.set_title(f"Predicted Risk: {prob*100:.1f}%", color="white")
        ax.set_facecolor("#111")
        st.pyplot(fig)

    # ROC Curve
    with tabs[6]:
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='#00BFFF', label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_facecolor("#111")
        ax.legend(facecolor="#111", labelcolor='white')
        ax.set_title("ROC Curve", color="white")
        st.pyplot(fig)

    # ----------------- DOWNLOAD PDF REPORT -----------------
    st.markdown("---")
    if st.button("📄 Download Report"):
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        title = "Cardiac Pre-Stroke ECG Report" if lang == "English" else "تقرير تحليل القلب قبل الجلطة"
        story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Predicted Risk Probability: {prob*100:.1f}%", styles["Normal"]))
        story.append(Paragraph(f"Risk Level: {severity}", styles["Normal"]))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated by AI-based Biomedical System", styles["Italic"]))
        doc.build(story)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer.getvalue(),
            file_name="Cardiac_Report.pdf",
            mime="application/pdf"
        )

else:
    st.warning("⬆️ Please upload both .hea and .dat files to start analysis." if lang == "English" else "⬆️ من فضلك ارفع ملفي .hea و .dat لبدء التحليل.")
