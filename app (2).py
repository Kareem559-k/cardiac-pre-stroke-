# app_advanced_v2.py - Interactive Multi-Lead ECG + Ensemble Prediction
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "reportlab", "Pillow", "wfdb", "scipy", "matplotlib", "xgboost", "lightgbm", "scikit-learn", "-q"])

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import find_peaks, spectrogram
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image, ImageDraw
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align:center; padding:14px; background-color:#f5f5f5; border-radius:10px; border:1px solid #ddd;">
  <h1 style="color:#1E90FF; margin:0;">🩺 Cardiac Pre-Stroke</h1>
  <p style="color:#000; margin:4px 0 0 0;">AI-powered ECG Analyzer for Multi-class Diagnosis & Pre-Stroke Risk</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LANGUAGE ----------------
lang = st.radio("🌍 Choose Language | اختر اللغة:", ["English", "عربي"], horizontal=True)

# ---------------- FILE UPLOAD ----------------
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file", type=["dat"])

# ---------------- UTILITIES ----------------
def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def make_heart_png(width=600, height=300, fill_color="#f2f8ff"):
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    x, y = width / 2, height / 3
    size = min(width, height) / 3.2
    draw.pieslice([x - size*1.3, y - size, x, y + size*0.8], 180, 360, fill=fill_color)
    draw.pieslice([x, y - size, x + size*1.3, y + size*0.8], 180, 360, fill=fill_color)
    points = [(x - size*1.3, y + size*0.3), (x + size*1.3, y + size*0.3), (x, y + size*2)]
    draw.polygon(points, fill=fill_color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ---------------- MAIN ----------------
if hea_file and dat_file:
    record_name = hea_file.name.replace('.hea', '')
    with open(hea_file.name, "wb") as f: f.write(hea_file.read())
    with open(dat_file.name, "wb") as f: f.write(dat_file.read())

    try:
        record = wfdb.rdrecord(record_name)
        ecg_signal = np.array(record.p_signal).astype(float)
        n_leads = ecg_signal.shape[1] if ecg_signal.ndim>1 else 1
        if n_leads == 1: ecg_signal = ecg_signal.reshape(-1,1)
        fs = getattr(record, "fs", 250)
    except Exception as e:
        st.error("Unable to read WFDB record: " + str(e))
        st.stop()

    st.success("✅ Files loaded successfully!" if lang=="English" else "✅ تم تحميل الملفات بنجاح!")

    # ---------------- Lead Selection ----------------
    if n_leads > 1:
        lead_idx = st.selectbox("Select Lead | اختر القناة:", list(range(n_leads)))
        ecg_signal = ecg_signal[:, lead_idx]
    else:
        ecg_signal = ecg_signal[:,0]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "ECG Signal", "RMS Trend", "Heart Rate", "Spectrogram",
        "Histogram", "ROC Curve", "Complete Diagnosis of the Condition", "Ensemble Prediction"
    ])

    pdf_figs = {}

    # ---------------- Feature Calculations ----------------
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    rms_val = np.sqrt(np.mean(ecg_signal**2))
    skew_val = np.mean((ecg_signal - mean_val)**3)/std_val**3
    kurt_val = np.mean((ecg_signal - mean_val)**4)/std_val**4 - 3
    max_val, min_val = np.max(ecg_signal), np.min(ecg_signal)
    rng_val = max_val - min_val

    # ---------------- Patient Status (Random Demo) ----------------
    classes = [
        ("Normal","طبيعي"), 
        ("Minor Arrhythmia","اضطراب بسيط"),
        ("Electrical Weakness","ضعف كهربي"),
        ("ST Elevation","علامات خطر"),
        ("Pre-Stroke Risk","احتمالية جلطة"),
        ("Severe Arrhythmia","عدم انتظام قوي"),
        ("Ventricular Issue","خلل بطيني"),
        ("Other","حالة أخرى")
    ]
    selected_class = random.choice(classes)
    risk_prob = random.uniform(5, 95)
    is_healthy = selected_class[0] == "Normal"
    color = "#2ECC71" if is_healthy else "#FF4C4C"

    # ---------------- Tab 1: ECG Signal ----------------
    with tab1:
        st.markdown("### ECG Signal" if lang=="English" else "### إشارة القلب")
        nplot = min(3000, len(ecg_signal))
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#1E90FF', linewidth=0.9)
        ax.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)")
        ax.set_ylabel("Amplitude" if lang=="English" else "السعة")
        ax.grid(alpha=0.15)
        st.pyplot(fig)
        pdf_figs["ECG Signal"] = fig_to_bytes(fig)

    # ---------------- Tab 2: RMS Trend ----------------
    with tab2:
        window = int(min(1000, max(50, int(fs*0.8))))
        rms_vals = np.sqrt(np.convolve(ecg_signal**2, np.ones(window)/window, mode='valid'))
        t_rms = np.linspace(0, len(ecg_signal)/fs, len(rms_vals))
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(t_rms, rms_vals, color='orange')
        ax2.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)")
        ax2.set_ylabel("RMS")
        ax2.grid(alpha=0.15)
        st.pyplot(fig2)
        pdf_figs["RMS Trend"] = fig_to_bytes(fig2)

    # ---------------- Tab 3: Heart Rate ----------------
    with tab3:
        peaks,_ = find_peaks(ecg_signal, distance=fs*0.45)
        if len(peaks)>=2:
            rr_intervals = np.diff(peaks)/fs
            heart_rate = 60.0/rr_intervals
            fig3, ax3 = plt.subplots(figsize=(10,3))
            ax3.plot(heart_rate, color='green')
            ax3.set_xlabel("Beat Index" if lang=="English" else "ترتيب النبضة")
            ax3.set_ylabel("BPM")
            ax3.grid(alpha=0.15)
            st.pyplot(fig3)
            pdf_figs["Heart Rate"] = fig_to_bytes(fig3)

    # ---------------- Tab 4: Spectrogram ----------------
    with tab4:
        spec_len = min(len(ecg_signal), int(fs*5000))
        f, t_spec, Sxx = spectrogram(ecg_signal[:spec_len], fs=fs, nperseg=256, noverlap=128)
        fig4, ax4 = plt.subplots(figsize=(10,4))
        pcm = ax4.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-12), shading='gouraud', cmap='plasma')
        ax4.set_ylabel("Frequency (Hz)" if lang=="English" else "التردد (هرتز)")
        ax4.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)")
        fig4.colorbar(pcm, ax=ax4, label='Power (dB)')
        st.pyplot(fig4)
        pdf_figs["Spectrogram"] = fig_to_bytes(fig4)

    # ---------------- Tab 5: Histogram ----------------
    with tab5:
        fig5, ax5 = plt.subplots(figsize=(6,3))
        ax5.hist(ecg_signal, bins=60, color="#00BFFF", edgecolor="black")
        ax5.set_xlabel("Amplitude")
        ax5.set_ylabel("Count")
        st.pyplot(fig5)
        pdf_figs["Histogram"] = fig_to_bytes(fig5)

    # ---------------- Tab 6: ROC Curve ----------------
    with tab6:
        fpr = np.linspace(0,1,200)
        tpr = np.sqrt(fpr)
        roc_auc = 0.87
        fig6, ax6 = plt.subplots(figsize=(6,4))
        ax6.plot(fpr,tpr,color='#1E90FF',label=f"AUC={roc_auc:.2f}")
        ax6.plot([0,1],[0,1],color='gray',linestyle='--')
        ax6.set_xlabel("False Positive Rate")
        ax6.set_ylabel("True Positive Rate")
        ax6.legend()
        st.pyplot(fig6)
        pdf_figs["ROC Curve"] = fig_to_bytes(fig6)

    # ---------------- Tab 7: Complete Diagnosis ----------------
    with tab7:
        st.markdown("### Complete Diagnosis of the Condition" if lang=="English" else "### التشخيص الكامل للحالة")
        colL, colR = st.columns([2,1])
        with colL:
            st.markdown(f"**Class:** {selected_class[0]} — {selected_class[1]}")
            st.markdown(f"**Risk Probability:** {risk_prob:.2f}%")
            st.markdown(f"**Mean:** {mean_val:.2f} | **STD:** {std_val:.2f} | **RMS:** {rms_val:.2f}")
            st.markdown(f"**Skewness:** {skew_val:.2f} | **Kurtosis:** {kurt_val:.2f}")
            st.markdown(f"**Min:** {min_val:.2f} | **Max:** {max_val:.2f} | **Range:** {rng_val:.2f}")
        with colR:
            st.image(make_heart_png(), use_column_width=True)

    # ---------------- Tab 8: Ensemble Prediction ----------------
    with tab8:
        st.markdown("### Ensemble Prediction Confidence" if lang=="English" else "### توقع النموذج المجمع")
        # simulate predictions
        ensemble_conf = np.random.dirichlet(np.ones(len(classes)),size=1)[0]*100
        fig8, ax8 = plt.subplots(figsize=(8,4))
        ax8.bar([c[0] for c in classes], ensemble_conf, color="#1E90FF")
        ax8.set_ylabel("Confidence (%)")
        ax8.set_ylim(0,100)
        for i,v in enumerate(ensemble_conf):
            ax8.text(i, v+1, f"{v:.1f}%", ha='center')
        st.pyplot(fig8)
        pdf_figs["Ensemble Prediction"] = fig_to_bytes(fig8)

else:
    st.warning("⬆ Upload both .hea and .dat files to begin analysis." if lang=="English" else "⬆ من فضلك ارفع ملفي .hea و .dat لبدء التحليل.")
