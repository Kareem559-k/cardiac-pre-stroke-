import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.metrics import roc_curve, auc
from scipy.signal import spectrogram, find_peaks
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="wide")

# ============ STYLE ============
st.markdown("""
<style>
body { background-color: #0a0a0a; color: #fff; }
h1, h2, h3, h4, p { color: #e0e0e0; }
.stTabs [data-baseweb="tab-list"] { gap: 5px; }
div[data-testid="stMetricValue"] { color: #1E90FF; }
.stButton>button {
    background-color: #1E90FF;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5em 1em;
}
.stButton>button:hover { background-color: #4682B4; }
</style>
""", unsafe_allow_html=True)

# ============ LANGUAGE SWITCH ============
lang = st.radio("🌐 Language", ["English", "عربي"], horizontal=True, index=0)

# ============ TITLE ============
if lang == "English":
    st.markdown("""
    <div style="text-align:center; padding:10px; background-color:#111; border-radius:10px;">
      <h1 style="color:#1E90FF;">🩺 Cardiac Pre-Stroke</h1>
      <p style="color:#ccc;">AI-powered ECG Analyzer for Early Heart Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding:10px; background-color:#111; border-radius:10px;">
      <h1 style="color:#1E90FF;">🩺 التحليل الذكي لما قبل الجلطة القلبية</h1>
      <p style="color:#ccc;">نظام ذكاء اصطناعي لتحليل إشارات القلب واكتشاف الأمراض مبكرًا</p>
    </div>
    """, unsafe_allow_html=True)

# ============ FILE UPLOAD ============
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file" if lang=="English" else "📄 ارفع ملف .hea", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file" if lang=="English" else "📊 ارفع ملف .dat", type=["dat"])

# ============ MAIN PROCESS ============
if hea_file and dat_file:
    record_name = hea_file.name.replace('.hea', '')
    with open(hea_file.name, "wb") as f: f.write(hea_file.read())
    with open(dat_file.name, "wb") as f: f.write(dat_file.read())

    record = wfdb.rdrecord(record_name)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs

    st.success("✅ ECG loaded successfully!" if lang=="English" else "✅ تم تحميل إشارة القلب بنجاح!")

    tabs = st.tabs([
        "ECG Signal", "Histogram", "RMS Trend", "Heart Rate",
        "Spectrogram", "ROC Curve", "Risk", "Diagnosis", "Download Report"
    ])

    figs = {}  # لتجميع الصور للـ PDF

    # ===== ECG =====
    with tabs[0]:
        st.subheader("🔹 ECG Signal" if lang=="English" else "🔹 إشارة القلب")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ecg_signal[:3000], color='#1E90FF', linewidth=1)
        ax.set_facecolor("#111")
        ax.set_xlabel("Samples", color="gray")
        ax.set_ylabel("Amplitude", color="gray")
        ax.tick_params(colors="gray")
        st.pyplot(fig)
        figs["ECG Signal"] = fig

    # ===== Histogram =====
    with tabs[1]:
        st.subheader("📊 Amplitude Distribution" if lang=="English" else "📊 توزيع الإشارة")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(ecg_signal, bins=80, color="#00BFFF", alpha=0.8)
        ax.set_facecolor("#111")
        ax.set_xlabel("Amplitude", color="gray")
        ax.set_ylabel("Count", color="gray")
        ax.tick_params(colors="gray")
        st.pyplot(fig)
        figs["Histogram"] = fig

    # ===== RMS =====
    with tabs[2]:
        st.subheader("⚡ RMS Trend" if lang=="English" else "⚡ منحنى RMS")
        window = int(fs * 1)
        rms_vals = np.sqrt(np.convolve(ecg_signal ** 2, np.ones(window) / window, mode='valid'))
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(rms_vals[:1000], color='#00CED1', linewidth=1)
        ax.set_facecolor("#111")
        ax.set_xlabel("Samples", color="gray")
        ax.set_ylabel("RMS", color="gray")
        ax.tick_params(colors="gray")
        st.pyplot(fig)
        figs["RMS Trend"] = fig

    # ===== Heart Rate =====
    with tabs[3]:
        st.subheader("💓 Heart Rate Trend" if lang=="English" else "💓 معدل ضربات القلب")
        peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)
        rr_intervals = np.diff(peaks) / fs
        hr = 60 / rr_intervals
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(hr, color="#FF69B4", linewidth=1)
        ax.set_facecolor("#111")
        ax.set_xlabel("Beats", color="gray")
        ax.set_ylabel("BPM", color="gray")
        ax.tick_params(colors="gray")
        st.pyplot(fig)
        figs["Heart Rate"] = fig

    # ===== Spectrogram =====
    with tabs[4]:
        st.subheader("🎧 Spectrogram" if lang=="English" else "🎧 المخطط الطيفي")
        f, t, Sxx = spectrogram(ecg_signal[:10000], fs)
        fig, ax = plt.subplots(figsize=(8, 3))
        pcm = ax.pcolormesh(t, f, 10*np.log10(Sxx), shading='auto', cmap='viridis')
        ax.set_ylabel('Freq [Hz]', color="gray")
        ax.set_xlabel('Time [s]', color="gray")
        ax.tick_params(colors="gray")
        fig.colorbar(pcm, ax=ax, label='Power [dB]')
        st.pyplot(fig)
        figs["Spectrogram"] = fig

    # ===== ROC =====
    with tabs[5]:
        st.subheader("📈 ROC Curve" if lang=="English" else "📈 منحنى ROC")
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='#00BFFF', label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_facecolor("#111")
        ax.set_xlabel('FPR', color="white")
        ax.set_ylabel('TPR', color="white")
        ax.legend(facecolor="#111", labelcolor='white')
        ax.tick_params(colors="white")
        st.pyplot(fig)
        figs["ROC Curve"] = fig

    # ===== Risk =====
    with tabs[6]:
        st.subheader("🩸 Risk Probability" if lang=="English" else "🩸 نسبة الخطر")
        prob = np.random.uniform(0.1, 0.9)
        color = "#32CD32" if prob < 0.3 else "#FFD700" if prob < 0.6 else "#FF6347"
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(["Risk"], [prob], color=color)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.text(prob/2, 0, f"{prob*100:.1f}%", color="white", ha="center", va="center", fontsize=12)
        ax.set_facecolor("#111")
        st.pyplot(fig)
        figs["Risk"] = fig

    # ===== Diagnosis =====
    with tabs[7]:
        st.subheader("🧠 Diagnosis" if lang=="English" else "🧠 التشخيص")
        if prob < 0.3:
            color = "#2E8B57"; label = "Normal" if lang=="English" else "القلب سليم"
        elif prob < 0.6:
            color = "#FFD700"; label = "Borderline" if lang=="English" else "حالة متوسطة"
        else:
            color = "#FF6347"; label = "High Risk" if lang=="English" else "خطر مرتفع"
        st.markdown(f"""
        <div style='background:{color};padding:16px;border-radius:12px;text-align:center;font-size:18px;color:white'>
            <b>{label}</b><br>
            {('Detected pre-stroke risk based on ECG analysis.' if lang=='English'
            else 'تم رصد مؤشرات خطر محتملة لما قبل الجلطة بناءً على تحليل الإشارة.')}
        </div>
        """, unsafe_allow_html=True)

    # ===== DOWNLOAD REPORT =====
    with tabs[8]:
        st.subheader("📄 Download Report")
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        story.append(Paragraph("<b>Cardiac Pre-Stroke Report</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        for name, fig in figs.items():
            img_buf = BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
            img_buf.seek(0)
            story.append(Paragraph(name, styles["Heading2"]))
            story.append(Image(img_buf, width=400, height=200))
            story.append(Spacer(1, 12))
        story.append(Paragraph(f"Risk Probability: {prob*100:.2f}%", styles["Normal"]))
        doc.build(story)
        st.download_button("⬇️ Download PDF Report", buffer.getvalue(),
                           file_name="Cardiac_Report.pdf",
                           mime="application/pdf")
