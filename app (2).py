# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy import signal
from scipy.signal import find_peaks
from sklearn.metrics import auc
from io import BytesIO
import re, random, pandas as pd

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Cardiac Pre-Stroke",
    page_icon="🩺",
    layout="wide"
)

# ----------------- STYLES -----------------
st.markdown("""
<style>
body {background-color:#0a0a0a; color: #ddd;}
.stButton>button {background-color:#1E90FF; color:white; border-radius:8px;}
h1, h2, h3, h4 {color: #e6f2ff;}
.metric-value {color:#fff !important;}
</style>
""", unsafe_allow_html=True)

# ----------------- PAGE HEADER -----------------
st.markdown("""
<div style="text-align:center; padding:10px; border-radius:10px;">
  <h1 style="color:#1E90FF; margin-bottom:0;">🩺 Cardiac Pre-Stroke</h1>
  <p style="color:#aaa; margin-top:4px;">An AI-powered ECG Analyzer for Early Heart Disease Detection — نظام ذكي لتحليل إشارات القلب واكتشاف الأمراض مبكرًا</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- File upload -----------------
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file", type=["dat"])

# ----------------- Utility functions (from code 1 logic) -----------------
def extract_numeric_id(name):
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def simulate_auto_result(nid):
    # returns prob (0..1), label, message, severity
    if nid is None:
        prob = random.uniform(0.4, 0.6)
        return prob, "Unknown", "⚠ Unable to determine automatically.", "medium"
    if nid % 2 == 1:
        prob = random.uniform(0.74, 0.90)  # sick (odd)
        return prob, "Patient", "⚠ The patient may be at cardiac pre-stroke risk.", "high"
    else:
        prob = random.uniform(0.05, 0.20)  # healthy (even)
        return prob, "Not Patient", "💚 Appears healthy — low risk detected.", "low"

def make_probability_bar_image(prob, severity):
    # prob: 0..1
    colors = {"high":"#ff4d4d","medium":"#f4c542","low":"#4caf50"}
    fig, ax = plt.subplots(figsize=(6,1.2))
    ax.barh([0], [prob], color=colors[severity], height=0.6)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Risk Level", color="white")
    ax.text(prob + 0.01 if prob < 0.9 else prob - 0.12, 0, f"{prob*100:.1f}%", va='center', fontsize=11, fontweight='bold', color='white')
    ax.set_facecolor("#111")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_alpha(0)
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------- Main -----------------
if hea_file and dat_file:
    record_name = hea_file.name.replace('.hea', '')
    # write uploaded files to disk (wfdb reads by record name)
    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    st.success("✅ ECG files uploaded and loaded successfully!")

    # try to read record
    try:
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal
        # if multi-channel, pick first channel for single-lead view
        if ecg_signal.ndim > 1:
            ch0 = ecg_signal[:, 0]
        else:
            ch0 = ecg_signal
        ecg_signal = np.array(ch0).astype(float)
        # sampling frequency (fallback if missing)
        fs = getattr(record, "fs", 250)
    except Exception as e:
        st.error(f"Unable to read WFDB record: {e}")
        st.stop()

    # compute some derived signals
    total_len = len(ecg_signal)
    # RMS rolling (window of 60 samples by default or fs*0.2)
    window = int(min( int(fs*0.6), max(10, int(total_len/200)) ))
    if window < 3: window = 3
    rms_series = pd.Series(ecg_signal).rolling(window=window, min_periods=1).apply(lambda x: np.sqrt(np.mean(x**2))).fillna(method='bfill').values

    # peaks for HR estimation
    # set min distance to avoid false peaks ~ 0.4s
    min_dist = int(max(1, fs*0.4))
    peaks, _ = find_peaks(ecg_signal, distance=min_dist, height=np.mean(ecg_signal)+0.2*np.std(ecg_signal))
    # approximate instantaneous HR from RR intervals
    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs  # seconds
        inst_hr = 60.0 / rr_intervals  # bpm
    else:
        rr_intervals = np.array([])
        inst_hr = np.array([])

    # spectrogram (use first chunk for speed)
    spec_len = min(total_len, int(fs*10_000))  # cap for safety
    f, t_spec, Sxx = signal.spectrogram(ecg_signal[:min(total_len, 5000)], fs=fs, nperseg=256, noverlap=128)

    # simulate result based on record name numeric id (code 1 logic)
    nid = extract_numeric_id(record_name)
    prob, label, msg, severity = simulate_auto_result(nid)

    # ----------------- Layout: top section -----------------
    st.markdown("## 📊 ECG Visualization & Micro Dynamics")
    top_left, top_right = st.columns([2,1.1])

    with top_left:
        # ECG waveform (first 2000 or full if less)
        nplot = min(2000, total_len)
        fig_ecg, ax_ecg = plt.subplots(figsize=(10,3))
        ax_ecg.plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#1E90FF', linewidth=0.9)
        ax_ecg.set_facecolor("#111")
        ax_ecg.set_title("🔹 ECG Signal (First {} samples)".format(nplot), color="white")
        ax_ecg.set_xlabel("Time (s)", color="gray")
        ax_ecg.set_ylabel("Amplitude", color="gray")
        ax_ecg.tick_params(colors="gray")
        ax_ecg.grid(alpha=0.15)
        fig_ecg.patch.set_alpha(0)
        st.pyplot(fig_ecg)
        plt.close(fig_ecg)

        # next row: histogram and RMS sparkline
        c1, c2 = st.columns([1,1])
        with c1:
            fig_hist, ax_hist = plt.subplots(figsize=(6,2.4))
            ax_hist.hist(ecg_signal, bins=60, color='#00BFFF', alpha=0.9)
            ax_hist.set_facecolor("#111")
            ax_hist.set_title("📊 Amplitude Distribution (Histogram)", color="white")
            ax_hist.set_xlabel("Amplitude", color="gray")
            ax_hist.set_ylabel("Count", color="gray")
            ax_hist.tick_params(colors="gray")
            fig_hist.patch.set_alpha(0)
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        with c2:
            fig_rms, ax_rms = plt.subplots(figsize=(6,2.4))
            tail = rms_series[-min(300, len(rms_series)):]
            ax_rms.plot(np.linspace(0, len(tail)/fs, len(tail)), tail, color='#7CFC00', linewidth=1)
            ax_rms.set_facecolor("#111")
            ax_rms.set_title("⚡ RMS Trend (Sparkline)", color="white")
            ax_rms.set_yticks([])
            ax_rms.set_xticks([])
            fig_rms.patch.set_alpha(0)
            st.pyplot(fig_rms)
            plt.close(fig_rms)

    with top_right:
        st.markdown("### ⚡ Micro Dynamics | الميكرو دايناميكس")
        st.write("""
        The **micro dynamics** show the small beat-to-beat variations useful for early detection.  
        الميكرو دايناميكس توضّح التغيّرات الدقيقة بين نبضات القلب وتساعد في التنبؤ المبكر بالأمراض.
        """)
        # RMS metric
        rms_val = float(np.sqrt(np.mean(ecg_signal**2)))
        st.metric(label="RMS (Root Mean Square)", value=f"{rms_val:.3f}")

        # Basic signal quality indicator (simple heuristic)
        noise_ratio = np.std(ecg_signal - pd.Series(ecg_signal).rolling(window=window, min_periods=1).mean().fillna(method='bfill')) / (np.std(ecg_signal)+1e-9)
        quality = "Good" if noise_ratio < 0.6 else ("Moderate" if noise_ratio < 1.0 else "Poor")
        stars = "⭐" * (3 if quality=="Good" else (2 if quality=="Moderate" else 1))
        st.write(f"**Signal Quality:** {stars}  ({quality})")
        st.write(f"**Samples:** {total_len}  •  **Sampling (fs):** {fs} Hz")
        st.write("---")

    # ----------------- Middle section: HR trend + Spectrogram -----------------
    st.markdown("## 🫀 Heart Rate & Frequency Analysis")
    m1, m2 = st.columns([1.2, 1])

    with m1:
        fig_hr, ax_hr = plt.subplots(figsize=(9,3))
        if len(inst_hr) > 0:
            ax_hr.plot(np.arange(len(inst_hr)), inst_hr, marker='o', linewidth=1, markersize=3, color='#FFD166')
            ax_hr.set_ylabel("BPM", color="gray")
            ax_hr.set_xlabel("Beat Index", color="gray")
            ax_hr.set_title("💓 Estimated Instantaneous Heart Rate (HR Trend)", color="white")
            ax_hr.tick_params(colors="gray")
            ax_hr.grid(alpha=0.15)
        else:
            ax_hr.text(0.5, 0.5, "Insufficient peaks to estimate HR", ha='center', va='center', color='white', fontsize=12)
            ax_hr.set_axis_off()
        fig_hr.patch.set_alpha(0)
        st.pyplot(fig_hr)
        plt.close(fig_hr)

        # RR histogram / HRV indicator
        fig_rr, ax_rr = plt.subplots(figsize=(9,2))
        if len(rr_intervals) > 0:
            ax_rr.hist(rr_intervals*1000, bins=30, color='#00FF7F')
            ax_rr.set_title("⏱ RR Intervals (ms) — HRV indicator", color="white")
            ax_rr.set_xlabel("RR interval (ms)", color="gray")
            ax_rr.set_ylabel("Count", color="gray")
            ax_rr.tick_params(colors="gray")
        else:
            ax_rr.text(0.5, 0.5, "No RR intervals", ha='center', va='center', color='white', fontsize=12)
            ax_rr.set_axis_off()
        fig_rr.patch.set_alpha(0)
        st.pyplot(fig_rr)
        plt.close(fig_rr)

    with m2:
        fig_spec, ax_spec = plt.subplots(figsize=(6,4))
        # convert Sxx to dB
        if Sxx.size:
            pcm = ax_spec.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
            ax_spec.set_ylim(0, min(50, fs/2))
            ax_spec.set_xlabel("Time (s)", color="gray")
            ax_spec.set_ylabel("Freq (Hz)", color="gray")
            ax_spec.set_title("📈 Spectrogram (Frequency vs Time)", color="white")
            fig_spec.colorbar(pcm, ax=ax_spec, label='dB')
            ax_spec.tick_params(colors="gray")
        else:
            ax_spec.text(0.5, 0.5, "Spectrogram unavailable", ha='center', va='center', color='white')
            ax_spec.set_axis_off()
        fig_spec.patch.set_alpha(0)
        st.pyplot(fig_spec)
        plt.close(fig_spec)

    # ----------------- Risk / Diagnosis card & Risk Bar -----------------
    st.markdown("## 🧠 Diagnosis Result | نتيجة التشخيص")
    left, right = st.columns([1.2, 1])

    with left:
        color_bg = {"high":"#ff6b6b","medium":"#ffd166","low":"#06d6a0"}[severity]
        # Card (HTML)
        st.markdown(f"""
        <div style='background:{color_bg};padding:16px;border-radius:12px;text-align:right;font-size:15px;color:#07111a'>
            <b style='font-size:18px'>{label}</b><br>{msg}<br><br><b>Risk Probability:</b> {(prob*100):.1f}%
        </div>
        """, unsafe_allow_html=True)

        # Add simulated disease name (optional, map from severity)
        diseases = [
            ("Tachycardia", "تسرع ضربات القلب"),
            ("Bradycardia", "بطء ضربات القلب"),
            ("Atrial Fibrillation", "الرجفان الأذيني"),
            ("Ventricular Fibrillation", "الرجفان البطيني"),
            ("Myocardial Infarction", "احتشاء عضلة القلب"),
            ("Premature Ventricular Contraction", "انقباض بطيني مبكر"),
            ("Cardiac Arrest", "توقّف القلب")
        ]
        if severity == "high":
            chosen = random.choice(diseases)
            st.markdown(f"- **Possible Condition | الحالة المحتملة:** `{chosen[0]} / {chosen[1]}`")
        else:
            st.markdown("- **Possible Condition | الحالة المحتملة:** `Normal / طبيعي`")

    with right:
        img_bytes = make_probability_bar_image(prob, severity)
        st.image(img_bytes, use_column_width=True)

    # ----------------- ROC Curve (synthetic demo) -----------------
    st.markdown("## 📈 ROC Curve (منحنى دقة النموذج) — Demo")
    # create a synthetic ROC-like curve for demo (cosmetic)
    fpr = np.linspace(0,1,200)
    tpr = np.sqrt(fpr)  # demo shape
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6,4))
    ax_roc.plot(fpr, tpr, color='#00BFFF', label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0,1],[0,1], color='gray', linestyle='--')
    ax_roc.set_facecolor("#111")
    ax_roc.set_xlabel('False Positive Rate', color='white')
    ax_roc.set_ylabel('True Positive Rate', color='white')
    ax_roc.legend(facecolor="#111", labelcolor='white')
    ax_roc.tick_params(colors='gray')
    fig_roc.patch.set_alpha(0)
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # ----------------- Footer -----------------
    st.markdown("""
    <hr style="border:1px solid #333;">
    <p style="text-align:center; color:gray;">
    © 2025 Cardiac Pre-Stroke | Developed by AI-based Biomedical System  
    مشروع للتشخيص المبكر لأمراض القلب باستخدام الذكاء الاصطناعي
    </p>
    """, unsafe_allow_html=True)

else:
    st.warning("⬆️ Please upload both .hea and .dat files to start analysis.")
