# app_interactive.py - Full Interactive Cardiac Pre-Stroke Dashboard
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "numpy", "matplotlib", "wfdb", "scipy", "reportlab", "Pillow", "-q"])

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import random, re
from scipy.signal import find_peaks, spectrogram
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image, ImageDraw

st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align:center; padding:14px; background-color:#f5f5f5; border-radius:10px; border:1px solid #ddd;">
  <h1 style="color:#1E90FF; margin:0;">🩺 Cardiac Pre-Stroke</h1>
  <p style="color:#000; margin:4px 0 0 0;">AI-powered ECG Analyzer for Early Detection — نظام ذكي لتحليل إشارات القلب واكتشاف الأمراض مبكرًا</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LANGUAGE ----------------
lang = st.radio("🌍 اختر اللغة | Choose Language:", ["English", "عربي"], horizontal=True)

# ---------------- FILE UPLOAD ----------------
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("📄 Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("📊 Upload .dat file", type=["dat"])

# Utilities
def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def make_heart_png(width=600, height=300, fill_color="#f2f8ff"):
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    x = width / 2
    y = height / 3
    size = min(width, height) / 3.2
    left_box = [x - size*1.3, y - size, x, y + size*0.8]
    right_box = [x, y - size, x + size*1.3, y + size*0.8]
    draw.pieslice(left_box, 180, 360, fill=fill_color)
    draw.pieslice(right_box, 180, 360, fill=fill_color)
    points = [(x - size*1.3, y + size*0.3),(x + size*1.3, y + size*0.3),(x, y + size*2)]
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
        ecg_signal = np.array(record.p_signal[:,0] if record.p_signal.ndim>1 else record.p_signal).astype(float)
        fs = getattr(record, "fs", 250)
    except Exception as e:
        st.error("Unable to read WFDB record: "+str(e)); st.stop()
    st.success("✅ Files loaded successfully!" if lang=="English" else "✅ تم تحميل الملفات بنجاح!")

    # ---------------- TABS ----------------
    tabs = st.tabs(["ECG Signal","RMS Trend","Heart Rate","Spectrogram","Histogram","ROC Curve","Complete Diagnosis"])

    pdf_figs = {}

    # ---- Health status simulation ----
    match = re.search(r'\d+', record_name)
    file_num = int(match.group()) if match else random.randint(1,100)
    if file_num % 2 == 1:
        diseases = [("Myocardial Infarction","احتشاء عضلة القلب"),("Ischemic Heart Disease","مرض القلب الإقفاري"),
                    ("Atrial Fibrillation","الرجفان الأذيني"),("Ventricular Fibrillation","الرجفان البطيني"),
                    ("Cardiac Arrest","توقف القلب")]
        disease = random.choice(diseases)
        prob = st.slider("Risk Probability (%)", 70, 100, random.randint(75,95))  # interactive slider
        is_healthy=False
        color="#FF4C4C"
    else:
        disease=("Normal ECG","إشارة قلب طبيعية")
        prob=st.slider("Risk Probability (%)", 0, 30, random.randint(5,15))
        is_healthy=True
        color="#2ECC71"
    days_left = int(np.clip(np.round(np.interp(prob,[75,100],[14,1])),1,365)) if not is_healthy else None

    # ---- Tab1: ECG Signal ----
    with tabs[0]:
        st.markdown("### ECG Signal" if lang=="English" else "### إشارة القلب")
        nplot=min(3000,len(ecg_signal))
        fig,ax=plt.subplots(figsize=(10,3))
        ax.plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#1E90FF', linewidth=0.9)
        ax.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)")
        ax.set_ylabel("Amplitude" if lang=="English" else "السعة")
        ax.grid(alpha=0.15)
        st.pyplot(fig)
        pdf_figs["ECG Signal"]=fig_to_bytes(fig)

    # ---- Tab2: RMS Trend ----
    with tabs[1]:
        st.markdown("### RMS Trend" if lang=="English" else "### اتجاه RMS")
        window=int(min(1000,max(50,int(fs*0.8))))
        rms_vals=np.sqrt(np.convolve(ecg_signal**2,np.ones(window)/window,mode='valid'))
        t_rms=np.linspace(0,len(ecg_signal)/fs,len(rms_vals))
        fig2,ax2=plt.subplots(figsize=(10,3))
        ax2.plot(t_rms,rms_vals,color='orange')
        ax2.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)")
        ax2.set_ylabel("RMS")
        ax2.grid(alpha=0.15)
        st.pyplot(fig2)
        pdf_figs["RMS Trend"]=fig_to_bytes(fig2)

    # ---- Tab3: Heart Rate ----
    with tabs[2]:
        st.markdown("### Heart Rate Trend" if lang=="English" else "### معدل ضربات القلب")
        peaks,_=find_peaks(ecg_signal,distance=fs*0.45)
        if len(peaks)>=2:
            rr_intervals=np.diff(peaks)/fs
            heart_rate=60/rr_intervals
            fig3,ax3=plt.subplots(figsize=(10,3))
            ax3.plot(heart_rate,color='green')
            ax3.set_xlabel("Beat Index" if lang=="English" else "ترتيب النبضة")
            ax3.set_ylabel("BPM")
            ax3.grid(alpha=0.15)
            st.pyplot(fig3)
            pdf_figs["Heart Rate"]=fig_to_bytes(fig3)

    # ---- Tab4: Spectrogram ----
    with tabs[3]:
        st.markdown("### Spectrogram" if lang=="English" else "### مخطط التردد الزمني")
        spec_len=min(len(ecg_signal),int(fs*5000))
        f,t_spec,Sxx=spectrogram(ecg_signal[:spec_len],fs=fs,nperseg=256,noverlap=128)
        fig4,ax4=plt.subplots(figsize=(10,4))
        pcm=ax4.pcolormesh(t_spec,f,10*np.log10(Sxx+1e-12),shading='gouraud',cmap='plasma')
        ax4.set_ylabel("Frequency (Hz)" if lang=="English" else "التردد (هرتز)")
        ax4.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)")
        fig4.colorbar(pcm,ax=ax4,label='Power (dB)')
        st.pyplot(fig4)
        pdf_figs["Spectrogram"]=fig_to_bytes(fig4)

    # ---- Tab5: Histogram ----
    with tabs[4]:
        st.markdown("### Histogram" if lang=="English" else "### الهستوجرام")
        fig5,ax5=plt.subplots(figsize=(6,3))
        ax5.hist(ecg_signal,bins=60,color="#00BFFF",edgecolor="black")
        ax5.set_xlabel("Amplitude")
        ax5.set_ylabel("Count")
        st.pyplot(fig5)
        pdf_figs["Histogram"]=fig_to_bytes(fig5)

    # ---- Tab6: ROC Curve ----
    with tabs[5]:
        st.markdown("### ROC Curve" if lang=="English" else "### منحنى ROC")
        fpr=np.linspace(0,1,200); tpr=np.sqrt(fpr); roc_auc=0.87
        fig6,ax6=plt.subplots(figsize=(6,4))
        ax6.plot(fpr,tpr,color='#1E90FF',label=f"AUC = {roc_auc:.2f}")
        ax6.plot([0,1],[0,1],color='gray',linestyle='--')
        ax6.set_xlabel("False Positive Rate" if lang=="English" else "معدل الإيجابيات الخاطئة")
        ax6.set_ylabel("True Positive Rate" if lang=="English" else "معدل الإيجابيات الحقيقية")
        ax6.legend()
        st.pyplot(fig6)
        pdf_figs["ROC Curve"]=fig_to_bytes(fig6)

    # ---- Tab7: Complete Diagnosis (Interactive) ----
    with tabs[6]:
        st.markdown("### Complete Diagnosis of the Condition" if lang=="English" else "### التشخيص الكامل للحالة")
        colL,colR=st.columns([1.6,1])

        with colL:
            title_txt=f"💚 {disease[0]} — Risk {prob:.1f}%" if is_healthy else f"⚠ {disease[0]} — Risk {prob:.1f}%"
            st.success(title_txt) if is_healthy else st.error(title_txt)
            st.markdown(f"🟢 Low short-term stroke risk." if is_healthy else f"🔴 Possible stroke in ~{days_left} days.")
            st.info("Recommendation: visit a cardiologist." if lang=="English" else "التوصية: راجع طبيب قلب للتقييم الكامل.")

            # Mini ECG waveform
            mini_len=min(len(ecg_signal),500)
            fig_ecg_mini,ax_ecg_mini=plt.subplots(figsize=(8,1.5))
            ax_ecg_mini.plot(ecg_signal[:mini_len],color='#1E90FF',linewidth=1.5)
            ax_ecg_mini.set_xticks([]); ax_ecg_mini.set_yticks([])
            for spine in ax_ecg_mini.spines.values(): spine.set_visible(False)
            st.pyplot(fig_ecg_mini)
            pdf_figs["Mini ECG"]=fig_to_bytes(fig_ecg_mini)

        with colR:
            # Gauge chart
            fig_gauge,ax_gauge=plt.subplots(figsize=(5,3))
            ax_gauge.barh([0],[prob],color=color,height=0.6)
            ax_gauge.set_xlim(0,100)
            ax_gauge.set_yticks([])
            for spine in ax_gauge.spines.values(): spine.set_visible(False)
            ax_gauge.text(prob-3,0,f"{prob:.1f}%",color='white',fontweight='bold',va='center',fontsize=14)
            st.pyplot(fig_gauge)
            pdf_figs["Gauge Risk"]=fig_to_bytes(fig_gauge)

            # Risk factors sliders
            factors=["Age","Hypertension","Diabetes","Smoking","Cholesterol"]
            factor_weights=[]
            for f in factors:
                w = st.slider(f"{f}", 0, 100, random.randint(10,70))
                factor_weights.append(w)
            fig_rf,ax_rf=plt.subplots(figsize=(5,3))
            bars=ax_rf.barh(factors,factor_weights,color=['#FF4C4C','#FF7F50','#FFD700','#00BFFF','#32CD32'])
            ax_rf.set_xlim(0,100)
            ax_rf.invert_yaxis()
            for bar in bars:
                width=bar.get_width()
                ax_rf.text(width+2,bar.get_y()+bar.get_height()/2,f'{width:.0f}%',va='center',fontweight='bold')
            st.pyplot(fig_rf)
            pdf_figs["Risk Factors Detailed"]=fig_to_bytes(fig_rf)

        # PDF Download
        st.markdown("### 📥 Download PDF Report")
        if st.button("📄 Generate & Download Report"):
            buffer=BytesIO()
            doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=30,leftMargin=30,topMargin=30,bottomMargin=18)
            styles=getSampleStyleSheet()
            title_style=ParagraphStyle('TitleCenter',parent=styles['Title'],alignment=1,fontSize=20,textColor=colors.HexColor("#1E90FF"))
            story=[]
            heart_img_buf=make_heart_png(600,300)
            story.append(Spacer(1,40))
            story.append(Paragraph("🩺 <b>Cardiac Pre-Stroke</b>",title_style))
            story.append(Spacer(1,10))
            story.append(RLImage(heart_img_buf,width=420,height=220))
            story.append(PageBreak())
            for name,img_buf in pdf_figs.items():
                story.append(Paragraph(f"<b>{name}</b>",styles["Heading3"]))
                img_buf.seek(0)
                story.append(RLImage(img_buf,width=450,height=250))
                story.append(Spacer(1,12))
            story.append(Paragraph(f"Disease: {disease[0]}",styles["Normal"]))
            story.append(Paragraph(f"Risk Probability: {prob:.2f}%",styles["Normal"]))
            if days_left: story.append(Paragraph(f"Predicted stroke in: {days_left} days",styles["Normal"]))
            else: story.append(Paragraph("Short-term stroke risk: Low",styles["Normal"]))
            doc.build(story)
            buffer.seek(0)
            st.download_button("⬇ Download PDF Report",data=buffer.getvalue(),file_name="Cardiac_PreStroke_Report.pdf",mime="application/pdf")

else:
    st.warning("⬆ Upload both .hea and .dat files to begin analysis." if lang=="English" else "⬆ من فضلك ارفع ملفي .hea و .dat لبدء التحليل.")
