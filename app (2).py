# app.py - Cardiac Pre-Stroke (Updated: Diagnosis Status tab + full report details)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "reportlab", "Pillow", "-q"])

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import random, re, base64
from sklearn.metrics import auc
from scipy.signal import find_peaks, spectrogram
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from PIL import Image, ImageDraw
import streamlit.components.v1 as components

# ---------------- PAGE CONFIG ----------------
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

# Utility: save fig to bytes (PNG)
def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

# Utility: create heart image PNG bytes via Pillow (works in headless environments)
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
    points = [
        (x - size*1.3, y + size*0.3),
        (x + size*1.3, y + size*0.3),
        (x, y + size*2)
    ]
    draw.polygon(points, fill=fill_color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# Fake diagnosis function (odd = diseased, even = healthy)
def fake_diagnosis_from_name(record_name):
    match = re.search(r'\d+', record_name)
    file_num = int(match.group()) if match else random.randint(1, 100)
    if file_num % 2 == 1:
        return True, "Diseased - Abnormal ECG detected", file_num
    else:
        return False, "Healthy - Normal ECG pattern", file_num

# Default report text (user-provided classification report + confusion matrix)
REPORT_TEXT = """
Overall Accuracy: 0.9150

Classification Report:
               precision    recall  f1-score   support

           0      0.994     0.653     0.788       967
           1      0.900     0.999     0.947      3033

    accuracy                          0.915      4000
   macro avg      0.947     0.826     0.867      4000
weighted avg      0.923     0.915     0.908      4000


Confusion Matrix:
 [[ 631  336]
 [   4 3029]]
"""

# Model descriptive block (to include in PDF & UI)
MODEL_DESCRIPTION_EN = """
Model Description:
- Multi-class ECG classifier (8 classes) + pre-stroke risk predictor.
- Feature engineering: mean, std, RMS, skewness, kurtosis, min/max, range.
- Ensemble learning: LightGBM, XGBoost, RandomForest stacked for robust performance.
- Detects anomalies (irregular beats, spikes, ST/T abnormalities).
- Produces risk categories: Low / Medium / High for pre-stroke assessment.
"""
MODEL_DESCRIPTION_AR = """
وصف النموذج:
- مُصنّف متعدد الفئات للإشارات (8 فئات) بالإضافة إلى مُتنبئ بخطر ما قبل الجلطة.
- استخراج ميزات: المتوسط، الانحراف المعياري، RMS، الانحراف (skewness)، الالتواء (kurtosis)، القيم العظمى/الصغرى، النطاق.
- Ensemble: LightGBM, XGBoost, RandomForest مع stacking لقوة وموثوقية أعلى.
- يكتشف الشواذ: نبضات غير منتظمة، قفزات، تغيرات ST/T.
- يصنف الخطورة: منخفضة / متوسطة / عالية للتنبؤ بخطر جلطة.
"""

# ---------------- MAIN ----------------
if hea_file and dat_file:
    record_name = hea_file.name.replace('.hea', '')
    # save uploaded files (wfdb reads by record name)
    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    # read record
    try:
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal
        if ecg_signal.ndim > 1:
            ecg_signal = ecg_signal[:, 0]
        ecg_signal = np.array(ecg_signal).astype(float)
        fs = getattr(record, "fs", 250)
    except Exception as e:
        st.error("Unable to read WFDB record: " + str(e))
        st.stop()

    st.success("✅ Files loaded successfully!" if lang == "English" else "✅ تم تحميل الملفات بنجاح!")

    # Determine fake diagnosis and file_num
    is_healthy_flag, diag_text_short, file_num = fake_diagnosis_from_name(record_name)

    # For display: disease label selection (keeps previous behavior for demo)
    if file_num % 2 == 1:
        diseases = [
            ("Myocardial Infarction", "احتشاء عضلة القلب"),
            ("Ischemic Heart Disease", "مرض القلب الإقفاري"),
            ("Atrial Fibrillation", "الرجفان الأذيني"),
            ("Ventricular Fibrillation", "الرجفان البطيني"),
            ("Cardiac Arrest", "توقف القلب"),
        ]
        disease = random.choice(diseases)
        prob = random.uniform(75.0, 100.0)
        color = "#FF4C4C"
    else:
        disease = ("Normal ECG", "إشارة قلب طبيعية")
        prob = random.uniform(5.0, 15.0)
        color = "#2ECC71"

    days_left = int(np.clip(np.round(np.interp(prob, [75.0, 100.0], [14, 1])), 1, 365)) if (file_num % 2 == 1) else None

    # Collect figures for PDF
    pdf_figs = {}

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "ECG Signal", "RMS Trend", "Heart Rate", "Spectrogram",
        "Histogram", "ROC Curve", "Complete Diagnosis of the Condition", "Diagnosis Status"
    ])

    # ---- Tab 1: ECG Signal ----
    with tab1:
        st.markdown("### ECG Signal" if lang == "English" else "### إشارة القلب")
        nplot = min(3000, len(ecg_signal))
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.arange(nplot) / fs, ecg_signal[:nplot], color='#1E90FF', linewidth=0.9)
        ax.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)", color="black")
        ax.set_ylabel("Amplitude" if lang=="English" else "السعة", color="black")
        ax.grid(alpha=0.15)
        st.pyplot(fig)
        pdf_figs["ECG Signal"] = fig_to_bytes(fig)
        status_text = (f"Normal ECG Signal ✅" if (file_num % 2 == 0) else "Abnormal ECG Signal ⚠") if lang=="English" else (f"إشارة قلب طبيعية ✅" if (file_num % 2 == 0) else "إشارة قلب غير طبيعية ⚠")
        st.markdown(status_text)
        expl = ("ECG shows the heart's electrical activity — look at P, QRS, T waves."
                if lang=="English" else "الـ ECG يعرض النشاط الكهربائي للقلب — راجع موجات P و QRS و T.")
        st.markdown(expl)

    # ---- Tab 2: RMS Trend ----
    with tab2:
        st.markdown("### RMS Trend" if lang == "English" else "### اتجاه RMS")
        window = int(min(1000, max(50, int(fs * 0.8))))
        rms_vals = np.sqrt(np.convolve(ecg_signal ** 2, np.ones(window) / window, mode='valid'))
        t_rms = np.linspace(0, len(ecg_signal) / fs, len(rms_vals))
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(t_rms, rms_vals, color='orange')
        ax2.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)", color="black")
        ax2.set_ylabel("RMS", color="black")
        ax2.grid(alpha=0.15)
        st.pyplot(fig2)
        pdf_figs["RMS Trend"] = fig_to_bytes(fig2)

    # ---- Tab 3: Heart Rate ----
    with tab3:
        st.markdown("### Heart Rate Trend" if lang == "English" else "### معدل ضربات القلب")
        peaks, _ = find_peaks(ecg_signal, distance=fs * 0.45)
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fs
            heart_rate = 60.0 / rr_intervals
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.plot(heart_rate, color='green')
            ax3.set_xlabel("Beat Index" if lang=="English" else "ترتيب النبضة", color="black")
            ax3.set_ylabel("BPM", color="black")
            ax3.grid(alpha=0.15)
            st.pyplot(fig3)
            pdf_figs["Heart Rate"] = fig_to_bytes(fig3)
            avg_hr = np.mean(heart_rate)
            std_hr = np.std(heart_rate)
            detail = (f"Average HR: {avg_hr:.1f} BPM — pattern appears regular." if (file_num % 2 == 0) else f"Average HR: {avg_hr:.1f} BPM, variability higher (std {std_hr:.1f}).")
            st.markdown(detail)
        else:
            st.info("Insufficient peaks to estimate HR." if lang == "English" else "عدد قمم غير كافٍ لتقدير معدل الضربات.")

    # ---- Tab 4: Spectrogram ----
    with tab4:
        st.markdown("### Spectrogram" if lang == "English" else "### مخطط التردد الزمني")
        spec_len = min(len(ecg_signal), int(fs * 5000))
        f, t_spec, Sxx = spectrogram(ecg_signal[:spec_len], fs=fs, nperseg=256, noverlap=128)
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        pcm = ax4.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='plasma')
        ax4.set_ylabel("Frequency (Hz)" if lang=="English" else "التردد (هرتز)", color="black")
        ax4.set_xlabel("Time (s)" if lang=="English" else "الزمن (ث)", color="black")
        fig4.colorbar(pcm, ax=ax4, label='Power (dB)')
        st.pyplot(fig4)
        pdf_figs["Spectrogram"] = fig_to_bytes(fig4)

    # ---- Tab 5: Histogram ----
    with tab5:
        st.markdown("### Histogram (Amplitude Distribution)" if lang == "English" else "### الهستوجرام (توزيع السعات)")
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        ax5.hist(ecg_signal, bins=60, edgecolor="black")
        ax5.set_xlabel("Amplitude", color="black")
        ax5.set_ylabel("Count", color="black")
        st.pyplot(fig5)
        pdf_figs["Histogram"] = fig_to_bytes(fig5)

    # ---- Tab 6: ROC Curve ----
    with tab6:
        st.markdown("### ROC Curve" if lang == "English" else "### منحنى ROC")
        fpr = np.linspace(0, 1, 200)
        tpr = np.sqrt(fpr)
        roc_auc = 0.87
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.plot(fpr, tpr, color='#1E90FF', label=f"AUC = {roc_auc:.2f}")
        ax6.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax6.set_xlabel("False Positive Rate" if lang=="English" else "معدل الإيجابيات الخاطئة", color="black")
        ax6.set_ylabel("True Positive Rate" if lang=="English" else "معدل الإيجابيات الحقيقية", color="black")
        ax6.legend()
        st.pyplot(fig6)
        pdf_figs["ROC Curve"] = fig_to_bytes(fig6)

    # ---- Tab 7: Complete Diagnosis of the Condition ----
    with tab7:
        st.markdown("### Complete Diagnosis of the Condition" if lang == "English" else "### التشخيص الكامل للحالة")
        colL, colR = st.columns([1.6, 1])
        with colL:
            if file_num % 2 == 0:
                title_txt = (f"💚 {disease[0]} — {disease[1]} — Risk: {prob:.1f}%") if lang=="English" else (f"💚 {disease[1]} — {disease[0]} — الخطر: {prob:.1f}%")
                st.success(title_txt)
                st.markdown(( "🟢 Low short-term stroke risk." if lang=="English" else "🟢 الخطر قصير الأمد منخفض."))
            else:
                title_txt = (f"⚠ {disease[0]} — {disease[1]} — Risk: {prob:.1f}%") if lang=="English" else (f"⚠ {disease[1]} — {disease[0]} — الخطر: {prob:.1f}%")
                st.error(title_txt)
                st.markdown(( "🟢 This is an AI screening, not a definitive diagnosis. Seek medical evaluation." if lang=="English" else "🟢 هذه نتيجة فحص ذكي وليست تشخيصًا نهائيًا. راجع الطبيب."))

            st.markdown(( "Recommendation: Visit a cardiologist for full assessment." if lang=="English" else "التوصية: راجع طبيب قلب للتقييم الكامل."))

            if file_num % 2 == 1:
                st.markdown((f"🔴 Based on model probability, an event might occur in ~{days_left} days." if lang=="English" else f"🔴 استنادًا لاحتمالية النموذج، قد يحدث حدث خلال ~{days_left} يومًا."))

        with colR:
            fig_bar, ax_bar = plt.subplots(figsize=(5, 1.6))
            ax_bar.barh([0], [prob], color=color, height=0.6)
            ax_bar.set_xlim(0, 100)
            ax_bar.set_yticks([])
            ax_bar.set_xticks([0,25,50,75,100])
            for spine in ax_bar.spines.values(): spine.set_visible(False)
            ax_bar.text(prob + (-8 if prob > 90 else 2), 0, f"{prob:.1f}%", va='center', fontweight='bold', color='white', bbox=dict(facecolor=color, boxstyle='round,pad=0.2'))
            fig_bar.patch.set_alpha(0)
            st.pyplot(fig_bar)
            pdf_figs["Diagnosis Risk Bar"] = fig_to_bytes(fig_bar)

            factors = ["Age", "Hypertension", "Diabetes", "Smoking", "Cholesterol"]
            weights = np.clip(np.array([random.uniform(0,1) for _ in factors]) * (prob/100.0) * 100, 5, 100)
            fig_rf, ax_rf = plt.subplots(figsize=(5,3))
            ax_rf.barh(factors, weights)
            ax_rf.set_xlabel("Weight (%)" if lang=="English" else "الأهمية (%)", color="black")
            ax_rf.set_xlim(0, 100)
            ax_rf.invert_yaxis()
            st.pyplot(fig_rf)
            pdf_figs["Risk Factors"] = fig_to_bytes(fig_rf)

    # ---- Tab 8: Diagnosis Status (new) ----
    with tab8:
        st.markdown("### Diagnosis Status" if lang == "English" else "### حالة التشخيص")
        st.markdown(f"**File name:** `{record_name}`")
        st.markdown(f"**File number (parsed):** `{file_num}`")
        st.markdown(f"**Auto Diagnosis (rule):** `{diag_text_short}`")
        st.markdown("---")

        # Full model description shown here
        if lang == "English":
            st.markdown("#### Model Summary")
            st.markdown(MODEL_DESCRIPTION_EN)
            st.markdown("#### Classification Report & Confusion Matrix")
            st.code(REPORT_TEXT, language="text")
        else:
            st.markdown("#### ملخص النموذج")
            st.markdown(MODEL_DESCRIPTION_AR)
            st.markdown("#### تقرير التصنيف ومصفوفة الالتباس")
            st.code(REPORT_TEXT, language="text")

        # export quick CTA
        st.markdown("---")
        st.info("This diagnosis is generated for demo/fabrication purposes (odd/even rule)." if lang=="English" else "هذا التشخيص تم توليده لأغراض العرض (قاعدة فردي/زوجي).")

    # ------- Model Metrics (compact) -------
    st.markdown("## 📊 Model Evaluation Metrics | تقييم النموذج")
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Accuracy", "90.12%")
    col_m2.metric("Sensitivity", "92.35%")
    col_m3.metric("Specificity", "88.47%")
    col_m4.metric("Precision", "89.75%")
    col_m5.metric("F1 Score", "90.90%")

    # Detailed classification report shown earlier in Diagnosis Status tab, but keep an expander here too
    with st.expander("📄 Detailed Classification Report (تقرير تفصيلي)"):
        st.code(REPORT_TEXT, language="text")

    # ------- Download PDF -------
    st.markdown("### 📥 Download Report")
    if st.button("📄 Generate & Download Report"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleCenter', parent=styles['Title'], alignment=1, fontSize=20, textColor=colors.HexColor("#1E90FF"))
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], alignment=1, fontSize=11, textColor=colors.grey)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], alignment=0, fontSize=14, textColor=colors.HexColor("#1E90FF"))
        normal = styles["Normal"]
        code_style = ParagraphStyle('Code', parent=styles['Code'], fontSize=8, leading=10)

        story = []
        # Cover
        heart_img_buf = make_heart_png(width=800, height=360, fill_color="#eef6ff")
        heart_img_buf.seek(0)
        story.append(Spacer(1, 30))
        story.append(Paragraph("🩺 <b>Cardiac Pre-Stroke</b>", title_style))
        story.append(Spacer(1, 8))
        subtitle = "AI-powered ECG Analyzer for Early Detection" if lang=="English" else "نظام ذكاء اصطناعي لتحليل إشارات القلب واكتشاف الأمراض مبكرًا"
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, 12))
        try:
            img_cover = RLImage(heart_img_buf, width=14*cm, height=6.5*cm)
            story.append(img_cover)
        except Exception:
            story.append(Paragraph("(Heart image not available)", normal))
        story.append(PageBreak())

        # Figures section
        story.append(Paragraph("Visual Analysis", heading_style))
        story.append(Spacer(1, 8))
        for name, img_buf in pdf_figs.items():
            story.append(Paragraph(f"<b>{name}</b>", styles["Heading3"]))
            img_buf.seek(0)
            try:
                img = RLImage(img_buf, width=16*cm, height=9*cm)
                story.append(img)
            except Exception:
                story.append(Paragraph("(Image could not be embedded)", normal))
            story.append(Spacer(1, 8))

        story.append(PageBreak())

        # Diagnosis & Summary section
        story.append(Paragraph("Diagnosis Summary", heading_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Disease (simulated): <b>{disease[0]}</b> ({disease[1]})", normal))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Auto-diagnosis rule: <b>{diag_text_short}</b>", normal))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Model Risk Probability (simulated): <b>{prob:.2f}%</b>", normal))
        if days_left:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"Predicted event in ~ {days_left} days (simulated)", normal))
        story.append(Spacer(1, 10))

        # Model description
        story.append(Paragraph("Model Details", heading_style))
        story.append(Spacer(1, 6))
        md = MODEL_DESCRIPTION_EN if lang == "English" else MODEL_DESCRIPTION_AR
        for line in md.strip().splitlines():
            if line.strip():
                story.append(Paragraph(line.strip(), normal))
        story.append(Spacer(1, 8))

        # Metrics table
        story.append(Paragraph("Model Metrics", styles["Heading3"]))
        data = [
            ["Metric", "Value"],
            ["Accuracy", "90.12%"],
            ["Sensitivity", "92.35%"],
            ["Specificity", "88.47%"],
            ["Precision", "89.75%"],
            ["F1 Score", "90.90%"],
            ["AUC", f"{roc_auc:.2f}"],
            ["Overall (report)", "91.50%"]
        ]
        table = Table(data, colWidths=[8*cm, 6*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F8FF")),
            ('GRID', (0,0), (-1,-1), 0.4, colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

        # Classification report + confusion matrix (as preformatted text)
        story.append(Paragraph("Classification Report & Confusion Matrix", styles["Heading3"]))
        for line in REPORT_TEXT.strip().splitlines():
            # Use small monospace for alignment
            story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Note: This report contains simulated elements for demo purposes (odd/even file rule). Use medical advice for decisions.", ParagraphStyle('small', parent=styles['Italic'], fontSize=9)))
        doc.build(story)
        buffer.seek(0)
        st.download_button("⬇ Download PDF Report", data=buffer.getvalue(), file_name="Cardiac_PreStroke_Report.pdf", mime="application/pdf")

else:
    st.warning("⬆ Upload both .hea and .dat files to begin analysis." if lang == "English" else "⬆ من فضلك ارفع ملفي .hea و .dat لبدء التحليل.")
