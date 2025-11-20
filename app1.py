# app.py - Cardiac Pre-Stroke (Full Advanced Diagnostic System)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import random
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------- تهيئة الصفحة والجلسة -------------------
st.set_page_config(page_title="نظام تشخيص القلب المتقدم", page_icon="❤️", layout="wide")

# تهيئة متغيرات الجلسة لإدارة الصفحات والمستخدم
if "page" not in st.session_state: st.session_state["page"] = "login"
if "user_name" not in st.session_state: st.session_state["user_name"] = ""

# ------------------- تصميم الواجهة (CSS) -------------------
st.markdown("""
<style>
    body { background-color: #f0f2f6; color: #1e293b; }
    .header-card { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;}
    .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .h1 { color: white; margin: 0; font-weight: 700; font-size: 32px; }
    .lead { color: #cbd5e1; margin-top: 8px; font-size: 16px; }
    .footer { color: #64748b; font-size: 12px; text-align: center; padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ------------------- دوال مساعدة ونموذج التعلم الآلي -------------------
@st.cache_resource
def get_model_and_scaler():
    """تدريب وتحميل النموذج والمحول وتخزينهما في الذاكرة المخبئية."""
    np.random.seed(42)
    X = np.random.rand(200, 8)
    y = np.random.randint(0, 4, 200)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(multi_class='multinomial').fit(scaler.transform(X), y)
    return model, scaler

def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def extract_features(signal, fs):
    """استخلاص ميزات إحصائية وطبية من الإشارة"""
    peaks, _ = find_peaks(signal, distance=fs * 0.4, height=np.mean(signal))
    if len(peaks) < 5: return None
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60.0 / rr_intervals
    features = {
        "المتوسط (Mean)": np.mean(signal),
        "الانحراف المعياري (Std Dev)": np.std(signal),
        "القيمة الفعالة (RMS)": np.sqrt(np.mean(signal**2)),
        "المدى (Range)": np.ptp(signal),
        "الالتواء (Skewness)": skew(signal),
        "التفرطح (Kurtosis)": kurtosis(signal),
        "متوسط معدل القلب (Mean HR)": np.mean(heart_rate),
        "تقلب معدل القلب (SDNN)": np.std(rr_intervals) * 1000,
    }
    return features

def get_diagnosis(features_vector, model, scaler):
    """محاكاة الحصول على تشخيص متعدد الفئات وتنبؤ بالمخاطر"""
    scaled_features = scaler.transform(features_vector.reshape(1, -1))
    class_id = model.predict(scaled_features)[0]
    classifications = {
        0: ("نبض طبيعي", "الإشارة تظهر نمطًا منتظمًا وصحيًا.", "منخفض"),
        1: ("اضطرابات بسيطة", "تم رصد عدم انتظام طفيف في الإيقاع.", "متوسط"),
        2: ("ضعف في الإشارات", "السعة الكهربائية للإشارة منخفضة.", "متوسط"),
        3: ("حالة خطرة محتملة", "تم رصد علامات قد تشير إلى خطر مرتفع.", "مرتفع")
    }
    classification, class_desc, risk_level = classifications[class_id]
    anomalies = []
    if features_vector[7] > 150: anomalies.append("تباين كبير في معدل ضربات القلب (قد يشير لرجفان).")
    if abs(features_vector[4]) > 0.5: anomalies.append("توزيع الإشارة غير متماثل (شكل موجة غير طبيعي).")
    if class_id == 3: anomalies.append("تم رصد علامات تتوافق مع خطر الجلطة (مثل ST Elevation).")
    if not anomalies and class_id != 0: anomalies.append("عدم انتظام عام في الإيقاع.")
    return classification, class_desc, risk_level, anomalies

def build_pdf_report(user_name, features, diagnosis, anomalies, risk, figs):
    """إنشاء تقرير PDF شامل"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>تقرير تشخيص القلب المتقدم</b>", styles['h1']))
    story.append(Paragraph(f"<i>صادر بواسطة: {user_name}</i>", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>🚨 تقييم المخاطر العام</b>", styles['h2']))
    story.append(Paragraph(f"مستوى الخطر: {risk}", styles['Normal']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>🩺 تصنيف إشارة القلب</b>", styles['h2']))
    story.append(Paragraph(f"التصنيف: {diagnosis[0]}", styles['Normal']))
    story.append(Paragraph(f"توضيح: {diagnosis[1]}", styles['Normal']))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>🔍 الأنماط غير الطبيعية المكتشفة</b>", styles['h2']))
    if anomalies:
        for anomaly in anomalies:
            story.append(Paragraph(f"- {anomaly}", styles['Normal']))
    else:
        story.append(Paragraph("لم يتم رصد أنماط غير طبيعية واضحة.", styles['Normal']))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>🔬 الميزات الطبية المستخرجة</b>", styles['h2']))
    features_data = [['الميزة', 'القيمة']] + [[k, f"{v:.2f}"] for k, v in features.items()]
    table = Table(features_data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("<b>📊 الرسوم البيانية للتحليل</b>", styles['h2']))
    for title, fig_buffer in figs.items():
        story.append(Paragraph(f"<b>{title}</b>", styles['h3']))
        img = RLImage(fig_buffer, width=450, height=225)
        story.append(img)
        story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------- واجهة التطبيق الرئيسية -------------------
# تحميل النموذج والمحول
model, scaler = get_model_and_scaler()

# --- صفحة تسجيل الدخول ---
if st.session_state.page == "login":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>تسجيل الدخول</h2>", unsafe_allow_html=True)
    with st.form('login_form'):
        name = st.text_input('الاسم الكامل')
        submitted = st.form_submit_button('دخول')
        if submitted and name.strip() != "":
            st.session_state.user_name = name
            st.session_state.page = "welcome"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- صفحة الترحيب ---
elif st.session_state.page == "welcome":
    st.markdown(f"## أهلاً بك, {st.session_state.user_name}!")
    st.info("أنت الآن في لوحة التحكم. يمكنك البدء بتحليل جديد.")
    if st.button("🚀 ابدأ تحليل جديد"):
        st.session_state.page = "analysis"
        st.rerun()
    if st.button("🔒 تسجيل خروج"):
        st.session_state.page = "login"
        st.session_state.user_name = ""
        st.rerun()

# --- صفحة التحليل ---
elif st.session_state.page == "analysis":
    st.markdown(f'<div class="header-card"><h1 class="h1">❤️ نظام تشخيص القلب المتقدم</h1><p class="lead">المستخدم: {st.session_state.user_name}</p></div>', unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📂 رفع ملفات ECG (.hea & .dat)")
    hea_file = st.file_uploader('📄 ارفع ملف .hea', type=['hea'])
    dat_file = st.file_uploader('📊 ارفع ملف .dat', type=['dat'])
    st.markdown("</div>", unsafe_allow_html=True)

    if hea_file and dat_file:
        try:
            record_name = hea_file.name.replace('.hea','')
            with open(hea_file.name,'wb') as f: f.write(hea_file.getvalue())
            with open(dat_file.name,'wb') as f: f.write(dat_file.getvalue())
            record = wfdb.rdrecord(record_name)
            ecg_signal = record.p_signal[:,0] if record.p_signal.ndim > 1 else record.p_signal
            fs = record.fs
            st.success('✅ تم تحميل الملفات بنجاح! جاري التحليل...')
        except Exception as e:
            st.error(f'❌ تعذر قراءة السجل: {e}')
            st.stop()

        features = extract_features(ecg_signal, fs)
        if features is None:
            st.warning("⚠️ لا يمكن تحليل الإشارة. قد تكون قصيرة جدًا أو غير واضحة.")
            st.stop()
        
        features_vector = np.array(list(features.values()))
        classification, class_desc, risk_level, anomalies = get_diagnosis(features_vector, model, scaler)

        st.markdown("## 📋 ملخص التشخيص الآلي")
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚨 تقييم المخاطر العام")
        risk_color = {"منخفض": "green", "متوسط": "orange", "مرتفع": "red"}
        st.markdown(f"<h4>مستوى الخطر: <span style='color:{risk_color[risk_level]};'>{risk_level}</span></h4>", unsafe_allow_html=True)
        st.markdown(class_desc)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🩺 تصنيف إشارة القلب")
            st.markdown(f"**التصنيف:** `{classification}`")
            st.info(f"**توضيح:** {class_desc}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔬 الميزات الطبية المستخرجة")
            st.table(features)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔍 الأنماط غير الطبيعية المكتشفة")
            if anomalies:
                for anomaly in anomalies:
                    st.warning(f"⚠️ {anomaly}")
            else:
                st.success("✅ لم يتم رصد أنماط غير طبيعية واضحة.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📊 الرسوم البيانية للتحليل")
        pdf_figs = {}

        # رسم الإشارة
        fig, ax = plt.subplots(figsize=(10, 3))
        nplot = min(3000, len(ecg_signal))
        ax.plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#3b82f6', linewidth=1)
        ax.set_title("إشارة القلب (ECG Signal)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        pdf_figs["إشارة القلب"] = fig_to_bytes(fig)

        # رسم معدل القلب
        peaks, _ = find_peaks(ecg_signal, distance=fs * 0.4, height=np.mean(ecg_signal))
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks) / fs
            heart_rate = 60.0 / rr_intervals
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(heart_rate, 'o-', color='#16a34a', markersize=3, label=f"متوسط: {np.mean(heart_rate):.1f} BPM")
            ax2.set_title("تتبع معدل ضربات القلب (Heart Rate)")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig2)
            pdf_figs["معدل ضربات القلب"] = fig_to_bytes(fig2)

        # زر إنشاء وتنزيل التقرير
        st.markdown("---")
        st.markdown("## 📥 إنشاء تقرير PDF")
        if st.button("📄 إنشاء وتنزيل التقرير"):
            with st.spinner("جاري إنشاء التقرير..."):
                pdf_buffer = build_pdf_report(st.session_state.user_name, features, (classification, class_desc), anomalies, risk_level, pdf_figs)
                st.download_button(
                    label="⬇️ تحميل التقرير الآن",
                    data=pdf_buffer,
                    file_name=f"Cardiac_Report_{record_name}.pdf",
                    mime="application/pdf"
                )

    if st.button("العودة إلى الصفحة الرئيسية"):
        st.session_state.page = "welcome"
        st.rerun()

st.markdown('<div class="footer">نظام Cardiac Pre-Stroke | للاستخدام التعليمي والتجريبي فقط</div>', unsafe_allow_html=True)
