# app.py - Cardiac Pre-Stroke (Advanced Diagnostic System)
import subprocess, sys
# تثبيت المكتبات الضرورية
subprocess.run([sys.executable, "-m", "pip", "install", "reportlab", "Pillow", "wfdb", "scikit-learn", "-q"])

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
if "page" not in st.session_state: st.session_state["page"] = "login"
if "user_name" not in st.session_state: st.session_state["user_name"] = ""
if "model" not in st.session_state: st.session_state["model"] = None
if "scaler" not in st.session_state: st.session_state["scaler"] = None

# ------------------- تصميم الواجهة (CSS) -------------------
st.markdown("""
<style>
    body { background-color: #f0f2f6; color: #1e293b; }
    .header-card { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 20px; border-radius: 12px; }
    .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .h1 { color: white; margin: 0; font-weight: 700; font-size: 32px; }
    .lead { color: #cbd5e1; margin-top: 8px; font-size: 16px; }
    .footer { color: #64748b; font-size: 12px; text-align: center; padding-top: 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { height: 44px; background-color: #f1f5f9; border-radius: 8px; gap: 8px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #3b82f6; color: white; }
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; }
    .metric-card-label { font-size: 14px; font-weight: 500; color: #475569; }
    .metric-card-value { font-size: 24px; font-weight: 700; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ------------------- دوال مساعدة ونموذج التعلم الآلي -------------------
def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def extract_features(signal, fs):
    """استخلاص ميزات إحصائية وطبية من الإشارة"""
    peaks, _ = find_peaks(signal, distance=fs * 0.4, height=np.mean(signal))
    if len(peaks) < 5: return None  # لا يمكن التحليل إذا كانت القمم قليلة جداً

    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60.0 / rr_intervals

    # الميزات الإحصائية الأساسية
    features = {
        "المتوسط (Mean)": np.mean(signal),
        "الانحراف المعياري (Std Dev)": np.std(signal),
        "القيمة الفعالة (RMS)": np.sqrt(np.mean(signal**2)),
        "المدى (Range)": np.ptp(signal),
        "الالتواء (Skewness)": skew(signal),
        "التفرطح (Kurtosis)": kurtosis(signal),
        "متوسط معدل القلب (Mean HR)": np.mean(heart_rate),
        "تقلب معدل القلب (SDNN)": np.std(rr_intervals) * 1000, # بالمللي ثانية
    }
    return features

def train_model():
    """محاكاة تدريب نموذج لتصنيف متعدد الفئات"""
    if st.session_state.get("model") is None:
        np.random.seed(42)
        # 8 features, 200 samples
        X = np.random.rand(200, 8)
        # إنشاء 4 فئات وهمية
        y = np.random.randint(0, 4, 200)
        
        scaler = StandardScaler().fit(X)
        model = LogisticRegression().fit(scaler.transform(X), y)
        
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler

train_model() # تدريب النموذج عند بدء التطبيق

def get_diagnosis(features_vector):
    """محاكاة الحصول على تشخيص متعدد الفئات وتنبؤ بالمخاطر"""
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    
    scaled_features = scaler.transform(features_vector.reshape(1, -1))
    prediction_proba = model.predict_proba(scaled_features)[0]
    class_id = np.argmax(prediction_proba)

    # --- 1) تصنيف إشارات القلب (ECG Classification) ---
    classifications = {
        0: ("نبض طبيعي", "الإشارة تظهر نمطًا منتظمًا وصحيًا.", "منخفض"),
        1: ("اضطرابات بسيطة", "تم رصد عدم انتظام طفيف في الإيقاع أو شكل الموجة.", "متوسط"),
        2: ("ضعف في الإشارات", "السعة الكهربائية للإشارة منخفضة، قد يشير إلى مشاكل في عضلة القلب.", "متوسط"),
        3: ("حالة خطرة محتملة", "تم رصد علامات قد تشير إلى خطر مرتفع مثل تغيرات ST أو موجات T غير طبيعية.", "مرتفع")
    }
    classification, class_desc, risk_level = classifications[class_id]

    # --- 2) اكتشاف الأنماط غير الطبيعية (Anomaly Detection) ---
    anomalies = []
    if features_vector[7] > 150: anomalies.append("تباين كبير في معدل ضربات القلب (قد يشير لرجفان).")
    if abs(features_vector[4]) > 0.5: anomalies.append("توزيع الإشارة غير متماثل (شكل موجة غير طبيعي).")
    if features_vector[3] > 3.0: anomalies.append("سعة الإشارة مرتفعة بشكل غير طبيعي.")
    if class_id == 3: anomalies.append("تم رصد علامات تتوافق مع خطر الجلطة (مثل ST Elevation).")
    if not anomalies and class_id != 0: anomalies.append("عدم انتظام عام في الإيقاع.")
    
    return classification, class_desc, risk_level, anomalies

# ------------------- واجهة التطبيق الرئيسية -------------------
# (تم إبقاء كود تسجيل الدخول والترحيب كما هو للاختصار)
st.session_state.page = "analysis" # الانتقال مباشرة لصفحة التحليل للعرض

if st.session_state["page"] == "analysis":
    st.markdown('<div class="header-card"><h1 class="h1">❤️ نظام تشخيص القلب المتقدم</h1><p class="lead">تحليل شامل لإشارات ECG باستخدام الذكاء الاصطناعي</p></div>', unsafe_allow_html=True)
    
    # --- رفع الملفات ---
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📂 رفع ملفات ECG (.hea & .dat)")
        hea_file = st.file_uploader('📄 ارفع ملف .hea', type=['hea'])
        dat_file = st.file_uploader('📊 ارفع ملف .dat', type=['dat'])
        st.markdown("</div>", unsafe_allow_html=True)

    if hea_file and dat_file:
        # --- قراءة وتحليل الإشارة ---
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

        # --- 3) استخراج الميزات الطبية (Feature Engineering) ---
        features = extract_features(ecg_signal, fs)
        if features is None:
            st.warning("⚠️ لا يمكن تحليل الإشارة. قد تكون قصيرة جدًا أو غير واضحة.")
            st.stop()
        
        features_vector = np.array(list(features.values()))
        
        # --- الحصول على التشخيص ---
        classification, class_desc, risk_level, anomalies = get_diagnosis(features_vector)

        # --- عرض النتائج للطبيب ---
        st.markdown("## 📋 ملخص التشخيص الآلي")
        
        # --- 4) توقع خطر الجلطة (Pre-Stroke Risk Prediction) ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚨 تقييم المخاطر العام")
        risk_color = {"منخفض": "green", "متوسط": "orange", "مرتفع": "red"}
        st.markdown(f"<h4>مستوى الخطر: <span style='color:{risk_color[risk_level]};'>{risk_level}</span></h4>", unsafe_allow_html=True)
        st.markdown(class_desc)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # --- 1) تصنيف إشارات القلب (ECG Classification) ---
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🩺 تصنيف إشارة القلب")
            st.markdown(f"**التصنيف:** `{classification}`")
            st.info(f"**توضيح:** {class_desc}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- 3) عرض الميزات المستخرجة ---
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔬 الميزات الطبية المستخرجة")
            st.table(features)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # --- 2) اكتشاف الأنماط غير الطبيعية (Anomaly Detection) ---
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔍 الأنماط غير الطبيعية المكتشفة")
            if anomalies:
                for anomaly in anomalies:
                    st.warning(f"⚠️ {anomaly}")
            else:
                st.success("✅ لم يتم رصد أنماط غير طبيعية واضحة.")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- الرسوم البيانية التوضيحية ---
        st.markdown("---")
        st.markdown("## 📊 الرسوم البيانية للتحليل")
        
        # رسم الإشارة
        fig, ax = plt.subplots(figsize=(10, 3))
        nplot = min(3000, len(ecg_signal))
        ax.plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#3b82f6', linewidth=1)
        ax.set_title("إشارة القلب (ECG Signal)")
        ax.set_xlabel('الزمن (ثانية)')
        ax.set_ylabel('السعة (mV)')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        st.info("**شرح:** هذا الرسم يوضح النشاط الكهربائي للقلب. نبحث عن انتظام الموجات (P-QRS-T) والمسافات بينها.")

        # رسم معدل القلب
        peaks, _ = find_peaks(ecg_signal, distance=fs * 0.4, height=np.mean(ecg_signal))
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks) / fs
            heart_rate = 60.0 / rr_intervals
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(heart_rate, 'o-', color='#16a34a', markersize=3, label=f"متوسط: {np.mean(heart_rate):.1f} BPM")
            ax2.set_title("تتبع معدل ضربات القلب (Heart Rate)")
            ax2.set_xlabel('ترتيب النبضة')
            ax2.set_ylabel('نبضة في الدقيقة (BPM)')
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()
            st.pyplot(fig2)
            st.info("**شرح:** يوضح هذا الرسم تغير معدل ضربات القلب. التباين الكبير (Arrhythmia) أو المعدل المرتفع/المنخفض باستمرار قد يكون علامة على وجود مشكلة.")

# ------------------- تذييل الصفحة -------------------
st.markdown('<div class="footer">نظام Cardiac Pre-Stroke | للاستخدام التعليمي والتجريبي فقط</div>', unsafe_allow_html=True)
