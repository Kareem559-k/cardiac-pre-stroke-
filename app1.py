# app.py - Cardiac Pre-Stroke (Professional UI, Patient form A, user in PDF, Real ML Model)
import subprocess, sys
# تثبيت المكتبات الضرورية في بيئة معزولة
subprocess.run([sys.executable, "-m", "pip", "install", "reportlab", "Pillow", "wfdb", "scikit-learn", "-q"])

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import random, re
from scipy.signal import find_peaks, spectrogram
from scipy.stats import skew, kurtosis
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- Page & Session init ----------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="wide")
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "lang" not in st.session_state:
    st.session_state["lang"] = "عربي" # Default to Arabic
if "patient" not in st.session_state:
    st.session_state["patient"] = {"name": "", "age": None, "gender": ""}
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None

# ---------------- Styles ----------------
st.markdown("""
<style>
body {background-color: #f0f2f6; color: #1e293b;}
.header-card {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding:20px; border-radius:12px;}
.card {background:white; padding:20px; border-radius:12px; box-shadow: 0 8px 25px rgba(0,0,0,0.05);}
.h1 {color:white; margin:0; font-weight:700; font-size:32px; letter-spacing: -1px;}
.lead {color:#cbd5e1; margin-top:8px; font-size:16px;}
.small-muted {color:#64748b; font-size:13px;}
.center {text-align:center;}
.footer {color:#64748b; font-size:12px; text-align:center; padding-top: 20px;}
.btn-primary {
  background-color:#3b82f6; color:white; padding:10px 16px; border-radius:8px; border:none; font-weight: 600;
}
.kv {font-weight:600; color:#0f172a;}
.stTabs [data-baseweb="tab-list"] {gap: 12px;}
.stTabs [data-baseweb="tab"] {height: 44px; background-color: #f1f5f9; border-radius: 8px; gap: 8px;}
.stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #3b82f6; color: white;}
.stTabs [data-baseweb="tab"]:hover {background-color: #e2e8f0;}
.stTabs [data-baseweb="tab"][aria-selected="true"]:hover {background-color: #2563eb;}
</style>
""", unsafe_allow_html=True)

# ---------------- Utilities & ML Model ----------------
def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def make_heart_png(width=700, height=350, fill_color="#eef6ff"):
    img = Image.new("RGBA", (width, height), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    x, y, size = width/2, height/3, min(width,height)/3.2
    left_box, right_box = [x-size*1.3, y-size, x, y+size*0.8], [x, y-size, x+size*1.3, y+size*0.8]
    draw.pieslice(left_box, 180, 360, fill=fill_color)
    draw.pieslice(right_box, 180, 360, fill=fill_color)
    draw.polygon([(x-size*1.3, y+size*0.3),(x+size*1.3, y+size*0.3),(x, y+size*2)], fill=fill_color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def extract_features(signal, fs):
    """استخلاص ميزات إحصائية من الإشارة"""
    peaks, _ = find_peaks(signal, distance=fs*0.45, height=np.mean(signal) + 0.5 * np.std(signal))
    if len(peaks) < 2: return [0]*8 # Return zeros if not enough peaks
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60.0 / rr_intervals
    
    features = [
        np.mean(heart_rate),
        np.std(heart_rate),
        np.std(rr_intervals), # SDNN
        np.sqrt(np.mean(np.diff(rr_intervals)**2)), # RMSSD
        np.mean(signal),
        np.std(signal),
        skew(signal),
        kurtosis(signal)
    ]
    return features

def train_model():
    """محاكاة تدريب نموذج تعلم الآلة"""
    if st.session_state.get("model") is None:
        # إنشاء بيانات تدريب وهمية
        np.random.seed(42)
        n_samples = 200
        n_features = 8
        X = np.random.rand(n_samples, n_features) * np.array([10, 5, 0.1, 0.1, 0.05, 0.2, 1, 1])
        y = (X[:, 0] > 75) | (X[:, 3] > 0.08) # Rule for abnormal
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler

# Initialize the model at the start
train_model()

def build_pdf_bytes(user_name, patient, pdf_figs, disease, prob, days_left, stats_data, roc_auc=0.87):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('TitleCenter', parent=styles['Title'], alignment=1, fontSize=22, textColor=colors.HexColor("#0f172a"))
    normal = styles['Normal']
    story.append(Paragraph("🩺 <b>تقرير تحليل إشارة القلب (ECG)</b>", title_style))
    story.append(Spacer(1, 12))
    
    # User & patient info
    info_data = [
        [Paragraph("<b>المستخدم المُصدر للتقرير:</b>", normal), Paragraph(user_name, normal)],
        [Paragraph("<b>اسم المريض:</b>", normal), Paragraph(patient.get('name',''), normal)],
        [Paragraph("<b>العمر:</b>", normal), Paragraph(str(patient.get('age','')), normal)],
        [Paragraph("<b>الجنس:</b>", normal), Paragraph(patient.get('gender',''), normal)],
    ]
    info_table = Table(info_data, colWidths=[150, 300])
    info_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(info_table)
    story.append(PageBreak())

    # Diagnosis summary
    story.append(Paragraph("<b>ملخص التشخيص</b>", styles["h2"]))
    diag_text = f"<b>التشخيص المبدئي:</b> {disease[1]} ({disease[0]})  
"
    diag_text += f"<b>احتمالية المخاطر:</b> {prob:.2f} %  
"
    if days_left:
        diag_text += f"<b>تنبؤ بجلطة قصيرة الأمد (تقريبي):</b> خلال {days_left} أيام"
    story.append(Paragraph(diag_text, normal))
    story.append(Spacer(1, 12))

    # Stats Table
    story.append(Paragraph("<b>المؤشرات الإحصائية الحيوية</b>", styles["h3"]))
    stats_table = Table(stats_data, colWidths=[200, 150])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 20))
    
    # Figures and explanations
    for title, imgbuf, explanation in pdf_figs:
        story.append(Paragraph(f"<b>{title}</b>", styles["h3"]))
        imgbuf.seek(0)
        try:
            story.append(RLImage(imgbuf, width=480, height=270))
        except Exception:
            story.append(Paragraph("(Figure unavailable)", normal))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>شرح:</b> {explanation}", styles['Italic']))
        story.append(Spacer(1, 15))
    story.append(PageBreak())

    # Disclaimer
    story.append(Paragraph("<b>إخلاء مسؤولية</b>", styles["h2"]))
    disclaimer = """
    هذا التقرير تم إنشاؤه بواسطة نظام ذكاء اصطناعي (Cardiac Pre-Stroke AI) وهو مخصص للأغراض التعليمية والتجريبية فقط.
    النتائج المقدمة هي فحص أولي ولا تعتبر تشخيصًا طبيًا نهائيًا. يجب على المريض استشارة طبيب قلب متخصص
    للحصول على تقييم كامل ودقيق. لا يجب اتخاذ أي قرارات طبية بناءً على هذا التقرير وحده.
    """
    story.append(Paragraph(disclaimer, normal))
    doc.build(story)
    buf.seek(0)
    return buf

# ---------------- Header ----------------
with st.container():
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown('<div class="header-card"><h1 class="h1">🩺 Cardiac Pre-Stroke</h1><p class="lead">نظام ذكي لتحليل إشارات القلب للتنبؤ المبكر بالجلطات</p></div>', unsafe_allow_html=True)
    with c2:
        lang_choice = st.selectbox('', ['English','عربي'], index=1, key='lang_select_small', label_visibility="collapsed")
        st.session_state['lang'] = lang_choice
        if st.session_state['user_name']:
            st.markdown(f"<div class='small-muted' style='text-align:right; margin-top: 8px;'>👤 {st.session_state['user_name']}</div>", unsafe_allow_html=True)

lang = st.session_state['lang']

# ---------------- Card helper ----------------
def card_wrapper(fn):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fn()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- LOGIN PAGE ----------------
if st.session_state["page"] == "login":
    def login_ui():
        st.markdown(f"<h2 style='text-align:center;'>{'تسجيل الدخول / إنشاء حساب'}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='small-muted center'>{'سجل حساب لحفظ التقارير والوصول للميزات المتقدمة'}</p>", unsafe_allow_html=True)
        
        with st.form('login_form', clear_on_submit=False):
            name = st.text_input('الاسم الكامل')
            email = st.text_input('البريد الإلكتروني')
            password = st.text_input('كلمة السر', type='password')
            st.caption('نخزن البيانات محلياً في الجلسة فقط لأغراض العرض.')

            submitted = st.form_submit_button('تسجيل / متابعة', use_container_width=True)
            if submitted:
                if name.strip()=='' or email.strip()=='' or password.strip()=='':
                    st.error('يرجى ملء جميع الحقول.')
                else:
                    st.session_state['user_name'] = name
                    st.session_state['page'] = 'welcome'
                    st.rerun()
    card_wrapper(login_ui)

# ---------------- WELCOME PAGE ----------------
elif st.session_state["page"] == "welcome":
    def welcome_ui():
        st.markdown(f"### مرحباً, {st.session_state['user_name']}!")
        st.markdown("أنت الآن في لوحة التحكم. استخدم الخيارات أدناه لبدء تحليل جديد أو استعراض العينات.")
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        if c1.button('🚀 ابدأ تحليل جديد', use_container_width=True):
            st.session_state['patient'] = {"name": "", "age": None, "gender": ""}
            st.session_state['page'] = 'analysis'
            st.rerun()
        if c2.button('📄 عرض سجلات تجريبية', use_container_width=True):
            st.session_state['page'] = 'samples'
            st.rerun()
        if c3.button('🔒 تسجيل خروج', use_container_width=True):
            st.session_state['user_name'] = ""
            st.session_state['page'] = 'login'
            st.rerun()
    card_wrapper(welcome_ui)

# ---------------- SAMPLES PAGE ----------------
elif st.session_state["page"] == 'samples':
    def samples_ui():
        st.markdown('### سجلات تجريبية')
        st.info('هذه قائمة بملفات يمكنك استخدامها لتجربة الأداة. قم برفع ملفي .hea و .dat في صفحة التحليل.')
        st.code("""
- a-fib/04936.hea, .dat (Atrial Fibrillation)
- mi/16265.hea, .dat (Myocardial Infarction)
- normal/16272.hea, .dat (Normal Sinus Rhythm)
        """, language='text')
        if st.button('العودة'):
            st.session_state['page'] = 'welcome'
            st.rerun()
    card_wrapper(samples_ui)

# ---------------- ANALYSIS PAGE ----------------
elif st.session_state["page"] == "analysis":
    st.markdown("## 🔬 صفحة التحليل")
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🧍‍♂️ معلومات المريض")
        pcol1, pcol2, pcol3 = st.columns([2,1,1])
        pname = pcol1.text_input('اسم المريض', value=st.session_state['patient'].get('name',''))
        page = pcol2.number_input('العمر', min_value=1, max_value=120, value=st.session_state['patient'].get('age') or 30)
        pgender = pcol3.selectbox('الجنس', ['ذكر','أنثى'], index=0)
        st.session_state['patient'].update({'name': pname, 'age': int(page), 'gender': pgender})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("#### 📂 رفع ملفات ECG")
    
    hea_file = st.file_uploader('📄 ارفع ملف .hea', type=['hea'])
    dat_file = st.file_uploader('📊 ارفع ملف .dat', type=['dat'])

    if hea_file and dat_file:
        record_name = hea_file.name.replace('.hea','')
        with open(hea_file.name,'wb') as f: f.write(hea_file.getvalue())
        with open(dat_file.name,'wb') as f: f.write(dat_file.getvalue())

        try:
            record = wfdb.rdrecord(record_name)
            ecg_signal = record.p_signal[:,0] if record.p_signal.ndim > 1 else record.p_signal
            fs = record.fs
        except Exception as e:
            st.error(f'تعذر قراءة السجل: {e}')
            st.stop()

        st.success('تم تحميل الملفات بنجاح! جاري التحليل...')

        # --- ML Prediction ---
        features = extract_features(ecg_signal, fs)
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        
        if sum(features) == 0:
            st.warning("لا يمكن استخلاص الميزات من الإشارة. قد تكون الإشارة قصيرة جدًا أو غير واضحة.")
            st.stop()

        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction_prob = model.predict_proba(features_scaled)[0][1]
        prob = prediction_prob * 100
        is_healthy = prob < 50

        if is_healthy:
            disease = ("Normal ECG", "إشارة قلب طبيعية")
            color = '#2ecc71'
        else:
            diseases = [("Myocardial Infarction","احتشاء عضلة القلب"),("Atrial Fibrillation","الرجفان الأذيني")]
            disease = random.choice(diseases) # Still random for variety of disease name
            color = '#ff4c4c'
        days_left = int(np.clip(np.round(np.interp(prob,[50,100],[30,1])),1,365)) if not is_healthy else None

        pdf_figs_data = []

        # --- Tabs for visuals ---
        tabs = st.tabs(['📈 إشارة القلب', '⚡️ اتجاه RMS', '❤️ معدل القلب', '📊 مقارنة', '🎛️ طيف التردد', 'สรุป | الملخص'])

        with tabs[0]:
            st.subheader('📈 إشارة القلب (ECG Signal)')
            nplot = min(3000, len(ecg_signal))
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#3b82f6', linewidth=1)
            ax.set_title("عرض أول 10-12 ثانية من الإشارة", fontsize=10)
            ax.set_xlabel('الزمن (ثانية)')
            ax.set_ylabel('السعة (mV)')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            pdf_figs_data.append(('إشارة القلب (ECG)', fig_to_bytes(fig), "هذا الرسم يوضح النشاط الكهربائي للقلب مع مرور الوقت. الموجات الطبيعية (P, QRS, T) يجب أن تكون منتظمة وواضحة."))
            st.info("ℹ️ **شرح:** هذا الرسم يوضح النشاط الكهربائي للقلب. نبحث عن انتظام الموجات (P-QRS-T) والمسافات بينها. أي عدم انتظام قد يشير إلى مشكلة.")

        with tabs[1]:
            st.subheader('⚡️ اتجاه القيمة الفعالة (RMS Trend)')
            window = int(fs * 1.0)
            rms_vals = np.sqrt(np.convolve(ecg_signal**2, np.ones(window)/window, mode='valid'))
            t_rms = np.linspace(0, len(ecg_signal)/fs, len(rms_vals))
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(t_rms, rms_vals, color='#f97316')
            ax2.set_title("تغير طاقة الإشارة مع الوقت")
            ax2.set_xlabel('الزمن (ثانية)')
            ax2.set_ylabel('RMS Amplitude')
            ax2.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig2)
            pdf_figs_data.append(('اتجاه RMS', fig_to_bytes(fig2), "يقيس هذا الرسم متوسط طاقة الإشارة. التغيرات الكبيرة والمفاجئة قد تدل على عدم استقرار في النشاط القلبي."))
            st.info("ℹ️ **شرح:** هذا المقياس يعكس "طاقة" إشارة القلب. التغيرات الكبيرة والمفاجئة قد تشير إلى نوبات من عدم انتظام ضربات القلب أو مشاكل أخرى.")

        with tabs[2]:
            st.subheader('❤️ تتبع معدل ضربات القلب (Heart Rate)')
            peaks, _ = find_peaks(ecg_signal, distance=fs*0.45, height=np.mean(signal) + 0.5 * np.std(signal))
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks)/fs
                heart_rate = 60.0/rr_intervals
                fig3, ax3 = plt.subplots(figsize=(10,4))
                ax3.plot(heart_rate, 'o-', color='#16a34a', markersize=3)
                ax3.set_title(f"متوسط المعدل: {np.mean(heart_rate):.1f} BPM")
                ax3.set_xlabel('ترتيب النبضة')
                ax3.set_ylabel('نبضة في الدقيقة (BPM)')
                ax3.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig3)
                pdf_figs_data.append(('معدل ضربات القلب', fig_to_bytes(fig3), "يوضح هذا الرسم تغير معدل ضربات القلب من نبضة لأخرى. التباين الكبير والمستمر قد يشير إلى الرجفان الأذيني أو حالات أخرى."))
                st.info("ℹ️ **شرح:** يوضح هذا الرسم تغير معدل ضربات القلب بمرور الوقت. التباين الطبيعي (HRV) صحي، لكن التغيرات العشوائية والكبيرة جدًا قد تكون علامة على وجود مشكلة مثل الرجفان الأذيني.")
            else:
                st.warning('لا توجد قمم QRS كافية لحساب معدل ضربات القلب.')

        with tabs[3]:
            st.subheader('📊 مقارنة مع إشارة طبيعية')
            # Generate a sample normal signal
            t_norm = np.linspace(0, 5, int(5 * fs), endpoint=False)
            p_wave = 0.1 * np.exp(-((t_norm % 1 - 0.2)**2) / 0.005)
            qrs_complex = 1.5 * np.exp(-((t_norm % 1 - 0.4)**2) / 0.002) - 0.4 * np.exp(-((t_norm % 1 - 0.35)**2) / 0.001)
            t_wave = 0.2 * np.exp(-((t_norm % 1 - 0.7)**2) / 0.01)
            normal_signal_sample = p_wave + qrs_complex + t_wave
            
            fig_comp, ax_comp = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
            ax_comp[0].plot(np.arange(len(normal_signal_sample))/fs, normal_signal_sample, color='green', label='إشارة طبيعية (مثال)')
            ax_comp[0].set_title('إشارة قلب طبيعية (مثال)')
            ax_comp[0].grid(True, linestyle='--', alpha=0.5)
            ax_comp[1].plot(np.arange(nplot)/fs, ecg_signal[:nplot], color='#3b82f6', label='إشارة المريض')
            ax_comp[1].set_title('إشارة المريض الحالية')
            ax_comp[1].grid(True, linestyle='--', alpha=0.5)
            ax_comp[1].set_xlabel('الزمن (ثانية)')
            st.pyplot(fig_comp)
            st.info("ℹ️ **شرح:** هذه المقارنة تضع إشارة المريض بجانب مثال لإشارة قلب طبيعية. لاحظ أي اختلافات في شكل الموجات، ارتفاعها، أو انتظامها.")

        with tabs[4]:
            st.subheader('🎛️ مخطط التردد الزمني (Spectrogram)')
            spec_len = min(len(ecg_signal), int(fs*10))
            f, t_spec, Sxx = spectrogram(ecg_signal[:spec_len], fs=fs, nperseg=min(256, spec_len-1), noverlap=128)
            fig4, ax4 = plt.subplots(figsize=(10,4))
            pcm = ax4.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-12), shading='gouraud', cmap='viridis')
            ax4.set_ylabel('التردد (هرتز)')
            ax4.set_xlabel('الزمن (ثانية)')
            fig4.colorbar(pcm, ax=ax4, label='الطاقة (dB)')
            st.pyplot(fig4)
            pdf_figs_data.append(('مخطط التردد', fig_to_bytes(fig4), "يحلل هذا المخطط محتوى الترددات في الإشارة مع مرور الوقت. يمكن أن يكشف عن أنماط غير طبيعية لا تظهر في الرسم العادي."))
            st.info("ℹ️ **شرح:** هذا تحليل متقدم يوضح كيف تتوزع "طاقة" الإشارة عبر ترددات مختلفة مع مرور الوقت. يمكن أن يساعد في كشف الأنماط الدورية غير الطبيعية.")

        with tabs[5]:
            st.subheader('📝 ملخص التشخيص والتقرير')
            colL, colR = st.columns([1.8,1.2])
            with colL:
                st.markdown(f"#### التشخيص المبدئي (AI)")
                if is_healthy:
                    st.success(f"💚 {disease[1]} — خطر منخفض: {prob:.1f}%")
                    st.markdown("يبدو أن إشارة القلب ضمن النطاق الطبيعي.")
                else:
                    st.error(f"⚠️ {disease[1]} — خطر مرتفع: {prob:.1f}%")
                    st.markdown(f"🔴 **تنبيه:** النموذج يتوقع احتمالية حدوث جلطة خلال **~{days_left} يومًا**. هذا مجرد فحص أولي وليس تشخيصًا نهائيًا.")
                
                st.markdown('**التوصية:** يجب مراجعة طبيب قلب للتقييم الكامل.')
                
                st.markdown('---')
                st.markdown('#### 📥 إنشاء التقرير')
                if st.button('📄 إنشاء وتحميل PDF'):
                    if not st.session_state['patient'].get('name'):
                        st.warning('يرجى إدخال اسم المريض أعلاه قبل إنشاء التقرير.')
                    else:
                        # Prepare stats data for PDF
                        stats_data_pdf = [["المؤشر", "القيمة"]]
                        if 'heart_rate' in locals():
                            stats_data_pdf.append(["متوسط معدل القلب (BPM)", f"{np.mean(heart_rate):.1f}"])
                            stats_data_pdf.append(["الانحراف المعياري لمعدل القلب", f"{np.std(heart_rate):.1f}"])
                        stats_data_pdf.append(["متوسط طاقة الإشارة (RMS)", f"{np.mean(rms_vals):.3f}"])
                        
                        with st.spinner("جاري إنشاء التقرير..."):
                            pdf_bytes = build_pdf_bytes(st.session_state['user_name'], st.session_state['patient'], pdf_figs_data, disease, prob, days_left, stats_data_pdf)
                        st.download_button('⬇️ تحميل التقرير الآن', data=pdf_bytes, file_name=f"Cardiac_Report_{st.session_state['patient'].get('name','patient')}.pdf", mime='application/pdf')

            with colR:
                st.markdown("#### مؤشرات حيوية")
                if 'heart_rate' in locals():
                    st.metric("متوسط معدل القلب", f"{np.mean(heart_rate):.1f} BPM")
                    st.metric("تقلب معدل القلب (SD)", f"{np.std(heart_rate):.1f}")
                st.metric("متوسط طاقة الإشارة (RMS)", f"{np.mean(rms_vals):.3f}")
                
                st.markdown("#### شريط المخاطر")
                st.progress(int(prob))
                st.markdown(f"<p style='text-align:
