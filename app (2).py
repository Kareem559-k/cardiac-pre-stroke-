import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# ----------------------------------------------------------
# Heart cover image
# ----------------------------------------------------------
def make_heart_png(width=600, height=300, fill_color="#eef6ff"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "❤️", fontsize=120, ha="center", va="center")
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf

# ----------------------------------------------------------
# Confusion Matrix WITHOUT Seaborn
# ----------------------------------------------------------
def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(4, 3))

    im = ax.imshow(conf_matrix, cmap="Blues")

    # Write numbers inside each cell
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]),
                    ha="center", va="center", color="white", fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    fig.colorbar(im)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ----------------------------------------------------------
# ROC Curve
# ----------------------------------------------------------
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf, roc_auc

# ----------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------
st.title("Cardiac Pre-Stroke – PDF Report Generator")

lang = "English"
pdf_figs = {}

# Your real model data (replace them with your actual values)
conf_matrix = np.array([[631, 336],
                        [4, 3029]])

y_true = np.array([0, 1, 1, 0, 1, 1, 0])  
y_scores = np.array([0.2, 0.8, 0.95, 0.3, 0.9, 0.87, 0.4])

cm_buf = plot_confusion_matrix(conf_matrix)
roc_buf, roc_auc_value = plot_roc_curve(y_true, y_scores)

# ----------------------------------------------------------
# PDF Builder
# ----------------------------------------------------------
st.markdown("### 📥 Download Report")

if st.button("📄 Generate & Download Report"):

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=18
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleCenter',
        parent=styles['Title'],
        alignment=1,
        fontSize=20,
        textColor=colors.HexColor("#1E90FF")
    )
    normal = styles["Normal"]

    story = []

    # ---------- COVER PAGE ----------
    heart_img_buf = make_heart_png()

    story.append(Spacer(1, 40))
    story.append(Paragraph("🩺 <b>Cardiac Pre-Stroke</b>", title_style))
    story.append(Spacer(1, 6))

    subtitle = (
        "AI-powered ECG Analyzer for Early Detection"
        if lang == "English"
        else "نظام ذكاء اصطناعي لتحليل إشارات القلب واكتشاف الأمراض مبكرًا"
    )

    story.append(Paragraph(subtitle, ParagraphStyle(
        'sub', parent=styles['Normal'],
        alignment=1, fontSize=11, textColor=colors.grey
    )))

    story.append(Spacer(1, 10))

    try:
        img_cover = RLImage(heart_img_buf, width=420, height=220)
        story.append(Spacer(1, 20))
        story.append(img_cover)
    except:
        story.append(Paragraph("(Heart image error)", normal))

    story.append(PageBreak())

    # ---------- MODEL PERFORMANCE ----------
    story.append(Paragraph("<b>Model Performance Report</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overall Accuracy: 0.9150", normal))
    story.append(Spacer(1, 12))

    # Classification Report
    story.append(Paragraph("<b>Classification Report:</b>", styles["Heading3"]))

    class_report_text = """
precision    recall  f1-score   support

0      0.994     0.653     0.788       967
1      0.900     0.999     0.947      3033

accuracy                          0.915      4000
macro avg      0.947     0.826     0.867      4000
weighted avg   0.923     0.915     0.908      4000
"""
    story.append(Paragraph(f"<pre>{class_report_text}</pre>", normal))
    story.append(Spacer(1, 20))

    # ---------- CONFUSION MATRIX ----------
    story.append(Paragraph("<b>Confusion Matrix Heatmap:</b>", styles["Heading3"]))
    try:
        story.append(RLImage(cm_buf, width=420, height=300))
    except:
        story.append(Paragraph("(Confusion matrix error)", normal))

    story.append(Spacer(1, 20))

    # ---------- ROC CURVE ----------
    story.append(Paragraph("<b>ROC Curve:</b>", styles["Heading3"]))
    try:
        story.append(RLImage(roc_buf, width=420, height=300))
        story.append(Paragraph(f"AUC Score: {roc_auc_value:.3f}", normal))
    except:
        story.append(Paragraph("(ROC image error)", normal))

    story.append(Spacer(1, 20))

    # Footer
    story.append(Paragraph(
        "Generated by Cardiac Pre-Stroke AI system.",
        ParagraphStyle('small', parent=styles['Italic'], fontSize=9)
    ))

    # ---------- BUILD PDF ----------
    doc.build(story)
    buffer.seek(0)

    st.download_button(
        "⬇️ Download PDF Report",
        data=buffer.getvalue(),
        file_name="Cardiac_PreStroke_Report.pdf",
        mime="application/pdf"
    )
