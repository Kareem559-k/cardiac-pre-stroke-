import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

# إنشاء الشكل
fig, ax = plt.subplots(figsize=(6, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

# ترتيب المربعات من الأعلى للأسفل
boxes = [
    ("1- Upload ECG File\n(HEA / DAT / Feature)", 10.5),
    ("2- Signal Visualization\n(Automatic ECG Plot)", 8.8),
    ("3- AI Prediction\n(Process ECG data)", 7.1),
    ("4- Output Result\n(Normal / High Stroke Risk)", 5.4),
    ("5- Integration\n(Wearable or Clinical Use)", 3.7)
]

# رسم المربعات (أصغر في العرض والارتفاع)
for text, y in boxes:
    ax.add_patch(FancyBboxPatch((2.6, y), 4.8, 0.7,   # أصغر من السابق
                                boxstyle="round,pad=0.15",
                                facecolor="#7A2721", edgecolor="black", lw=1.2))
    ax.text(5, y + 0.35, text, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="white")

# رسم الأسهم القصيرة جدًا
for i in range(len(boxes) - 1):
    y1 = boxes[i][1]
    ax.add_patch(FancyArrow(5, y1 - 0.25, 0, -0.35,   # أقصر سهم
                            width=0.006, head_length=0.12, head_width=0.12,
                            color="black", length_includes_head=True))

# العنوان في الأعلى
ax.text(5, 11.7, "Cardiac Pre-Stroke Predictor Flow",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#7A2721")

# النص السفلي
ax.text(5, 1.0, "AI-powered ECG Diagnosis & Risk Prediction",
        ha="center", va="center", fontsize=7.3, color="gray")

plt.tight_layout()
plt.show()
