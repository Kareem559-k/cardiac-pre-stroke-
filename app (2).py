# app.py
import streamlit as st
import numpy as np
import pandas as pd
import random, re, ast, os, warnings
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

warnings.filterwarnings("ignore")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🫀", layout="centered")

# Custom CSS (Dark Blue Theme)
st.markdown("""
<style>
body {
    background-color: #0a192f;
    color: #e6f1ff;
}
[data-testid="stSidebar"] {
    background-color: #112240;
}
h1, h2, h3, h4 {
    color: #64ffda;
}
.stButton>button {
    background-color: #112240;
    color: #64ffda;
    border-radius: 10px;
    border: 1px solid #64ffda;
}
.stButton>button:hover {
    background-color: #64ffda;
    color: #0a192f;
}
div.stAlert {
    background-color: #112240 !important;
    border-left: 4px solid #64ffda !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💙 Cardiac Pre-Stroke Risk Predictor")
st.caption("Simulated visualization — for demo and presentation only.")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Simulation Settings ⚙️")
demo_mode = st.sidebar.checkbox("Enable Simulation", True)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
randomness = st.sidebar.slider("Variability", 0.01, 0.4, 0.18, 0.01)
borderline_chance = st.sidebar.slider("Borderline Chance (%)", 0, 40, 10, 1)

random.seed(int(seed))
np.random.seed(int(seed))

# ----------------------------
# Upload PTB-XL Metadata (optional)
# ----------------------------
st.markdown("### 📁 Upload PTB-XL Metadata (Optional)")
ptbxl_file = st.file_uploader("Upload ptbxl_database.csv", type=["csv"])
ptbxl_df = None
if ptbxl_file is not None:
    try:
        ptbxl_df = pd.read_csv(ptbxl_file)
        st.success(f"✅ Metadata loaded ({len(ptbxl_df)} records).")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

# ----------------------------
# Upload ECG Files
# ----------------------------
st.markdown("### 📤 Upload ECG Record (.hea + .dat)")
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# ----------------------------
# Utility Functions
# ----------------------------
def extract_numeric_id(name):
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def simulate_result(nid, variability=0.18, borderline_pct=10):
    if nid is None:
        base = random.uniform(0.4, 0.6)
    elif nid % 2 == 1:
        base = random.uniform(0.65, 0.92)  # sick
    else:
        base = random.uniform(0.05, 0.55 if random.uniform(0,100) < borderline_pct else 0.35)
    prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)

    if prob >= 0.60:
        return prob, "Patient", "⚠️ High Risk — Needs medical follow-up.", "high"
    elif prob >= 0.35:
        return prob, "Borderline", "🩺 Borderline — regular monitoring advised.", "medium"
    else:
        return prob, "Not Patient", "💚 Appears healthy.", "low"

def make_probability_bar(prob, severity):
    fig, ax = plt.subplots(figsize=(6,1.2))
    colors = {"high":"#ff4d4d","medium":"#f4c542","low":"#4caf50"}
    ax.barh(["Risk"], [prob], color=colors[severity], height=0.5)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Risk Level")
    ax.text(prob, 0, f"{prob*100:.1f}%", va='center', fontsize=10, fontweight='bold', color='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Main Process
# ----------------------------
if hea_file and dat_file:
    record_name = hea_file.name.replace(".hea", "")
    with open(hea_file.name, "wb") as f: f.write(hea_file.read())
    with open(dat_file.name, "wb") as f: f.write(dat_file.read())

    st.markdown(f"**Record Name:** `{record_name}`")

    try:
        rec = rdrecord(record_name)
        sig = rec.p_signal
        y = sig[:,0] if sig.ndim > 1 else sig

        # ECG waveform
        st.markdown("#### 🩺 ECG Signal (first 2000 samples)")
        fig1, ax1 = plt.subplots(figsize=(8,2.2))
        ax1.plot(y[:2000], color="#64ffda", linewidth=0.9)
        ax1.set_xlim(0, min(2000, len(y)))
        ax1.set_ylabel("Amplitude", color="#e6f1ff")
        ax1.set_xlabel("Samples", color="#e6f1ff")
        ax1.grid(alpha=0.3)
        fig1.patch.set_facecolor("#0a192f")
        st.pyplot(fig1)
        plt.close(fig1)

        # Histogram
        st.markdown("#### 📊 Amplitude Distribution")
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.hist(y, bins=60, color="#00b4d8", alpha=0.9)
        ax2.set_xlabel("Amplitude", color="#e6f1ff")
        ax2.set_ylabel("Count", color="#e6f1ff")
        ax2.grid(alpha=0.2)
        fig2.patch.set_facecolor("#0a192f")
        st.pyplot(fig2)
        plt.close(fig2)

        # RMS Sparkline
        st.markdown("#### ⚡ Signal RMS Trend")
        if len(y) > 50:
            rms = np.sqrt(pd.Series(y).rolling(window=60).mean().fillna(method='bfill').values)
            fig3, ax3 = plt.subplots(figsize=(6,1.2))
            ax3.plot(rms[-200:], color="#64ffda", linewidth=0.9)
            ax3.set_yticks([])
            ax3.set_xticks([])
            fig3.patch.set_facecolor("#0a192f")
            st.pyplot(fig3)
            plt.close(fig3)
    except Exception as e:
        st.warning(f"Unable to render ECG: {e}")
        y = None

    # Metadata lookup
    true_label = "Unknown"
    if ptbxl_df is not None:
        matched = ptbxl_df[ptbxl_df["filename_hr"].astype(str).str.contains(record_name, na=False)]
        if not matched.empty:
            raw_code = matched.iloc[0]["scp_codes"]
            try:
                code_dict = ast.literal_eval(raw_code)
                true_label = list(code_dict.keys())[0]
            except:
                true_label = str(raw_code)
            st.markdown(f"**🧾 Database Label:** `{true_label}`")

    # Simulation
    if demo_mode:
        nid = extract_numeric_id(record_name)
        prob, label, msg, severity = simulate_result(nid, variability=randomness, borderline_pct=borderline_chance)

        color_bg = {"high":"#331a1a","medium":"#2f2a00","low":"#1a3320"}[severity]
        st.markdown(f"""
        <div style='background:{color_bg};padding:16px;border-radius:12px;text-align:center;font-size:16px'>
            <b>{label}</b><br>{msg}<br><br><b>Risk Probability:</b> {prob*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge
        st.markdown("#### 📈 Simulated Risk Gauge")
        img_bytes = make_probability_bar(prob, severity)
        st.image(img_bytes, use_container_width=True)

else:
    st.info("Please upload both `.hea` and `.dat` files to start analysis.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("💙 Cardiac Pre-Stroke © 2025 — Dark Blue Edition — For demo only.")
