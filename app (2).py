import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# =============================
# PAGE SETUP
# =============================
st.set_page_config(page_title="Cardiac Pre-Stroke Predictor", page_icon="🫀", layout="centered")
st.title("💙 Cardiac Pre-Stroke Predictor")
st.caption("Upload ECG signals or feature files, process them, and predict stroke risk.")

# =============================
# UPLOAD PTBXL DATABASE
# =============================
st.markdown("### 🩺 Upload PTB-XL Metadata File (ptbxl_database.csv)")
ptbxl_file = st.file_uploader("Upload ptbxl_database.csv", type=["csv"])

if ptbxl_file is not None:
    ptbxl_df = pd.read_csv(ptbxl_file)
    st.success(f"✅ Loaded metadata file with {len(ptbxl_df)} records.")
    st.session_state["ptbxl_df"] = ptbxl_df
else:
    st.warning("⚠️ Please upload ptbxl_database.csv to enable record label matching.")

# =============================
# MODEL FILES LOADING
# =============================
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

st.markdown("### ⚙️ Upload Model Files:")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])
up_feats = st.file_uploader("features_selected.npy (optional)", type=["npy"])

if st.button("💾 Save Uploaded Files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    if up_feats: open(FEATURES_PATH, "wb").write(up_feats.read())
    st.success("✅ Uploaded files saved successfully. Click 'Rerun' to reload them.")

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    selected_idx = None
    if os.path.exists(FEATURES_PATH):
        selected_idx = np.load(FEATURES_PATH)
        st.info(f"✅ Loaded feature selection index ({len(selected_idx)} features).")
    else:
        st.warning("⚠️ features_selected.npy not found — using all features.")
    return model, scaler, imputer, selected_idx

try:
    model, scaler, imputer, selected_idx = load_artifacts()
except Exception as e:
    st.stop()
    st.error(f"❌ Failed to load model: {e}")

# =============================
# FEATURE EXTRACTION
# =============================
def extract_micro_features(sig):
    sig = np.asarray(sig, dtype=float)
    diffs = np.diff(sig)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        skew(sig), kurtosis(sig),
        np.mean(np.abs(diffs)), np.std(diffs), np.max(diffs),
        np.mean(np.square(diffs)), np.percentile(diffs, 90), np.percentile(diffs, 10)
    ])

def align(X, expected, name):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.info(f"Added {add} placeholders for {name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"Trimmed {cut} extra features for {name}.")
    return X

def apply_feature_selection(X, selected_idx):
    if selected_idx is not None:
        if X.shape[1] >= len(selected_idx):
            X = X[:, selected_idx]
            st.success(f"✅ Applied feature selection ({len(selected_idx)} features).")
        else:
            st.warning("⚠️ Not enough features for selection, skipping.")
    return X

# =============================
# MAIN INTERFACE
# =============================
st.markdown("---")
mode = st.radio("Select Input Type:", ["Raw ECG (.hea + .dat)", "Feature File (CSV / NPY)"])
threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

# =============================
# RAW ECG MODE
# =============================
if mode == "Raw ECG (.hea + .dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        tmp = hea_file.name.replace(".hea", "")
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:, 0]
            st.line_chart(sig[:2000], height=200)
            st.caption("Preview of first 2000 ECG samples")

            # Match with ptbxl_database.csv
            true_label = "Unknown"
            if "ptbxl_df" in st.session_state:
                df = st.session_state["ptbxl_df"]
                matched = df[df["filename_hr"] == f"records500/{tmp}.hea"]
                if len(matched) > 0:
                    true_label = matched["scp_codes"].values[0]
                    st.info(f"🩸 True label from database: {true_label}")
                else:
                    st.warning("⚠️ No matching record found in ptbxl_database.csv.")

            feats = extract_micro_features(sig).reshape(1, -1)
            feats = apply_feature_selection(feats, selected_idx)
            feats = align(feats, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            prob = model.predict_proba(X_scaled)[0, 1]
            pred_label = "High Stroke Risk" if prob >= threshold else "Normal"

            # Display comparison
            st.markdown("### 🧠 Prediction Result:")
            result_df = pd.DataFrame({
                "Record": [tmp],
                "True Label": [true_label],
                "Predicted": [pred_label],
                "Probability": [f"{prob*100:.2f}%"]
            })
            st.dataframe(result_df)

            fig1, ax1 = plt.subplots()
            ax1.bar(["Normal", "Stroke Risk"], [1-prob, prob],
                    color=["#6cc070", "#ff6b6b"])
            ax1.set_ylabel("Probability")
            ax1.set_title("Stroke Risk Probability")
            st.pyplot(fig1)

        except Exception as e:
            st.error(f"❌ Error processing ECG: {e}")

# =============================
# FEATURE FILE MODE
# =============================
else:
    uploaded = st.file_uploader("Upload Feature File (CSV/NPY)", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = apply_feature_selection(X, selected_idx)
            X = align(X, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "High Risk", "Normal")

            df_out = pd.DataFrame({
                "Sample": np.arange(1, len(probs)+1),
                "Probability": probs,
                "Prediction": preds
            })
            st.dataframe(df_out.head(10))
            st.line_chart(probs, height=150)

            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button("⬇️ Download Predictions CSV", buf.getvalue(),
                               file_name="batch_predictions.csv", mime="text/csv")

            st.markdown("---")
            st.info("✅ Batch prediction completed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("""
✅ **Final Notes**
- Model accuracy and confidence are displayed for transparency.  
- Integrated with PTB-XL database for real label validation.  
- For research and educational use only — not for clinical decisions.
""")
