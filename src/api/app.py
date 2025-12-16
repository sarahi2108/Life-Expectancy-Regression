import streamlit as st
import pandas as pd
import boto3
import os
from pathlib import Path
import requests

# =============================
# AWS CONFIG
# =============================
S3_BUCKET = os.getenv("S3_BUCKET", "life-expectancy-regression-sarah-2025")
REGION = os.getenv("AWS_REGION", "eu-west-1")
s3 = boto3.client("s3", region_name=REGION)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# =============================
# S3 DOWNLOAD FUNCTION
# =============================
def load_from_s3(key, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        st.info(f"📥 Downloading {key} from S3…")
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# =============================
# FETCH FEATURE-ENGINEERED HOLDOUT
# =============================
FE_HOLDOUT_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)

LABELS_PATH = load_from_s3(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv"
)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    fe = pd.read_csv(FE_HOLDOUT_PATH)
    labels = pd.read_csv(LABELS_PATH)

    # Ensure Life Expectancy exists
    if "Life Expectancy" not in labels.columns:
        labels["Life Expectancy"] = None

    # Align rows just in case
    n = min(len(fe), len(labels))
    fe = fe.iloc[:n]
    labels = labels.iloc[:n]

    return fe, labels

fe_df, label_df = load_data()

# =============================
# UI
# =============================
st.title("🧬 Life Expectancy Prediction Viewer")
st.caption("Using FEATURE-ENGINEERED data → FastAPI skips FE")

if st.button("Run Predictions 🚀"):
    payload = fe_df.to_dict(orient="records")

    try:
        resp = requests.post(API_URL, json=payload)

        if resp.status_code != 200:
            st.error(f"❌ API Error: {resp.status_code}")
            st.write(resp.text)
            st.stop()

        preds = resp.json().get("predictions", [])

        if len(preds) != len(label_df):
            st.error("❌ Prediction length mismatch.")
            st.stop()

        result_df = label_df.copy()
        result_df["Prediction"] = preds

        st.subheader("📊 Predictions")
        st.dataframe(result_df.head(50), use_container_width=True)

        # ---- Metrics if actuals are present ----
        if result_df["Life Expectancy"].notna().any():
            mae = (result_df["Prediction"] - result_df["Life Expectancy"]).abs().mean()
            rmse = ((result_df["Prediction"] - result_df["Life Expectancy"])**2).mean() ** 0.5

            c1, c2 = st.columns(2)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")

    except Exception as e:
        st.error("🚨 Prediction failed.")
        st.exception(e)
