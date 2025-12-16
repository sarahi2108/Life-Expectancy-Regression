#!/usr/bin/env python
# coding: utf-8

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
API_URL = os.getenv("API_URL", "http://api:8000/predict")


s3 = boto3.client("s3", region_name=REGION)

# =============================
# S3 FETCH FUNCTION
# =============================
def load_from_s3(key, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        st.info(f"📥 Downloading {key} from S3…")
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# =============================
# ARTIFACT PATHS
# =============================
HOLDOUT_PATH = load_from_s3(
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
    fe = pd.read_csv(HOLDOUT_PATH)
    labels = pd.read_csv(LABELS_PATH)

    # Ensure Life Expectancy column exists in labels
    if "Life Expectancy" not in labels.columns:
        labels["Life Expectancy"] = None

    # Align if mismatch
    if len(fe) != len(labels):
        st.warning("Row mismatch — aligning by index")
        n = min(len(fe), len(labels))
        fe = fe.iloc[:n]
        labels = labels.iloc[:n]

    return fe, labels

fe_df, label_df = load_data()

# =============================
# DROP TARGET BEFORE INFERENCE
# =============================
MODEL_FEATURES = [
    "80th Percentile Income",
    "% Voter Turnout",
    "% Excessive Drinking",
    "Men's Median Earnings",
    "% Households with Broadband Access",
    "% with Annual Mammogram",
    "% with access to parks",
    "% Frequent Physical Distress",
    "% Physically Inactive",
    "% Children in Poverty",
    "% Adults with Obesity",
    "% Low Birth Weight",
    "% Drive Alone to Work"
]

fe_predictors = fe_df[MODEL_FEATURES].copy()

# =============================
# UI
# =============================
st.title("🧬 Life Expectancy Prediction Viewer")

if st.button("Run Predictions 🚀"):
    payload = fe_predictors.to_dict(orient="records")

    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()

        preds = resp.json().get("predictions", [])

        result_df = label_df.copy()
        result_df["Prediction"] = preds

        st.subheader("📊 Predictions")
        st.dataframe(result_df.head(50))

        # ---- Metrics if actual present ----
        if result_df["Life Expectancy"].notna().any():
            mae = (result_df["Prediction"] - result_df["Life Expectancy"]).abs().mean()
            rmse = ((result_df["Prediction"] - result_df["Life Expectancy"]) ** 2).mean() ** 0.5

            c1, c2 = st.columns(2)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)
