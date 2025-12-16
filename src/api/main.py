#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI
from typing import List
from pathlib import Path
import pandas as pd
import sys, os

# ------------------------------------------------------
# Project Root
# ------------------------------------------------------
PROJECT_ROOT = Path("/Users/sarahilyas/Life_Expectancy_Regression")
sys.path.append(str(PROJECT_ROOT))

# Import the inference pipeline
from src.Inference.inference import predict

# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(title="Life Expectancy Prediction API")

@app.get("/")
def root():
    return {"message": "Life Expectancy Prediction API is running 🚀"}

@app.post("/predict")
def predict_batch(data: List[dict]):
    """
    Accepts FEATURE-ENGINEERED rows and returns predictions.
    No FE is performed here.
    """
    df = pd.DataFrame(data)

    if df.empty:
        return {"error": "No data provided"}

    # IMPORTANT: Skip FE because FE already applied
    preds_df = predict(df, already_fe=True)

    # inference.py always outputs "prediction" column
    return {
        "predictions": preds_df["prediction"].astype(float).tolist()
    }
