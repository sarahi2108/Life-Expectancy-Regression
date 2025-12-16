from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from joblib import load

# ----------------------------
# Project Root & Imports
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

print("📂 Using project root:", PROJECT_ROOT)

from src.Feature_Pipeline.feature_engineering import (
    freq_encode_state,
    drop_state,
    select_top_features,
)
from src.Feature_Pipeline.load import drop_missing_columns

# ----------------------------
# Paths (RELATIVE & DOCKER-SAFE)
# ----------------------------
DEFAULT_MODEL = PROJECT_ROOT / "src" / "Training_Pipeline" / "artifacts" / "best_model_xgb.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
TRAIN_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaning_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

# ----------------------------
# Load strict schema from training
# ----------------------------
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1).columns
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols if c != "Life Expectancy"]
else:
    TRAIN_FEATURE_COLUMNS = None
    print("⚠️ Missing feature_engineered_train.csv → cannot enforce schema strictly")

# ----------------------------
# Core FE for inference only
# ----------------------------
def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    if not TRAIN_CLEAN_PATH.exists():
        raise FileNotFoundError(f"❌ Missing training cleaned data at {TRAIN_CLEAN_PATH}")

    df = drop_missing_columns(df)

    train_clean = pd.read_csv(TRAIN_CLEAN_PATH)

    _, X_inf, _ = freq_encode_state(train_clean, df.copy(), df.copy())

    X_inf = drop_state(X_inf)
    X_inf = select_top_features(X_inf)

    if "Life Expectancy" in X_inf.columns:
        X_inf = X_inf.drop(columns=["Life Expectancy"])

    if TRAIN_FEATURE_COLUMNS is not None:
        X_inf = X_inf.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    return X_inf

# ----------------------------
# Predict
# ----------------------------
def predict(input_df: pd.DataFrame, model_path: Path | str = DEFAULT_MODEL, already_fe=False):
    if already_fe:
        X_inf = input_df.drop(columns=["Life Expectancy"], errors="ignore")
        X_inf = X_inf.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
    else:
        X_inf = prepare_features_for_inference(input_df)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found at {model_path}")

    model = load(model_path)
    preds = model.predict(X_inf)

    out = input_df.loc[X_inf.index].copy()
    out["prediction"] = preds
    return out
