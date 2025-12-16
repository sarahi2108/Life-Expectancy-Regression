from pathlib import Path
from typing import Tuple

import pandas as pd


PROC_DIR = Path("data/processed")


def freq_encode_state(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Frequency encode State column, matching your notebook:

    state_counts = train_df["State"].value_counts()
    train_df["State_freq"] = train_df["State"].map(state_counts)
    eval_df["State_freq"] = eval_df["State"].map(state_counts).fillna(0)
    holdout_df["State_freq"] = holdout_df["State"].map(state_counts).fillna(0)
    """
    state_counts = train_df["State"].value_counts()

    train_df["State_freq"] = train_df["State"].map(state_counts)
    eval_df["State_freq"] = eval_df["State"].map(state_counts).fillna(0)
    holdout_df["State_freq"] = holdout_df["State"].map(state_counts).fillna(0)

    return train_df, eval_df, holdout_df


def drop_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same definition as in 02_Feature_Eng_Encoding:
    Drop 'State' after encoding.
    """
    cols_to_drop = ["State"]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)

def select_top_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 01_EDA_cleaning:
    Keep only the defined top predictor variables and remove rows 
    containing any missing values within this set.
    """

    # 1. Define selected predictors (unchanged)
    selected_columns = [
        "80th Percentile Income",
        "% Voter Turnout",
        "% Excessive Drinking",
        "Men's Median Earnings",
        "% Households with Broadband Access",
        "% with Annual Mammogram",
        "% with access to parks",
        "% Teen Birth Rate",
        "% Frequent Physical Distress",
        "% Physically Inactive",
        "% Children in Poverty",
        "% Adults with Obesity",
        "% Low Birth Weight",
        "% Drive Alone to Work",
        "Life Expectancy"
    ]

    # 2. Filter to only columns that exist in the dataframe
    cols_present = [c for c in selected_columns if c in df.columns]

    # 3. Subset dataframe to selected columns
    df_selected = df[cols_present].copy()

    # 4. Drop rows containing NaN in any of the selected predictors
    df = df_selected.dropna(axis=0, how="any")

    # 5. Return cleaned dataset
    return df



def build_features(
    train_clean_path: Path = PROC_DIR / "cleaning_train.csv",
    eval_clean_path: Path = PROC_DIR / "cleaning_eval.csv",
    holdout_clean_path: Path = PROC_DIR / "cleaning_holdout.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combine all feature engineering steps:

    - Load cleaned splits
    - Frequency encode State
    - Drop State
    - Add interaction terms (if columns exist)
    - Save feature_engineered_*.csv

    This mirrors 02_Feature_Eng_Encoding.
    """
    train_df = pd.read_csv(train_clean_path)
    eval_df = pd.read_csv(eval_clean_path)
    holdout_df = pd.read_csv(holdout_clean_path)

    # 1. Frequency encoding for State
    train_df, eval_df, holdout_df = freq_encode_state(train_df, eval_df, holdout_df)

    # 2. Drop State
    train_df = drop_state(train_df)
    eval_df = drop_state(eval_df)
    holdout_df = drop_state(holdout_df)

    # 3. Optional: income interaction features
    train_df = select_top_features(train_df)
    eval_df = select_top_features(eval_df)
    holdout_df = select_top_features(holdout_df)

    # 4. Save feature engineered splits
    fe_train_path = PROC_DIR / "feature_engineered_train.csv"
    fe_eval_path = PROC_DIR / "feature_engineered_eval.csv"
    fe_holdout_path = PROC_DIR / "feature_engineered_holdout.csv"

    train_df.to_csv(fe_train_path, index=False)
    eval_df.to_csv(fe_eval_path, index=False)
    holdout_df.to_csv(fe_holdout_path, index=False)

    print("✅ Saved feature engineered splits to data/processed/:")
    print("   feature_engineered_train:", train_df.shape, "->", fe_train_path)
    print("   feature_engineered_eval:", eval_df.shape, "->", fe_eval_path)
    print("   feature_engineered_holdout:", holdout_df.shape, "->", fe_holdout_path)

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    build_features()
