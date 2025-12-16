"""
load.py

Split the original County Health Rankings Excel into train / eval / holdout
and save them under data/raw/, matching the notebook 00_data_split.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_EXCEL = Path("2025 County Health Rankings Data__.xlsx")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def drop_missing_columns(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """
    Drop columns where fraction of missing values is greater than `threshold`.
    (from 00_data_split)
    """
    missing_fraction = df.isna().mean()
    cols_to_keep = missing_fraction[missing_fraction <= threshold].index
    return df[cols_to_keep]


def load_and_split_raw(
    excel_path: Path = RAW_EXCEL,
    target: str = "Life Expectancy",
    missing_threshold: float = 0.10,
    n_strata: int = 10,
    train_frac: float = 0.70,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Mirror the logic of 00_data_split:

    - Read Excel
    - Drop sparse columns
    - Drop rows with missing target
    - Create quantile-based strata on target
    - 70% train, 15% eval, 15% holdout
    - Save as data/raw/*.csv
    """
    df = pd.read_excel(excel_path)

    df = drop_missing_columns(df, threshold=missing_threshold)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    df["strata"] = pd.qcut(df[target], q=n_strata, duplicates="drop")

    temp_frac = 1.0 - train_frac
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_frac,
        random_state=random_state,
        stratify=df["strata"],
    )

    eval_df, holdout_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_df["strata"],
    )

    for d in (train_df, eval_df, holdout_df):
        d.drop(columns=["strata"], inplace=True)

    train_path = RAW_DIR / "train.csv"
    eval_path = RAW_DIR / "eval.csv"
    holdout_path = RAW_DIR / "holdout.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    print("✅ Saved raw splits to data/raw/:")
    print("   train:", train_df.shape, "->", train_path)
    print("   eval:", eval_df.shape, "->", eval_path)
    print("   holdout:", holdout_df.shape, "->", holdout_path)

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_raw()
