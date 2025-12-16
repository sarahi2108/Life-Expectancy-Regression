"""
tune.py

Hyperparameter tuning for XGBoost with Optuna + MLflow,
using feature_engineered_train/eval, like in 06_hyperparameter_tuning_MLflow.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


PROC_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_fe_data(target: str = "Life Expectancy"):
    train_df = pd.read_csv(PROC_DIR / "feature_engineered_train.csv")
    eval_df = pd.read_csv(PROC_DIR / "feature_engineered_eval.csv")

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]

    return X_train, y_train, X_eval, y_eval


def objective(trial: optuna.Trial) -> float:
    X_train, y_train, X_eval, y_eval = load_fe_data()

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        
        mse = mean_squared_error(y_eval, preds)
        rmse = np.sqrt(mse)
        rmse = mean_squared_error(y_eval, preds)

        mlflow.log_metric("rmse", rmse)

    return rmse


def tune_model(
    study_name: str = "xgb_tuning",
    n_trials: int = 30,
    best_model_path: Path = ARTIFACT_DIR / "best_model_xgb.pkl",
):
    mlflow.set_experiment("xgb_hyperparameter_tuning")

    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    print("=== Best trial ===")
    print("Params:", study.best_trial.params)
    print("Best RMSE:", study.best_value)

    # Refit best model on full train
    X_train, y_train, X_eval, y_eval = load_fe_data()
    best_params = study.best_trial.params
    best_params["random_state"] = 42

    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, best_model_path)
    print(f"✅ Saved best tuned model to {best_model_path}")

    return study, best_model


if __name__ == "__main__":
    tune_model()
