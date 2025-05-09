import pandas as pd
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
)
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
from train_models import train_and_evaluate_models

csv_path = "synthetic_senior_features.csv"
df = pd.read_csv(csv_path)

# Пулы моделей
CLASSIFIERS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    ),
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='saga', max_iter=2000, class_weight='balanced'
        )
    ),
    "XGBoost": lambda: xgb.XGBClassifier(eval_metric="logloss")
}

REGRESSORS = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "ElasticNet": lambda: ElasticNet(),
    "XGBoost": lambda: xgb.XGBRegressor()
}

# Метрики
clf_metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y, yhat: precision_score(y, yhat, zero_division=0),
    "recall":    lambda y, yhat: recall_score(y, yhat, zero_division=0),
    "f1":        lambda y, yhat: f1_score(y, yhat, zero_division=0),
    "roc_auc":   lambda y, yhat, ys=None: roc_auc_score(y, ys) if ys is not None else None,
    "pr_auc":    lambda y, yhat, ys=None: average_precision_score(y, ys) if ys is not None else None
}

reg_metrics = {
    "r2":  r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error
}

# Обучаем классификатор health_event
feature_cols_health = [
    "T_out", "monthly_income", "debt_to_income_ratio",
    "diabetes", "hypertension", "cvd",
    "diabetes_meds", "hypertension_meds", "cvd_meds",
    "living_alone", "has_caregiver"
]
train_and_evaluate_models(
    df=df,
    feature_cols=feature_cols_health,
    target_col="health_event",
    task="classification",
    model_pool=CLASSIFIERS,
    metrics=clf_metrics,
    primary_metric="f1",
    output_dir="trained_models",
    output_prefix="health_event"
)

# Обучаем классификатор hospitalized
feature_cols_hosp = [
    "health_event", "T_out", "diabetes", "hypertension", "cvd",
    "living_alone", "has_caregiver"
]
train_and_evaluate_models(
    df=df,
    feature_cols=feature_cols_hosp,
    target_col="hospitalized",
    task="classification",
    model_pool=CLASSIFIERS,
    metrics=clf_metrics,
    primary_metric="f1",
    output_dir="trained_models",
    output_prefix="hospitalized"
)

# mobility_level
feature_cols_mob = [
    "day_of_week", "T_out", "health_event", "hospitalized",
    "living_alone", "has_caregiver"
]
train_and_evaluate_models(
    df=df,
    feature_cols=feature_cols_mob,
    target_col="mobility_level",
    task="classification",
    model_pool=CLASSIFIERS,
    metrics=clf_metrics,
    primary_metric="accuracy",
    output_dir="trained_models",
    output_prefix="mobility_level"
)

# debt_to_income_ratio
feature_cols_fin = ["monthly_income", "monthly_expenses"]
train_and_evaluate_models(
    df=df,
    feature_cols=feature_cols_fin,
    target_col="debt_to_income_ratio",
    task="regression",
    model_pool=REGRESSORS,
    metrics=reg_metrics,
    primary_metric="r2",
    output_dir="trained_models",
    output_prefix="debt_to_income_ratio"
)

# Регрессия для tariff_change
feature_cols_tariff = ["month"]
train_and_evaluate_models(
    df=df,
    feature_cols=feature_cols_tariff,
    target_col="tariff_change",
    task="regression",
    model_pool=REGRESSORS,
    metrics=reg_metrics,
    primary_metric="r2",
    output_dir="trained_models",
    output_prefix="tariff_change"
)
