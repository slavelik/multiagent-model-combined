import pickle
import pandas as pd
import os
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
)
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge, Lasso
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import lightgbm as lgb
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from train_models import train_and_evaluate_models

df = pd.read_csv("synthetic_person_features.csv")

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

CLASSIFIERS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    ),
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear', max_iter=5000, class_weight='balanced')
    ),
    "XGBoost": lambda: xgb.XGBClassifier(
        eval_metric="mlogloss", verbosity=0, use_label_encoder=False
    ),
    "CatBoost": lambda: CatBoostClassifier(
        iterations=500, auto_class_weights='Balanced', logging_level='Silent'
    ),
    "LightGBM": lambda: lgb.LGBMClassifier(
        n_estimators=500, verbosity=-1
    ),
    "GradientBoosting": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "KNN": lambda: KNeighborsClassifier()
}

REGRESSORS = {
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "ElasticNet": lambda: ElasticNet(max_iter=10000),
    "XGBoost": lambda: xgb.XGBRegressor(verbosity=0),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=500, logging_level='Silent'
    ),
    "LightGBM": lambda: lgb.LGBMRegressor(
        n_estimators=500, verbosity=-1
    ),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
    "Ridge": lambda: Ridge(max_iter=10000),
    "Lasso": lambda: Lasso(max_iter=10000),
    "KNN": lambda: KNeighborsRegressor()
}

clf_metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y, yhat: precision_score(y, yhat, average='macro', zero_division=0),
    "recall": lambda y, yhat: recall_score(y, yhat, average='macro', zero_division=0),
    "f1": lambda y, yhat: f1_score(y, yhat, average='macro', zero_division=0),
    "roc_auc": lambda y, yhat, ys=None: roc_auc_score(y, ys, multi_class='ovr') if ys is not None else None
}

reg_metrics = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error
}

# Список признаков для каждой целевой переменной
feature_sets = {
  "has_kids":             ["age","family_size","region","income_level"],
  "weekend_relax_factor": ["socialness","occupation","family_size"],
  "movie_enthusiasm":     ["socialness","tv_time","evening_activity_duration"],
  "socialness":           ['age', 'sport_activity', 'weekend_outdoor_time'],
  "hospitalized":         ["healthy","age","T_out","sin_month","cos_month"]
}

targets = {
  "has_kids":"classification",
  "movie_enthusiasm":"regression",
  "weekend_relax_factor":"regression",
  "hospitalized":"classification",
  "socialness": "regression"
}

output_dir = r"trained_models"
os.makedirs(output_dir, exist_ok=True)

for target, task in targets.items():
    df_work = df.copy()
    features = feature_sets[target]

    for feat in features:
        if df_work[feat].dtype == object:
            le_feat = LabelEncoder()
            df_work[feat] = le_feat.fit_transform(df_work[feat])

    if task == 'classification':
        df_work[target] = LabelEncoder().fit_transform(df_work[target])

    if task == 'classification':
        pool, metrics, primary_metric = CLASSIFIERS, clf_metrics, "accuracy"
    else:
        pool, metrics, primary_metric = REGRESSORS, reg_metrics, "r2"

    train_and_evaluate_models(
        df=df_work,
        feature_cols=features,
        target_col=target,
        task=task,
        model_pool=pool,
        metrics=metrics,
        primary_metric=primary_metric,
        output_dir=output_dir,
        output_prefix=target
    )
