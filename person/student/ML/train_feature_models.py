import warnings
import pandas as pd
import os
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
import optuna
from sklearn.model_selection import cross_val_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

from train_models import train_and_evaluate_models

df = pd.read_csv("synthetic_student_features.csv")

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def tune_xgb_reg(X, y, n_trials=30):
    def objective(trial):
        params = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-3, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


CLASSIFIERS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "XGBoost": lambda: xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

REGRESSORS = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "ElasticNet": lambda: ElasticNet(),
    "XGBoost": lambda: xgb.XGBRegressor()
}

clf_metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y, yhat: precision_score(y, yhat, zero_division=0),
    "recall":    lambda y, yhat: recall_score(y, yhat, zero_division=0),
    "f1":        lambda y, yhat: f1_score(y, yhat, zero_division=0)
}

reg_metrics = {
    "r2":  r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error
}

feature_sets = {
    "hw_duration":           ["course", "on_campus", "diligence", "exam_period", "holiday_flag"],
    "sleep_start_hour":      ["on_campus", "course", "exam_period", "holiday_flag"],
    "leisure_duration":      ["on_campus", "holiday_flag", "exam_period", "diligence"],
    "commute_duration":      ["on_campus", "course"],
    "evening_study_duration":["course", "exam_period", "diligence"],
    "healthy":               ["hw_duration", "evening_study_duration", "sleep_start_hour", "sleep_duration", "T_out"],
}
targets = {
    "hw_duration": "regression",
    "sleep_start_hour": "regression",
    "leisure_duration": "regression",
    "commute_duration": "regression",
    "evening_study_duration": "regression",
    "healthy": "classification",
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
        X = df_work[features]
        y = df_work[target]
        best_params = tune_xgb_reg(X, y, n_trials=30)
        REGRESSORS['XGBoost'] = lambda bp=best_params: xgb.XGBRegressor(**bp)
        pool, metrics, primary_metric = REGRESSORS, reg_metrics, 'r2'

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
