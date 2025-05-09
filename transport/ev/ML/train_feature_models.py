import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
from train_models import train_and_evaluate_models

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

df = pd.read_csv("synthetic_ev_features.csv")

state_map = {'idle': 0, 'driving': 1, 'charging': 2}
df['state_label'] = df['state'].map(state_map)

# One-hot для charging_preference
df_enc = pd.get_dummies(df, columns=['charging_preference'], drop_first=False)

feature_cols = [
    'month',
    'holiday',
    'hour',
    'battery_capacity_wh',
    'current_charge_wh',
    'planned_distance_km',
    'efficiency_wh_per_km'
]

# дамми-признаки
pref_cols = [c for c in df_enc.columns if c.startswith('charging_preference_')]
feature_cols += pref_cols

CLASSIFIERS = {
    'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': lambda: LogisticRegression(max_iter=5000),
    'XGBoost': lambda: xgb.XGBClassifier(eval_metric='mlogloss')
}

clf_metrics = {
    'accuracy': accuracy_score,
    'precision': lambda y, yhat: precision_score(y, yhat, average='macro', zero_division=0),
    'recall':    lambda y, yhat: recall_score(y, yhat, average='macro', zero_division=0),
    'f1':        lambda y, yhat: f1_score(y, yhat, average='macro', zero_division=0)
}

output_root = 'trained_models'
for target in ['state_label']:
    train_and_evaluate_models(
        df=df_enc,
        feature_cols=feature_cols,
        target_col=target,
        task='classification',
        model_pool=CLASSIFIERS,
        metrics=clf_metrics,
        primary_metric='accuracy',
        output_dir=output_root,
        output_prefix=target
    )

