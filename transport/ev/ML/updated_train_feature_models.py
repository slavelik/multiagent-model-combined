import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


train_data = pd.read_csv("synthetic_ev_features.csv")
state_map = {'idle': 0, 'driving': 1, 'charging': 2}
train_data['state_label'] = train_data['state'].map(state_map)

# One-hot кодирование charging_preference
train_data_enc = pd.get_dummies(train_data, columns=['charging_preference'], drop_first=False)

feature_cols = [
    'month',
    'holiday',
    'hour',
    'battery_capacity_wh',
    'current_charge_wh',
    'planned_distance_km',
    'efficiency_wh_per_km',
    'charging_preference_fast',
    'charging_preference_midday',
    'charging_preference_overnight'
]

X = train_data_enc[feature_cols]
y = train_data_enc['state_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class_weights = {0: 1.0, 1: 36267/4663, 2: 36267/2870}
sample_weights = [class_weights[label] for label in y_train]

model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

metrics = {
    'model': 'XGBoost',
    'accuracy_train': accuracy_score(y_train, y_train_pred),
    'accuracy_test': accuracy_score(y_test, y_test_pred),
    'precision_train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
    'precision_test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
    'recall_train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
    'recall_test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
    'f1_train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
    'f1_test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
}

metrics_df = pd.DataFrame([metrics])
metrics_csv_path = 'model_metrics.csv'
if os.path.exists(metrics_csv_path):
    metrics_df.to_csv(metrics_csv_path, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_csv_path, index=False)

model_path = os.path.join('enhanced_model', 'XGBoost.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Вывод метрик
print("Метрики модели XGBoost:")
print(metrics_df)