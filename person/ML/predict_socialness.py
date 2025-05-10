import os
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

data = pd.read_csv("synthetic_person_features.csv")

target_col = 'socialness'
feature_cols = ["age", "sport_activity", "weekend_outdoor_time", "region", "has_kids"]
# , "activity_duration"
X = data[feature_cols]
y_raw = data[target_col]

target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]
preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('ohe', OneHotEncoder(sparse_output=False, drop='first'), cat_cols)
])

pipeline = make_pipeline(
    preprocessor,
    xgb.XGBRegressor(random_state=42, enable_categorical=False)
)

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

metrics = {
    'model': 'XGBoost',
    'r2_train': r2_score(y_train, y_train_pred),
    'r2_test': r2_score(y_test, y_test_pred),
    'mae_train': mean_absolute_error(y_train, y_train_pred),
    'mae_test': mean_absolute_error(y_test, y_test_pred),
    'mse_train': mean_squared_error(y_train, y_train_pred),
    'mse_test': mean_squared_error(y_test, y_test_pred)
}
metrics_df = pd.DataFrame([metrics])
metrics_csv = r'trained_models\socialness\socialness_metrics.csv'
os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
if os.path.exists(metrics_csv):
    metrics_df.to_csv(metrics_csv, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_csv, index=False)
print(metrics_df)

with open('trained_models/socialness/socialness_pipeline.pkl', 'wb+') as f:
    pickle.dump(pipeline, f)
with open('trained_models/socialness/socialness_encoder.pkl', 'wb+') as f:
    pickle.dump(target_le, f)

model = pipeline.named_steps['xgbregressor']
importances = model.feature_importances_
features = pipeline.named_steps['columntransformer'].get_feature_names_out()

plt.figure(figsize=(8,6))
plt.barh(features, importances)
plt.xlabel('Важность признака')
plt.ylabel('Признаки')
plt.title('Важность признаков для регрессии socialness')
plt.tight_layout()
plt.savefig('feature_importance_socialness.png')
plt.show()
