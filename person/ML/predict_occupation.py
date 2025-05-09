import os
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("synthetic_person_features.csv")

target_col = 'occupation'
feature_cols = ['age','income_level','region','socialness', 'family_size', 'has_kids', 'commute_duration']
X = data[feature_cols]
y_raw = data[target_col]

target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class_counts = pd.Series(y_train).value_counts().to_dict()
max_count = max(class_counts.values())
class_weights = {cls: max_count/count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in y_train]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('ord', OneHotEncoder(sparse_output=False, drop='first'), cat_cols)
])

pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(random_state=42, enable_categorical=False)
)

pipeline.fit(X_train, y_train, xgbclassifier__sample_weight=sample_weights)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

metrics = {
    'model': 'XGB_Pipeline_Occupation',
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
metrics_csv = r'trained_models\occupation\occupation_metrics.csv'
if os.path.exists(metrics_csv):
    metrics_df.to_csv(metrics_csv, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_csv, index=False)
print(metrics_df)

with open('trained_models/occupation/occupation_pipeline.pkl', 'wb+') as f:
    pickle.dump(pipeline, f)
with open('trained_models/occupation/occupation_encoder.pkl', 'wb+') as f:
    pickle.dump(target_le, f)

model = pipeline.named_steps['xgbclassifier']
importances = model.feature_importances_
features = pipeline.named_steps['columntransformer'].get_feature_names_out()

plt.figure(figsize=(8, 6))
plt.barh(features, importances)
plt.xlabel('Важность признака')
plt.ylabel('Признаки')
plt.title('Важность признаков для классификации occupation')
plt.tight_layout()
plt.savefig('feature_importance.png')
