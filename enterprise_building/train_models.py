"""
train_models.py

Скрипт для загрузки датасета по энергопотреблению предприятия
инженерии признаков для почасовой симуляции,
обучения нескольких моделей регрессии,
сравнения их через графики,
выбора лучшей по RMSE и сохранения её на диск по папкам.
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def load_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Парсинг даты
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    # Агрегация по часу
    agg = {
        'Usage_kWh': 'first',
        'Motor_and_Transformer_Load_kVarh': 'first',
        'Day_of_week': 'first',
        'WeekStatus': 'first',
        'Load_Type': 'first'
    }
    df_h = df.resample('H').agg(agg).dropna()

    # Признаки
    df_h['is_weekend'] = (df_h['WeekStatus'] != 'Weekday').astype(int)
    df_h['hour'] = df_h.index.hour
    df_h['month'] = df_h.index.month
    # Циклические кодирования времени
    df_h['hour_sin'] = np.sin(2 * np.pi * df_h['hour'] / 24)
    df_h['hour_cos'] = np.cos(2 * np.pi * df_h['hour'] / 24)

    # Базовый набор признаков
    base_features = [
        'Motor_and_Transformer_Load_kVarh',
        'is_weekend',
        'hour_sin', 'hour_cos'
    ]
    X_base = df_h[base_features]

    # One-hot кодирование Load_Type
    load_type_dummies = pd.get_dummies(df_h['Load_Type'], prefix='Load_Type', drop_first=True)

    # Финальный X с добавлением dummy-признаков
    X = pd.concat([X_base, load_type_dummies], axis=1)

    # Целевая переменная
    y = df_h['Usage_kWh']

    return X, y


def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = mean_squared_error(y_te, pred, squared=False)
    r2 = r2_score(y_te, pred)
    return {'name': name, 'model': model, 'rmse': rmse, 'r2': r2, 'pred': pred}


def plot_rmse(results, save_path: str):
    names = [r['name'] for r in results]
    vals = [r['rmse'] for r in results]
    plt.figure()
    plt.bar(names, vals)
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_scatter(best, X_te, y_te, save_path: str):
    pred = best['model'].predict(X_te)
    plt.figure()
    plt.scatter(y_te, pred, alpha=0.3)
    mn, mx = y_te.min(), y_te.max()
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel('True Usage_kWh')
    plt.ylabel('Predicted Usage_kWh')
    plt.title(f"True vs Predicted ({best['name']})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def train_enterprise_models():
    base = os.path.dirname(__file__)
    # Пути
    data_path = os.path.join(base, 'data', 'Steel_industry_data.csv')
    trained_dir = os.path.join(base, 'trained_models')
    plots_dir = os.path.join(trained_dir, 'plots')
    model_path = os.path.join(trained_dir, 'best_enterprise_model.pkl')

    # Создаем папки
    os.makedirs(plots_dir, exist_ok=True)

    # Данные
    X, y = load_preprocess(data_path)

    # Разбиение по времени (80/20)
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Определяем модели
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    # Оценка
    results = []
    for name, mdl in models.items():
        res = evaluate(name, mdl, X_tr, X_te, y_tr, y_te)
        print(f"{res['name']:20} RMSE={res['rmse']:.3f}  R2={res['r2']:.3f}")
        results.append(res)

    # Лучшая модель
    best = min(results, key=lambda x: x['rmse'])
    print(f"\nBest model: {best['name']} (RMSE={best['rmse']:.3f})")

    # Сохраняем
    with open(model_path, 'wb') as f:
        pickle.dump(best['model'], f)
    print(f"Model saved to {model_path}")

    metrics_path = os.path.join(trained_dir, 'model_metrics.txt')
    with open(metrics_path, 'w') as mf:
        for r in results:
            mf.write(f"{r['name']}: RMSE={r['rmse']:.3f}, R2={r['r2']:.3f}\n")

    # Визуализация
    rmse_plot = os.path.join(plots_dir, 'model_rmse_comparison.png')
    scatter_plot = os.path.join(plots_dir, 'best_model_scatter.png')
    plot_rmse(results, rmse_plot)
    plot_scatter(best, X_te, y_te, scatter_plot)
