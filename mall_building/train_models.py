#!/usr/bin/env python3
"""
train_mall_model.py

Скрипт для загрузки датасета трафика ТРЦ (колонки: datetime, T_out, occupancy_rate, day_off),
инженерии признаков с учётом цикличности времени,
обучения нескольких моделей регрессии,
сравнения их через графики, выбора лучшей по RMSE и сохранения её на диск.
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def load_data(path: str) -> pd.DataFrame:
    """Загружает CSV, парсит datetime и добавляет циклические признаки."""
    df = pd.read_csv(path, parse_dates=['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    # циклическое кодирование
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df


def evaluate_model(name: str, model, X_tr, X_te, y_tr, y_te):
    """Тренирует модель, возвращает RMSE и R2."""
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = mean_squared_error(y_te, preds, squared=False)
    r2 = r2_score(y_te, preds)
    return {'name': name, 'model': model, 'rmse': rmse, 'r2': r2}


def plot_comparison(results, output_dir):
    """Строит и сохраняет графики сравнения моделей."""
    names = [r['name'] for r in results]
    rmses = [r['rmse'] for r in results]
    # График RMSE
    plt.figure(figsize=(8,4))
    plt.bar(names, rmses)
    plt.ylabel('RMSE')
    plt.title('Model RMSE Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join('MallBuilding', 'trained_models', 'plots', 'model_rmse_comparison.png'))
    plt.show()


def plot_best_scatter(best, X_te, y_te, output_dir):
    """Строит и сохраняет scatter plot для лучшей модели."""
    preds = best['model'].predict(X_te)
    plt.figure(figsize=(6,6))
    plt.scatter(y_te, preds, alpha=0.3)
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'k--')
    plt.xlabel('True occupancy_rate')
    plt.ylabel('Predicted occupancy_rate')
    plt.title(f"True vs Predicted ({best['name']})")
    plt.tight_layout()
    plt.savefig(os.path.join('MallBuilding', 'trained_models', 'plots', 'best_model_scatter.png'))
    plt.show()


def train_mall_models():
    # Пути
    base = os.getcwd()
    data_path = os.path.join(base, 'MallBuilding', 'data', 'mall_traffic_synthetic.csv')
    model_path = os.path.join(base, 'MallBuilding', 'trained_models', 'best_mall_model.pkl')
    # Папка для графиков
    plots_dir = 'plots'

    # Загрузка данных
    df = load_data(data_path)
    features = [
        'T_out', 'day_off',
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos'
    ]
    target = 'occupancy_rate'
    X = df[features]
    y = df[target]

    # Разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Определение моделей
    model_defs = {
        'LinearRegression':   LinearRegression(),
        'RandomForest':       RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting':   GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Обучение и оценка
    results = []
    print('Evaluating models:')
    for name, mdl in model_defs.items():
        res = evaluate_model(name, mdl, X_train, X_test, y_train, y_test)
        print(f"{res['name']:20} RMSE={res['rmse']:.3f}  R2={res['r2']:.3f}")
        results.append(res)

    # Выбор лучшей модели
    best = min(results, key=lambda x: x['rmse'])
    print(f"\nBest model: {best['name']} (RMSE={best['rmse']:.3f})")

    # Сохранение модели
    with open(model_path, 'wb') as f:
        pickle.dump(best['model'], f)
    print(f"Model saved to {model_path}")

    # Визуализация
    plot_comparison(results, plots_dir)
    plot_best_scatter(best, X_test, y_test, plots_dir)
