import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def train_and_evaluate_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: str,  # "classification" или "regression"
    model_pool: dict[str, callable],
    metrics: dict[str, callable],
    primary_metric: str,  # оставил для совместимости
    output_dir: str,
    output_prefix: str  # оставил для совместимости
):
    """
    Обучает и оценивает несколько моделей, сохраняет:
      - Для каждого target_col свою папку в output_dir
      - В ней: все .pkl моделей и один CSV с метриками всех моделей
    """
    # Общая папка для данного таргета
    var_dir = os.path.join(output_dir, target_col)
    os.makedirs(var_dir, exist_ok=True)

    # Разбиваем выборку
    X = df[feature_cols]
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y if task=='classification' else None)
    )

    results = []

    print(f"Обучаем модели для {target_col}...")
    # Обучаем и сохраняем каждую модель
    for name, constructor in model_pool.items():
        model = constructor()
        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)

            # получаем score для ROC/PR, если есть
            if task == 'classification':
                if hasattr(model, 'predict_proba'):
                    y_score_test = model.predict_proba(x_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score_test = model.decision_function(x_test)
                else:
                    y_score_test = None
                # на train
                if hasattr(model, 'predict_proba'):
                    y_score_train = model.predict_proba(x_train)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score_train = model.decision_function(x_train)
                else:
                    y_score_train = None
            else:
                y_score_test = None
                y_score_train = None

            # Собираем метрики: тест и train
            row = {"model": name}
            for m_name, m_fn in metrics.items():
                # тест
                try:
                    if m_name in ('roc_auc', 'pr_auc'):
                        row[f"{m_name}_test"] = m_fn(y_test, y_pred, y_score_test)
                    else:
                        row[f"{m_name}_test"] = m_fn(y_test, y_pred)
                except Exception:
                    row[f"{m_name}_test"] = np.nan
                # train
                try:
                    if m_name in ('roc_auc', 'pr_auc'):
                        row[f"{m_name}_train"] = m_fn(y_train, y_train_pred, y_score_train)
                    else:
                        row[f"{m_name}_train"] = m_fn(y_train, y_train_pred)
                except Exception:
                    row[f"{m_name}_train"] = np.nan

            results.append(row)

            # Сохраняем эту модель
            pkl_path = os.path.join(var_dir, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Модель {name} сохранена в {pkl_path}")

        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")

    # Сохраняем CSV с результатами
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(var_dir, "results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Все метрики сохранены в {csv_path}")
    print("------------------------------------------------------------------------------")

    return df_res
