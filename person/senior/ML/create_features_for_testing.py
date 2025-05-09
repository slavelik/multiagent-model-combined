import pandas as pd
import numpy as np


def generate_senior_test_features(
    n_days: int,
    hours_per_day: int,
    num_seniors: int,
    seed: int = 42,
    output_path: str = "senior_test_features.csv"
):
    """
    Генерация набора признаков для тестирования моделей пенсионеров.
    Файл содержит по каждой временной точке и каждому агенту:
      datetime, agent_id, day_of_week, month, T_out,
      monthly_income, monthly_expenses, debt_to_income_ratio,
      diabetes, hypertension, cvd,
      diabetes_meds, hypertension_meds, cvd_meds,
      living_alone, has_caregiver
    Это входные признаки для предсказания health_event, hospitalized, mobility_level и tariff_change.
    """
    rng = np.random.RandomState(seed)
    # 1) Временной ряд глобальных признаков
    n = n_days * hours_per_day
    dates = pd.date_range(start="2021-01-01", periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["day_of_week"] = df_time["datetime"].dt.weekday
    df_time["month"] = df_time["datetime"].dt.month

    # Сезонная + суточная наружная температура
    month_base = {1:-5,2:-3,3:2,4:8,5:15,6:20,7:22,8:20,9:15,10:8,11:2,12:-2}
    df_time["T_season"] = df_time["month"].map(month_base)
    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["T_diurnal"] = 5 * np.sin((df_time["hour"] - 6) / 24 * 2 * np.pi)
    df_time["T_out"] = (
        df_time["T_season"] + df_time["T_diurnal"] + rng.normal(0, 1.5, size=n)
    )
    df_time.drop(columns=["T_season", "T_diurnal", "hour"], inplace=True)

    # 2) Статические признаки агентов
    agent_ids = np.arange(num_seniors)
    # Хронические заболевания и лекарства
    diseases = ["diabetes", "hypertension", "cvd"]
    df_agents = pd.DataFrame({"agent_id": agent_ids})
    for d in diseases:
        df_agents[d] = (rng.rand(num_seniors) < 0.2).astype(int)
        df_agents[f"{d}_meds"] = np.where(
            df_agents[d] == 1,
            (rng.rand(num_seniors) < 0.8).astype(int),
            (rng.rand(num_seniors) < 0.1).astype(int)
        )
    # Социальная изоляция и поддержка
    df_agents["living_alone"] = (rng.rand(num_seniors) < 0.3).astype(int)
    df_agents["has_caregiver"] = np.where(
        df_agents["living_alone"] == 1,
        (rng.rand(num_seniors) < 0.4).astype(int),
        1
    )
    # Финансы
    df_agents["monthly_income"] = rng.normal(25000, 8000, size=num_seniors).clip(8000, 80000)
    df_agents["monthly_expenses"] = (
        df_agents["monthly_income"] * rng.uniform(0.4, 0.7, size=num_seniors) +
        rng.normal(3000, 1500, size=num_seniors)
    ).clip(5000, None)
    df_agents["debt_to_income_ratio"] = (
        df_agents["monthly_expenses"] / df_agents["monthly_income"]
    ).round(2)

    # 3) Кросс-джоин
    df_time["_tmp"] = 1
    df_agents["_tmp"] = 1
    df_full = df_time.merge(df_agents, on="_tmp").drop(columns=["_tmp"])

    # 4) Сохраняем только нужные колонки
    cols = [
        "datetime", "agent_id", "day_of_week", "month", "T_out",
        "monthly_income", "monthly_expenses", "debt_to_income_ratio",
        *diseases,
        *[f"{d}_meds" for d in diseases],
        "living_alone", "has_caregiver"
    ]
    df_full = df_full[cols]

    # 5) Вывод
    df_full.to_csv(output_path, index=False)
    print(f"{output_path} создано: {len(df_full)} строк")

    return df_full
