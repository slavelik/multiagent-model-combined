import pandas as pd
import numpy as np


def generate_hourly_features(
    n_days: int,
    hours_per_day: int,
    start: str = "2021-01-01",
    seed: int = 42,
    output_path: str = "hourly_features.csv"
):
    """
    Генерация глобального временного ряда для симуляции.
    """
    rng = np.random.RandomState(seed)

    periods = n_days * hours_per_day
    dates = pd.date_range(start=start, periods=periods, freq="h")
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["holiday_flag"] = (df["day_of_week"] >= 5).astype(int)

    base_temp = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20,
                 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    df["T_season"] = df["month"].map(base_temp)
    df["T_diurnal"] = 5 * np.sin((df["hour"] - 6) / 24 * 2 * np.pi)
    df["T_out"] = df["T_season"] + df["T_diurnal"] + rng.normal(0, 1.5, size=periods)
    df.drop(columns=["T_season", "T_diurnal"], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"{output_path} создано: {len(df)} строк")
    return df
