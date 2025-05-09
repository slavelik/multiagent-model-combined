import pandas as pd
import numpy as np


def generate_ev_test_features(
    n_days: int,
    hours_per_day: int,
    num_evs: int,
    seed: int = 42,
    output_path: str = "ev_test_features.csv"
):
    """
    Генерация набора признаков для тестирования моделей агента ElectricCarAgent.
    Файл содержит по каждой временной точке и каждому агенту:
      - datetime: метка времени
      - hour: час суток
      - day_of_week: день недели 0-6
      - month: месяц 1-12
      - holiday: выходной (сб/вс)
      - battery_capacity_wh: вместимость батареи (Wh)
      - current_charge_wh: заряд в начале часа (Wh)
      - planned_distance_km: суммарный план пробега за день (км)
      - efficiency_wh_per_km: расход батареи (Wh/км)
      - charging_preference: ['overnight', 'midday', 'fast']
    """
    rng = np.random.RandomState(seed)
    
    total_hours = n_days * hours_per_day
    dates = pd.date_range(start="2021-01-01", periods=total_hours, freq="h")
    df_time = pd.DataFrame({
        "datetime": dates
    })
    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["day_of_week"] = df_time["datetime"].dt.weekday
    df_time["month"] = df_time["datetime"].dt.month
    df_time["holiday"] = (df_time["day_of_week"] >= 5).astype(int)

    # Почасовые веса для planned_distance_km (пики в 7:00–9:00 и 17:00–19:00)
    hour_weights = {
        0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2, 6: 0.5,
        7: 1.5, 8: 2.0, 9: 1.5, 10: 1.0, 11: 0.8, 12: 0.8, 13: 0.8,
        14: 0.8, 15: 1.0, 16: 1.2, 17: 2.0, 18: 1.5, 19: 1.2, 20: 0.8,
        21: 0.5, 22: 0.3, 23: 0.2
    }
    df_time["hour_weight"] = df_time["hour"].map(hour_weights)

    agent_ids = np.arange(num_evs)
    capacities = rng.uniform(40e3, 80e3, size=num_evs)  # Wh
    initial_charges = capacities * rng.uniform(0.2, 0.8, size=num_evs)  # 20-80% от емкости
    efficiencies = rng.uniform(150, 250, size=num_evs)  # Wh/km
    preferences = rng.choice(["overnight", "midday", "fast"], size=num_evs, p=[0.5, 0.3, 0.2])

    daily_distances = rng.uniform(20, 150, size=(n_days, num_evs))
    df_agents = pd.DataFrame({
        "agent_id": agent_ids,
        "battery_capacity_wh": capacities,
        "efficiency_wh_per_km": efficiencies,
        "charging_preference": preferences
    })

    df_time["_tmpkey"] = 1
    df_agents["_tmpkey"] = 1
    df_full = df_time.merge(df_agents, on="_tmpkey").drop(columns=["_tmpkey"])

    df_full["planned_distance_km"] = 0.0
    df_full["day_index"] = df_full["datetime"].dt.date
    df_full["daily_distance"] = 0.0
    for day in range(n_days):
        day_mask = (df_full["datetime"].dt.date == dates[day * hours_per_day].date())
        for agent in range(num_evs):
            agent_mask = (df_full["agent_id"] == agent)
            daily_dist = daily_distances[day, agent]
            if df_full.loc[day_mask, "holiday"].iloc[0] == 1:
                daily_dist *= 0.7
            hour_weights_day = df_full.loc[day_mask & agent_mask, "hour_weight"]
            total_weight = hour_weights_day.sum()
            df_full.loc[day_mask & agent_mask, "planned_distance_km"] = (
                daily_dist * hour_weights_day / total_weight
            )
            df_full.loc[day_mask & agent_mask, "daily_distance"] = daily_dist

    df_full["current_charge_wh"] = 0.0
    for day in range(n_days):
        day_mask = (df_full["datetime"].dt.date == dates[day * hours_per_day].date())
        for agent in range(num_evs):
            agent_mask = (df_full["agent_id"] == agent)
            if day == 0:
                df_full.loc[day_mask & agent_mask, "current_charge_wh"] = initial_charges[agent]
            else:
                prev_day_last_hour = df_full.loc[
                    (df_full["datetime"] == dates[(day - 1) * hours_per_day + hours_per_day - 1]) &
                    (df_full["agent_id"] == agent),
                    "current_charge_wh"
                ].iloc[0]
                df_full.loc[day_mask & agent_mask, "current_charge_wh"] = prev_day_last_hour * rng.uniform(0.5, 1.0)

    df_full = df_full.drop(columns=["hour_weight", "day_index", "daily_distance"])

    cols = [
        "agent_id", "datetime", "hour", "day_of_week", "month", "holiday",
        "battery_capacity_wh", "current_charge_wh", "planned_distance_km",
        "efficiency_wh_per_km", "charging_preference"
    ]
    df_full = df_full[cols]

    df_full.to_csv(output_path, index=False)
    print(f"{output_path} создано: {len(df_full)} строк")