import os
import pickle
import pandas as pd
import numpy as np


def generate_person_features(
    n_days: int,
    hours_per_day: int,
    num_persons: int,
    seed: int = 42,
    output_path: str = "person_test_features.csv"
):
    """
    Генерация набора признаков для тестирования обученных моделей агента Person.
    Файл содержит по каждой временной точке и каждому агенту:
      - Временные признаки: datetime, month, sin_month, cos_month, T_out,
      - Демографические признаки: age, gender, income_level, education_level, region
      - Статические поведенческие и средовые признаки: family_size, has_kids, socialness,
        evening_activity_duration, tv_time, cooking_time, commute_duration,
        weekend_outdoor_time, weekend_relax_factor, healthy, hospitalized
    Эти признаки используются для прогнозирования параметров агента.
    """
    rng = np.random.RandomState(seed)
    # 1) Временные признаки
    n = n_days * hours_per_day
    dates = pd.date_range(start="2021-01-01", periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["month"] = df_time["datetime"].dt.month
    df_time["sin_month"] = np.sin(2 * np.pi * (df_time["month"] - 1) / 12)
    df_time["cos_month"] = np.cos(2 * np.pi * (df_time["month"] - 1) / 12)
    month_base = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20, 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    df_time["T_out"] = (
        df_time["month"].map(month_base)
        + 5 * df_time["sin_month"]
        + rng.normal(0, 1.5, size=n)
    )

    # 2) Демографические и статические поведенческие признаки
    agent_ids = np.arange(num_persons)
    ages = rng.randint(24, 66, size=num_persons)
    genders = rng.choice(["male", "female"], size=num_persons)
    income_levels = rng.choice(["low", "medium", "high"], size=num_persons, p=[0.3, 0.5, 0.2])
    education_levels = rng.choice(
        ["high_school", "bachelor", "master", "phd"],
        size=num_persons,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    regions = rng.choice(["urban", "suburban", "rural"], size=num_persons, p=[0.5, 0.3, 0.2])
    df_agents = pd.DataFrame({
        "agent_id": agent_ids,
        "age": ages,
        "gender": genders,
        "income_level": income_levels,
        "education_level": education_levels,
        "region": regions,
    })

    # occupation зависит от возрастной группы и коррелирует с доходом и регионом
    def sample_occ(row):
        age, inc, reg = row["age"], row["income_level"], row["region"]
        if age > 60:
            return rng.choice(["retired","consultant"], p=[0.85, 0.15])
        p_off = 0.4 + (inc=="high") * 0.2 + (reg=="urban") * 0.1
        p_rem = 0.3 + (inc!="low") * 0.1
        probs = np.clip([p_off,p_rem,1 - p_off - p_rem], 0, None)
        probs /= probs.sum()
        return rng.choice(["office","remote","shift"], p=probs)
    df_agents["occupation"] = df_agents.apply(sample_occ, axis=1)
    
    # family_size и has_kids (в среднем 1–3)
    base_fs = (df_agents['age'] - 24) / 42 \
            + (df_agents['region'] == 'rural') * 0.7 \
            + (df_agents['income_level'] == 'high') * 0.5
    lam = np.clip(base_fs * 3.0, 1, 8)
    df_agents['family_size'] = rng.poisson(lam).clip(1, 6)
    df_agents['has_kids'] = ((df_agents['family_size'] + rng.normal(0, 0.8, num_persons)) > 2.5).astype(int)
    # socialness
    base_soc = 0.5 + (df_agents['region']=='urban') * 0.1 - (df_agents['age'] - 24) / 100
    df_agents['socialness'] = np.clip(rng.normal(base_soc, 0.1, num_persons), 0, 1)

    df_agents['evening_activity_duration'] = np.clip(rng.normal(2, 0.7, num_persons), 0.5, 5)
    df_agents['sleep_start_hour'] = np.clip(rng.normal(23, 1.5, num_persons), 20, 2)
    df_agents['tv_time'] = np.clip(rng.normal(1.5, 0.7,num_persons), 0, 5)
    df_agents['cooking_time'] = np.clip(rng.normal(1,0.4,num_persons), 0.2, 3)
    df_agents['commute_duration'] = np.where(
        df_agents['occupation'].isin(['office','shift']),
        np.clip(rng.normal(0.7, 0.3, num_persons), 0.1, 2), 0)
    # weekend_relax_factor
    df_agents['weekend_relax_factor'] = (0.5 + df_agents['socialness'] * 1.2 - (df_agents['family_size'] - 1) * 0.05 + rng.normal(0, 0.1,num_persons)).clip(0.8, 2.5)

    # movie_enthusiasm
    df_agents['movie_enthusiasm'] = np.clip(
        0.2 + df_agents['socialness'] * 0.4 + df_agents['tv_time'] / 5 * 0.2
        + df_agents['evening_activity_duration'] / 5 * 0.2 + rng.normal(0, 0.05, num_persons),
        0, 1
    )

    # healthy и hospitalized
    mean_T_out = df_time['T_out'].mean()
    base_health_prob = (0.8 - (df_agents['age'] - 24) / 100 + df_agents['socialness'] * 0.2 - (mean_T_out < 0) * 0.1).clip(0.05, 0.95)
    df_agents['healthy'] = rng.binomial(1, base_health_prob)
    hosp_prob = (0.01 + (1 - df_agents['healthy']) * 0.4 + (df_agents['age'] > 55) * 0.1).clip(0, 1)
    df_agents['hospitalized'] = rng.binomial(1, hosp_prob)

    # sport_activity и weekend_outdoor_time
    df_agents['sport_activity'] = np.clip(rng.normal(df_agents['healthy'] * 0.6 + 0.3 - (df_agents['age'] - 24) / 150 + (mean_T_out > 20) * 0.1 - (mean_T_out < 20) * 0.1, 0.1, size=num_persons), 0, 1)
    df_agents['weekend_outdoor_time'] = np.clip(rng.normal(df_agents['socialness'] * 2 + df_agents['weekend_relax_factor'], 0.5), 0, 6)
    

    # 3) Кросс-джойн
    df_time['_tmp'] = 1
    df_agents['_tmp'] = 1
    df_full = df_time.merge(df_agents, on='_tmp').drop(columns=['_tmp'])

    # 4) Сохраняем
    cols = [
        'agent_id','datetime','hour', 'month', 'age','gender','income_level','education_level','region',
        'occupation', 'family_size','has_kids','socialness', 'sport_activity', 'evening_activity_duration', 'sleep_start_hour',
        'tv_time', 'cooking_time','commute_duration','weekend_outdoor_time','weekend_relax_factor', 'healthy','hospitalized', 'movie_enthusiasm'
    ]
    df_full[cols].to_csv(output_path, index=False)
    print(f"{output_path} создано: {len(df_full)} строк")
