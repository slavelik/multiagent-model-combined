import pandas as pd
import numpy as np

START = "2021-01-01"
def generate_person_test_features(
    n_days: int,
    num_persons: int,
    seed: int = 42,
    hours_per_day: int = 24
) -> pd.DataFrame:
    """
    Генерация набора признаков для тестирования обученных моделей агента Person.
    Возвращает DataFrame с признаками для каждого агента на каждый час симуляции:
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
    dates = pd.date_range(start=START, periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["hour"] = df_time["datetime"].dt.hour.astype(np.int8)
    df_time["month"] = df_time["datetime"].dt.month.astype(np.int8)
    df_time["sin_month"] = np.sin(2 * np.pi * (df_time["month"] - 1) / 12).astype(np.float32)
    df_time["cos_month"] = np.cos(2 * np.pi * (df_time["month"] - 1) / 12).astype(np.float32)
    month_base = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20, 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    df_time["T_out"] = (
        df_time["month"].map(month_base).astype(np.float32)
        + 5 * df_time["sin_month"]
        + rng.normal(0, 1.5, size=n).astype(np.float32)
    )

    # 2) Демографические и статические поведенческие признаки
    agent_ids = np.arange(num_persons, dtype=np.int32)
    ages = rng.randint(24, 66, size=num_persons, dtype=np.int8)
    genders = pd.Categorical(rng.choice(["male", "female"], size=num_persons))
    income_levels = pd.Categorical(rng.choice(["low", "medium", "high"], size=num_persons, p=[0.3, 0.5, 0.2]))
    education_levels = pd.Categorical(
        rng.choice(
            ["high_school", "bachelor", "master", "phd"],
            size=num_persons,
            p=[0.4, 0.3, 0.2, 0.1]
        )
    )
    regions = pd.Categorical(
        rng.choice(["urban", "suburban", "rural"], size=num_persons, p=[0.5, 0.3, 0.2])
    )
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
    df_agents["occupation"] = pd.Categorical(df_agents.apply(sample_occ, axis=1))
    
    # family_size и has_kids (в среднем 1–3)
    base_fs = (df_agents['age'] - 24) / 42 \
            + (df_agents['region'] == 'rural') * 0.7 \
            + (df_agents['income_level'] == 'high') * 0.5
    lam = np.clip(base_fs * 3.0, 1, 8)
    df_agents['family_size'] = rng.poisson(lam).clip(1, 6).astype(np.int8)
    df_agents['has_kids'] = ((df_agents['family_size'] + rng.normal(0, 0.8, num_persons)) > 2.5).astype(int)
    # socialness
    base_soc = 0.5 + (df_agents['region']=='urban') * 0.1 - (df_agents['age'] - 24) / 100
    df_agents['socialness'] = np.clip(rng.normal(base_soc, 0.1, num_persons), 0, 1).astype(np.float32)

    df_agents['evening_activity_duration'] = np.clip(rng.normal(2, 0.7, num_persons), 0.5, 5).astype(np.float32)
    df_agents['sleep_start_hour'] = np.clip(rng.normal(23, 1.5, num_persons), 20, 2).astype(np.int8)
    df_agents['tv_time'] = np.clip(rng.normal(1.5, 0.7,num_persons), 0, 5).astype(np.float32)
    df_agents['cooking_time'] = np.clip(rng.normal(1,0.4,num_persons), 0.2, 3).astype(np.float32)
    df_agents['commute_duration'] = np.where(
        df_agents['occupation'].isin(['office','shift']),
        np.clip(rng.normal(0.7, 0.3, num_persons), 0.1, 2).astype(np.float32), 0).astype(np.float32)
    # weekend_relax_factor
    df_agents['weekend_relax_factor'] = (0.5 + df_agents['socialness'] * 1.2 - (df_agents['family_size'] - 1) * 0.05 + rng.normal(0, 0.1,num_persons)).clip(0.8, 2.5).astype(np.float32)

    # movie_enthusiasm
    df_agents['movie_enthusiasm'] = np.clip(
        0.2 + df_agents['socialness'] * 0.4 + df_agents['tv_time'] / 5 * 0.2
        + df_agents['evening_activity_duration'] / 5 * 0.2 + rng.normal(0, 0.05, num_persons),
        0, 1
    ).astype(np.float32)

    # healthy и hospitalized
    mean_T_out = df_time['T_out'].mean()
    base_health_prob = (0.8 - (df_agents['age'] - 24) / 100 + df_agents['socialness'] * 0.2 - (mean_T_out < 0) * 0.1).clip(0.05, 0.95)
    df_agents['healthy'] = rng.binomial(1, base_health_prob)
    hosp_prob = (0.01 + (1 - df_agents['healthy']) * 0.4 + (df_agents['age'] > 55) * 0.1).clip(0, 1)
    df_agents['hospitalized'] = rng.binomial(1, hosp_prob)

    # sport_activity и weekend_outdoor_time
    df_agents['sport_activity'] = np.clip(rng.normal(df_agents['healthy'] * 0.6 + 0.3 - (df_agents['age'] - 24) / 150 + (mean_T_out > 20) * 0.1 - (mean_T_out < 20) * 0.1, 0.1, size=num_persons), 0, 1).astype(np.float32)
    df_agents['weekend_outdoor_time'] = np.clip(rng.normal(df_agents['socialness'] * 2 + df_agents['weekend_relax_factor'], 0.5), 0, 6).astype(np.float32)

    # 3) Кросс-джойн
    df_time['_tmp'] = 1
    df_agents['_tmp'] = 1
    df_full = df_time.merge(df_agents, on='_tmp').drop(columns=['_tmp'])

    # 4) Возвращаем DataFrame
    cols = [
        'agent_id','datetime','hour', 'month', 'age','gender','income_level','education_level','region',
        'occupation', 'family_size','has_kids','socialness', 'sport_activity', 'evening_activity_duration', 'sleep_start_hour',
        'tv_time', 'cooking_time','commute_duration','weekend_outdoor_time','weekend_relax_factor', 'healthy','hospitalized', 'movie_enthusiasm'
    ]
    return df_full[cols]

def generate_senior_test_features(
    n_days: int,
    num_seniors: int,
    seed: int = 42,
    hours_per_day: int = 24
) -> pd.DataFrame:
    """
    Генерация набора признаков для тестирования моделей пенсионеров.
    Возвращает DataFrame с признаками для каждого агента на каждый час симуляции:
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
    dates = pd.date_range(start=START, periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["day_of_week"] = df_time["datetime"].dt.weekday.astype(np.int8)
    df_time["month"] = df_time["datetime"].dt.month.astype(np.int8)
    df_time["hour"] = df_time["datetime"].dt.hour.astype(np.int8)
    df_time["sin_month"] = np.sin(2 * np.pi * (df_time["month"] - 1) / 12).astype(np.float32)
    df_time["cos_month"] = np.cos(2 * np.pi * (df_time["month"] - 1) / 12).astype(np.float32)
    month_base = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20, 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    df_time["T_out"] = (
            df_time["month"].map(month_base).astype(np.float32)
            + 5 * df_time["sin_month"]
            + rng.normal(0, 1.5, size=n).astype(np.float32)
    )
    df_time.drop(columns=["sin_month", "cos_month", "hour"], inplace=True)

    # 2) Статические признаки агентов
    agent_ids = np.arange(num_seniors, dtype=np.uint16)
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
    df_agents["has_caregiver"] = np.where(df_agents["living_alone"] == 1, (rng.rand(num_seniors) < 0.4).astype(int), 1)
    # Финансы
    df_agents["monthly_income"] = rng.normal(25000, 8000, size=num_seniors).clip(8000, 80000).astype(np.float32)
    df_agents["monthly_expenses"] = (
        df_agents["monthly_income"] * rng.uniform(0.4, 0.7, size=num_seniors) +
        rng.normal(3000, 1500, size=num_seniors)
    ).clip(5000, None).astype(np.float32)
    df_agents["debt_to_income_ratio"] = (df_agents["monthly_expenses"] / df_agents["monthly_income"]).round(2).astype(np.float32)

    # 3) Кросс-джоин
    df_time["_tmp"] = 1
    df_agents["_tmp"] = 1
    df_full = df_time.merge(df_agents, on="_tmp").drop(columns=["_tmp"])

    cols = [
        "datetime", "agent_id", "day_of_week", "month", "T_out",
        "monthly_income", "monthly_expenses", "debt_to_income_ratio",
        *diseases,
        *[f"{d}_meds" for d in diseases],
        "living_alone", "has_caregiver"
    ]
    return df_full[cols]

def generate_student_test_features(
    n_days: int,
    num_students: int,
    seed: int = 42,
    hours_per_day: int = 24
) -> pd.DataFrame:
    """
    Генерация набора признаков для тестирования обученных моделей студентов.
    Возвращает DataFrame с признаками для каждого агента на каждый час симуляции:
      - datetime, hour, day_of_week, month, holiday_flag, T_out, sin_month, cos_month, exam_period
      - course, on_campus, diligence, sleep_duration
    Эти признаки будут использоваться для прогнозирования:
      hw_duration, sleep_start_hour, leisure_duration,
      commute_duration, evening_study_duration, healthy.
    """
    rng = np.random.RandomState(seed)
    # 1) Формируем временной ряд глобальных признаков
    n = n_days * hours_per_day
    dates = pd.date_range(start=START, periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["hour"] = df_time["datetime"].dt.hour.astype(np.int8)
    df_time["month"] = df_time["datetime"].dt.month.astype(np.int8)
    df_time["exam_period"] = ((df_time["month"] % 6).isin([5,0])).astype(int)

    # 2) Формируем статические признаки для агентов
    agent_ids = np.arange(num_students, dtype=np.int16)
    courses = rng.randint(1, 5, size=num_students).astype(np.int8)
    diligences = rng.normal(1.0, 0.1, size=num_students).astype(np.float32)  # среднее 1.0, σ=0.1
    base_prob = 1 - (courses - 1) * 0.1
    noisy_prob = (base_prob * 0.95) + 0.05 * rng.rand(num_students)
    on_campus_flags = (rng.rand(num_students) < noisy_prob).astype(np.bool)
    sleep_dur = rng.normal(7 - 0.2*(courses - 1), 1.0, size=num_students).clip(4,10).astype(np.float32)
    
    df_agents = pd.DataFrame({
        "agent_id": agent_ids,
        "course": courses,
        "diligence": diligences,
        "on_campus": on_campus_flags,
        "sleep_duration": sleep_dur
    })

    # 3) Кросс-джойн: каждая временная точка * каждый агент
    df_time["_tmpkey"] = 1
    df_agents["_tmpkey"] = 1
    df_full = df_time.merge(df_agents, on="_tmpkey").drop(columns=["_tmpkey"])

    df_full["hospitalized"] = (rng.rand(len(df_full)) < 0.005).astype(np.bool)
    # 4) Возвращаем DataFrame с нужными колонками
    features = [
        "exam_period", "course", "on_campus", "diligence", "sleep_duration", "hospitalized",
    ]
    return df_full[["agent_id", "datetime"] + features]

def generate_ev_test_features(
    n_days: int,
    num_evs: int,
    seed: int = 42,
    hours_per_day: int = 24
) -> pd.DataFrame:
    """
    Генерация набора признаков для тестирования моделей агента ElectricCarAgent.
    Возвращает DataFrame с признаками для каждого агента на каждый час симуляции:
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
    dates = pd.date_range(start=START, periods=total_hours, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["hour"] = df_time["datetime"].dt.hour.values.astype(np.int8)
    df_time["day_of_week"] = df_time["datetime"].dt.weekday.values.astype(np.int8)
    df_time["month"] = df_time["datetime"].dt.month.values.astype(np.int8)
    df_time["holiday"] = (df_time["day_of_week"] >= 5).astype(bool)

    agent_ids = np.arange(num_evs, dtype=np.int16)
    capacities = rng.uniform(40e3,80e3,num_evs).astype(np.float32)
    initial_charges = capacities * rng.uniform(0.2,0.8,num_evs).astype(np.float32)
    efficiencies = rng.uniform(150,250,num_evs).astype(np.float32)
    preferences = pd.Categorical(
        rng.choice(["overnight","midday","fast"], num_evs, p=[0.5,0.3,0.2])
    )
    df_agents = pd.DataFrame({
        "agent_id": agent_ids,
        "battery_capacity_wh": capacities,
        "efficiency_wh_per_km": efficiencies,
        "charging_preference": preferences
    })

    df_full = df_time.merge(df_agents, how="cross")

    # 4) Векторная планировка пробега, создаем матрицу shape=(n_days, hours_per_day, num_evs)
    hour_weights = np.array(
        [0.1,0.1,0.1,0.1,0.1,0.2,0.5,1.5,2.0,1.5,1.0,0.8,
         0.8,0.8,0.8,1.0,1.2,2.0,1.5,1.2,0.8,0.5,0.3,0.2],
        dtype=np.float32
    )
    # матрица дневных дистанций
    daily_dist = rng.uniform(20,150,(n_days, num_evs)).astype(np.float32)
    # если выходной, умножаем на 0.7
    is_holiday = df_time["holiday"].values.reshape(n_days, hours_per_day)
    daily_dist = daily_dist[:, None, :] * np.where(is_holiday[:,:,None], 0.7, 1.0)
    # нормируем по сумме весов каждого дня
    w_sum = hour_weights.sum()
    # распределяем по часам
    planned = (daily_dist * (hour_weights[None,:,None] / w_sum)).reshape(-1, num_evs)
    # кладем в df_full одной операцией
    df_full["planned_distance_km"] = planned.repeat(1).flatten()

    # 5) Текущий заряд
    # создаем cumulative factor для каждого дня-агента
    # first day — initial, потом каждый день умножаем на случайный [0.5…1.0]
    charge = np.empty_like(planned)
    charge[0] = initial_charges
    for d in range(1, n_days):
        # умножаем предыдущий день на случайный вектор
        charge[d] = charge[d-1] * rng.uniform(0.5,1.0,size=num_evs).astype(np.float32)
    df_full["current_charge_wh"] = charge.reshape(-1)

    return df_full[[
        "agent_id","datetime","hour","day_of_week","month","holiday",
        "battery_capacity_wh","current_charge_wh","planned_distance_km",
        "efficiency_wh_per_km","charging_preference"
    ]]