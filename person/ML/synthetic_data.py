import pandas as pd
import numpy as np


def generate_synthetic_person_params(
    n_agents: int = 1000,
    seed: int = 42,
    output_path: str = "synthetic_person_features.csv"
):
    """
    Генерация синтетического датасета для обучения моделей параметров PersonAgent.
    Внесены корреляции, чтобы увеличить предсказуемость:
      - occupation зависит от возраста, дохода и региона
      - family_size зависит от возраста, региона и income_level
      - has_kids шумно от family_size и возраста
      - movie_enthusiasm зависит от occupation и socialness
      - socialness зависит от region и age
      - weekend_relax_factor зависит от socialness и occupation
      - healthy зависит от age и socialness
      - hospitalized зависит от healthy и age
    Добавлены временные признаки: evening_activity_duration, sleep_start_hour, commute_duration, tv_time, cooking_time, sport_activity, weekend_outdoor_time,
    datetime, month, sin_month, cos_month, T_out
    """
    rng = np.random.RandomState(seed)

    dates = pd.date_range(start="2021-01-01", periods=n_agents, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["month"] = df_time["datetime"].dt.month
    df_time["sin_month"] = np.sin(2 * np.pi * (df_time["month"] - 1) / 12)
    df_time["cos_month"] = np.cos(2 * np.pi * (df_time["month"] - 1) / 12)
    # T_out по сезонной базовой + шум + синус/косинус
    month_base = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20, 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    df_time["T_out"] = (df_time["month"].map(month_base) + 5 * df_time["sin_month"] + rng.normal(0, 1.5, size=n_agents))

    # Демографические и персональные признаки
    ages = rng.randint(24, 66, size=n_agents)
    genders = rng.choice(["male", "female"], size=n_agents)
    income_levels = rng.choice(["low", "medium", "high"], size=n_agents, p=[0.3, 0.5, 0.2])
    education_levels = rng.choice(
        ["high_school", "bachelor", "master", "phd"],
        size=n_agents,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    regions = rng.choice(["urban", "suburban", "rural"], size=n_agents, p=[0.5, 0.3, 0.2])

    df_demo = pd.DataFrame({
        "age": ages,
        "gender": genders,
        "income_level": income_levels,
        "education_level": education_levels,
        "region": regions,
    })

    df = pd.concat([df_time, df_demo], axis=1)
    # Occupation зависимо от возрастной группы и коррелирует с доходом и регионом
    def sample_occupation(row):
        age, inc, reg = row["age"], row["income_level"], row["region"]
        if age > 60:
            return rng.choice(["retired", "consultant"], p=[0.85, 0.15])
        p_office = 0.4 + (inc == "high")*0.2 + (reg == 'urban')*0.1
        p_remote = 0.3 + (inc != "low")*0.1
        total = p_office + p_remote
        probs = np.array([p_office, p_remote,1 - p_office - p_remote])
        probs = np.clip(probs, 0, None)
        probs = probs/probs.sum()  # нормировка на случай, если total > 1
        return rng.choice(['office','remote','shift'], p=probs)
    df['occupation'] = df.apply(sample_occupation, axis=1)

    # Размер семьи коррелирует с возрастом, регионом и доходом
    base = (df['age'] - 24) / 42  # нормировка на 42 года
    base += (df['region'] == 'rural') * 0.7  
    base += (df['income_level'] == 'high') * 0.5
    # шум для более реалистичного разброса
    lam = np.clip(base * 3.0, 1, 8)
    df['family_size'] = rng.poisson(lam).clip(1, 8)
    df['has_kids'] = ((df['family_size'] + rng.normal(0, 0.8, n_agents)) > 2.5).astype(int)

    # Временные/поведенческие признаки
    df['evening_activity_duration'] = np.clip(rng.normal(2, 0.7, size=n_agents), 0.5, 5)
    df['sleep_start_hour'] = np.clip(rng.normal(23, 1.5, size=n_agents),20,2)
    df['commute_duration'] = np.where(
        df['occupation'].isin(['office','shift']),
        np.clip(rng.normal(0.7, 0.3, size=n_agents), 0.1, 2),
        0
    )
    df['tv_time'] = np.clip(rng.normal(1.5, 0.7, size=n_agents), 0, 5)
    df['cooking_time'] = np.clip(rng.normal(1,0.4,size=n_agents), 0.2, 3)

    # socialness: зависит от региона и возраста
    base_social = 0.5 + (df['region'] == 'urban') * 0.1 - (df['age'] - 24) / 100
    df['socialness'] = np.clip(rng.normal(base_social, 0.1, size=n_agents), 0, 1)

    # movie enthusiasm: зависит от tv, leisure_dur и socialness
    df['movie_enthusiasm'] = (0.2 + df['socialness'] * 0.4 + df['tv_time'] / 5 * 0.2 + df['evening_activity_duration'] / 5 * 0.2 + rng.normal(0, 0.05, n_agents)).clip(0,1)

    # weekend_relax_factor: линейно зависит от socialness и occupation + family size
    df['weekend_relax_factor'] = (0.5 + df['socialness'] * 1.2 + (df['occupation'] == 'retired') * 0.3 - (df['family_size']-1) * 0.05 + rng.normal(0, 0.1, n_agents)).clip(0.8, 2.5)

    
    sport_loc = (
        0.5                                       # базовый уровень
        - (df['age'] - 24) / 150                  # с возрастом чуть меньше активности
        + np.where(df['T_out'] > 20, 0.1, -0.1)   # жаркая погода — выше активность, холод — ниже
        + df['socialness'] * 0.2                  # социальность подстегивает спорт
    )
    df['sport_activity'] = np.clip(
        rng.normal(loc=sport_loc, scale=0.2, size=n_agents),
        0, 1
    )

    # healthy: теперь зависит от возраста, socialness и sport_activity, с шумом
    base_health_prob = (
        0.8
        - (df['age'] - 24) / 120
        + df['socialness'] * 0.1
        + df['sport_activity'] * 0.4
        - np.where(df['T_out'] < 0, 0.1, 0.0)
    ).clip(0.05, 0.95)
    df['healthy'] = rng.binomial(1, base_health_prob)

    # # healthy: зависит от возраста и socialness
    # base_health_prob = (0.8 - (df['age'] - 24) / 100   # со старением вероятность падает
    #     + df['socialness'] * 0.2   # больше socialness — выше шанс здоровья
    #     - np.where(df['T_out'] < 0, 0.1, 0.0)  # холод повышает риск
    # ).clip(0.05, 0.95)
    # df['healthy'] = rng.binomial(1, base_health_prob)

    # hospitalized: если не здоров — выше риск, плюс старше 55 ещё выше
    hosp_prob = (0.01 + (1 - df['healthy']) * 0.4 + (df['age'] > 55) * 0.1).clip(0, 1)
    df['hospitalized'] = rng.binomial(1, hosp_prob)

    # df['sport_activity'] = np.clip(rng.normal(loc = df['healthy'] * 0.6 + 0.3 - (df['age'] - 24) / 150 + np.where(df['T_out'] > 20, 0.1, -0.1), scale = 0.1, size = n_agents), 0, 1)
    # weekend_outdoor_time: зависит от socialness и weekend_relax
    df['weekend_outdoor_time'] = np.clip(rng.normal(df['socialness'] * 2 + df['weekend_relax_factor'], 0.5), 0, 6)

    # Сохраняем
    df.to_csv(output_path, index=False)
    print(f"{output_path} создано: {len(df)} строк")

n_strs = 30000
generate_synthetic_person_params(
    n_agents=n_strs,
    seed=42,
    output_path="synthetic_person_features.csv"
)
