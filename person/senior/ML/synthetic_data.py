import numpy as np
import pandas as pd


n = 30000
rng = np.random.RandomState(42)

# Функция для генерации синтетических значений
def seasonal_temperature(month):
    base_temp = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20,
                 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    return [rng.normal(base_temp[m], 3) for m in month]

month = rng.randint(1, 13, size=n)
day_of_week = rng.randint(0, 7, size=n)

df = pd.DataFrame({
    "month": month,
    "day_of_week": day_of_week,
    'T_out': seasonal_temperature(month),
    'holiday_flag': (day_of_week >= 5).astype(int)
})

# Хронические заболевания (диабет, гипертония, сердечно-сосудистые)
diseases = ['diabetes', 'hypertension', 'cvd']
for d in diseases:
    # вероятность болезни зависит от возраста косвенно (старше — чаще)
    df[d] = (rng.rand(n) < 0.2).astype(int)

# Приём ключевых препаратов как бинарный индикатор
# если есть хроническая болезнь, вероятность приёма лекарства высокая
for d in diseases:
    df[f'{d}_meds'] = np.where(df[d] == 1,
                                (rng.rand(n) < 0.8).astype(int),
                                (rng.rand(n) < 0.1).astype(int))

# Социальная изоляция и поддержка
# living_alone: 30% живут одни
df['living_alone'] = (rng.rand(n) < 0.3).astype(int)
# has_caregiver: если живёт один, 40% вероятность опекуна
df['has_caregiver'] = np.where(df['living_alone'] == 1,
                                (rng.rand(n) < 0.4).astype(int),
                                1)

# Финансовая устойчивость: доход и расходы в рублях
# monthly_income в рублях (средняя пенсия 25000, стд 8000)
df['monthly_income'] = rng.normal(25000, 8000, size=n).clip(8000, 80000)
# monthly_expenses ЖКХ и медицинские — зависят от дохода
df['monthly_expenses'] = (df['monthly_income'] * rng.uniform(0.4, 0.7, size=n) +
                          rng.normal(3000, 1500, size=n)).clip(5000, None)
# debt_to_income_ratio
df['debt_to_income_ratio'] = (df['monthly_expenses'] / df['monthly_income']).round(2)

# Вероятность ухудшения здоровья
base_prob = np.where(df['T_out'] < 0, 0.1, 0.02)
med_factor = np.where(df[[f'{d}_meds' for d in diseases]].sum(axis=1) > 0, 0.5, 1.0)
fin_factor = np.where(df['debt_to_income_ratio'] > 0.6, 1.3, 1.0)
df['health_risk'] = (base_prob * med_factor * fin_factor).clip(0, 1)
# Событие ухудшения здоровья за сутки
df['health_event'] = rng.binomial(1, df['health_risk'], size=n)
# Госпитализация: 7% при ухудшении, иначе 1%
df['hospitalized'] = rng.binomial(1, np.where(df['health_event'] == 1, 0.07, 0.01), size=n)

# Мобильность — уровень 0–3
def mobility_level(row):
    if row['hospitalized'] == 1 or (row['living_alone'] == 1 and row['has_caregiver'] == 0):
        return np.random.choice([0,1], p=[0.7,0.3])
    if row['T_out'] < 0:
        return np.random.choice([0,1,2], p=[0.3,0.5,0.2])
    return np.random.choice([1,2,3], p=[0.4,0.3,0.3])

df['mobility_level'] = df.apply(mobility_level, axis=1)

# Тарифная нагрузка — изменение тарифа в %
seasonal_tariff = df['month'].map({
    1: 10, 2: 9, 3: 7, 4: 5, 5: 4, 6: 5,
    7: 5, 8: 6, 9: 5, 10: 7, 11: 9, 12: 11
})
df['tariff_change'] = (seasonal_tariff + rng.normal(0, 1.5, size=n)).clip(0, None)

cols = [
    'month', 'day_of_week', 'T_out', 'holiday_flag',
    'diabetes', 'hypertension', 'cvd',
    'diabetes_meds', 'hypertension_meds', 'cvd_meds',
    'living_alone', 'has_caregiver',
    'monthly_income', 'monthly_expenses', 'debt_to_income_ratio',
    'health_risk', 'health_event', 'hospitalized',
    'mobility_level', 'tariff_change'
]
df = df[cols]

df.to_csv('synthetic_senior_features.csv', index=False)
print(f"synthetic_senior_features.csv сохранён, строк: {len(df)}")
