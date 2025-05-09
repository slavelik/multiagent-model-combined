import pandas as pd
import numpy as np


def generate_synthetic_student_data(n=8760, random_state=42):
    rng = np.random.RandomState(random_state)

    dates = pd.date_range(start="2021-01-01", periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["day_of_week"]  = df_time["datetime"].dt.weekday
    df_time["month"]        = df_time["datetime"].dt.month
    df_time["sin_month"]    = np.sin(2 * np.pi * (df_time["month"] - 1) / 12)
    df_time["cos_month"]    = np.cos(2 * np.pi * (df_time["month"] - 1) / 12)
    # T_out по сезонной базовой + шум + синус/косинус
    month_base = {1: -5, 2: -3, 3: 2, 4: 8, 5: 15, 6: 20, 7: 22, 8: 20, 9: 15, 10: 8, 11: 2, 12: -2}
    df_time["T_out"] = (
        df_time["month"].map(month_base)
        + 5 * df_time["sin_month"]
        + rng.normal(0, 1.5, size=n)
    )
    df_time["holiday_flag"] = (df_time["day_of_week"] >= 5).astype(int)

    df_person = pd.DataFrame({
        "course": rng.randint(1,5, size=n),
        "diligence": rng.normal(loc=1.0, scale=0.1, size=n).clip(0.5, 1.5),
    })

    df_person["exam_period"] = ((df_time["month"] % 6).isin([5, 0])).astype(int)
    df_person["on_campus"] = (rng.rand(n) < (1 - (df_person["course"] - 1) * 0.1)).astype(int)

    base_hw = 2 + (df_person["course"]-1) * 0.5
    exam_adj = df_person["exam_period"]    * rng.normal(1.0, 0.2, size=n)
    nonexam = (1 - df_person["exam_period"]) * rng.normal(-0.5, 0.2, size=n)
    df_person["hw_duration"] = (
        base_hw + exam_adj + nonexam
        + np.where(df_person["on_campus"] == 0, 1.0, 0.0)
        + rng.normal(0,0.5, size=n)
    ).clip(0.5, 6.0)

    df_person["sleep_start_hour"] = np.where(
        df_person["on_campus"]==1,
        rng.normal(23, 1, size=n),
        (rng.normal(1, 1, size=n) % 24)
    ).clip(0, 23)
    df_person["sleep_duration"] = rng.normal(7 - 0.2*(df_person["course"] - 1), 1.0, size=n).clip(4, 10)

    base_leis = np.where(df_person["on_campus"]==1,2.0,2.5)
    df_person["leisure_duration"] = (
        base_leis
        + df_time["holiday_flag"]*rng.normal(1.0,0.3,size=n)
        + rng.normal(0,0.5,size=n)
    ).clip(0.5,6.0)

    df_person["breakfast_duration"] = np.where(
        df_person["on_campus"] == 1,
        rng.normal(20, 5, size=n),
        rng.normal(30, 5, size=n)
    ).astype(int).clip(5, 60)
    df_person["commute_duration"] = np.where(df_person["on_campus"] == 1, rng.normal(40, 10, size=n), 0).astype(int).clip(0, 90)
    df_person["dinner_duration"] = rng.normal(45, 10, size=n).astype(int).clip(30, 60)

    base_ev = 60 + 10*df_person["course"]
    df_person["evening_study_duration"] = (
        base_ev
        + df_person["exam_period"]*rng.normal(30,10,size=n)
        + (1-df_person["exam_period"])*rng.normal(-15,5,size=n)
    ).clip(30,120).astype(int)

    df = pd.concat([df_time.reset_index(drop=True),
                    df_person.reset_index(drop=True)], axis=1)


    # healthy
    stress = df["hw_duration"] + df["evening_study_duration"]/60
    health_prob = (
        0.8
        - (df["sleep_start_hour"] - 22)/24    # поздний сон снижает
        - (df["sleep_duration"] - 7)/20       # мало сна снижает
        + (df["T_out"] > 0)*0.1               # тепло помогает
        - stress*0.05                         # стресс снижает
    )
    df["healthy"] = rng.binomial(1, health_prob.clip(0.05,0.95))

    # socialness
    soc = (
        df["leisure_duration"]/6 * 0.4
        + df["on_campus"]*0.2
        + df["holiday_flag"]*0.1
        - (df["course"]-1)*0.05
        + rng.normal(0,0.05,size=n)
    )
    df["socialness"] = soc.clip(0,1)
    
    # hospitalized
    hospitalization_prob = (
        0.001
        + (1 - df["healthy"]) * 0.01
        + (stress > 5) * 0.005
    )
    df["hospitalized"] = rng.binomial(1, hospitalization_prob.clip(0.0001, 0.02)).astype(bool)
    # Оставляем только нужные колонки
    cols = [
        "datetime", "month", "T_out", "holiday_flag",
        "course", "on_campus", "diligence", "exam_period",
        "hw_duration", "sleep_start_hour", "sleep_duration", "leisure_duration", 
        "breakfast_duration", "commute_duration", "socialness", "healthy",
        "dinner_duration", "evening_study_duration", "hospitalized"
    ]
    return df[cols]

df = generate_synthetic_student_data(n=30000)
df.to_csv("synthetic_student_features.csv", index=False)
print("synthetic_student_features.csv создано, строк:", len(df))
