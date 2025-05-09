import time
import pandas as pd
import numpy as np


def generate_student_test_features(
    n_days: int,
    hours_per_day: int,
    num_students: int,
    seed: int = 42,
    output_path: str = "student_test_features.csv"
):
    """
    Генерация набора признаков для тестирования обученных моделей студентов.
    Файл содержит по каждой временной точке и каждому агенту:
      - datetime, hour, day_of_week, month, holiday_flag, T_out, sin_month, cos_month, exam_period
      - course, on_campus, diligence, sleep_duration
    Эти признаки будут использоваться для прогнозирования:
      hw_duration, sleep_start_hour, leisure_duration,
      commute_duration, evening_study_duration, healthy.
    """
    rng = np.random.RandomState(seed)
    # 1) Формируем временной ряд глобальных признаков
    n = n_days * hours_per_day
    dates = pd.date_range(start="2021-01-01", periods=n, freq="h")
    df_time = pd.DataFrame({"datetime": dates})
    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["month"] = df_time["datetime"].dt.month
    df_time["exam_period"] = ((df_time["month"] % 6).isin([5,0])).astype(int)

    # 2) Формируем статические признаки для агентов
    agent_ids = np.arange(num_students)
    courses = rng.randint(1, 5, size=num_students)
    diligences = rng.normal(1.0, 0.1, size=num_students)  # среднее 1.0, σ=0.1
    base_prob = 1 - (courses - 1) * 0.1
    noisy_prob = (base_prob * 0.95) + 0.05 * rng.rand(num_students)
    on_campus_flags = (rng.rand(num_students) < noisy_prob).astype(int)
    sleep_dur = rng.normal(7 - 0.2*(courses - 1), 1.0, size=num_students).clip(4,10)
    
    df_agents = pd.DataFrame({
        "agent_id": agent_ids,
        "course": courses,
        "diligence": diligences,
        "on_campus": on_campus_flags,
        "sleep_duration": sleep_dur
    })

    # 3) Кросс-джойн: каждая временная точка × каждый агент
    df_time["_tmpkey"] = 1
    df_agents["_tmpkey"] = 1
    df_full = df_time.merge(df_agents, on="_tmpkey").drop(columns=["_tmpkey"])

    df_full["hospitalized"] = (rng.rand(len(df_full)) < 0.005).astype(bool)
    # 4) Оставляем только нужные колонки (входные признаки для моделей)
    features = [
        "exam_period", "course", "on_campus", "diligence", "sleep_duration", "hospitalized",
    ]
    df_full = df_full[["agent_id", "datetime"] + features]

    # 5) Сохранение
    df_full.to_csv(output_path, index=False)
    print(f"{output_path} создано: {len(df_full)} строк")
