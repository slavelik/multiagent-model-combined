import os
import pandas as pd
from multiagent_model import MultiAgentModel
import time
import cProfile, pstats
from agent_data_generators import (
    generate_senior_test_features,
    generate_student_test_features,
    generate_person_test_features,
    generate_ev_test_features
)

start = time.time()

correction_factors_file    = r"economy\correction_factors.pkl"
global_features_file       = r"data\environment_data.csv"

# N = 500
# pct_students = 0.056     # 5.6%
# pct_seniors  = 0.177     # 17.7%
# pct_evs      = 0.0087    # 0.87%

# n_students = round(N * pct_students)
# n_seniors  = round(N * pct_seniors)
# n_evs      = round(N * pct_evs)
# n_persons  = N - (n_students + n_seniors + n_evs)

num_days   = 31
start_date = pd.Timestamp("2021-01-01")

n_persons            = 400
n_students           = 190
n_seniors            = 320
n_evs                = 1
n_enterprises        = 1
n_offices            = 2
n_hospitals          = 1
n_malls              = 1
n_modern_residential = 2
n_residential        = 2

# n_persons            = 10800
# n_students           = 1900
# n_seniors            = 3200
# n_evs                = 80
# n_enterprises        = 1
# n_offices            = 25
# n_hospitals          = 1
# n_malls              = 1
# n_modern_residential = 26
# n_residential        = 63

senior_df = generate_senior_test_features(num_days, n_seniors)
student_df = generate_student_test_features(num_days, n_students)
person_df = generate_person_test_features(num_days, n_persons)
ev_df = generate_ev_test_features(num_days, n_evs)
print("Данные сгенерированы")

model = MultiAgentModel(
    n_persons=n_persons,
    n_students=n_students,
    n_seniors=n_seniors,
    n_evs=n_evs,
    n_enterprises=n_enterprises,
    n_offices=n_offices,
    n_hospitals=n_hospitals,
    n_malls=n_malls,
    n_modern_residential=n_modern_residential,
    n_residential=n_residential,
    global_features_file=global_features_file,
    senior_test_feat=senior_df,
    student_test_feat=student_df,
    person_test_feat=person_df,
    evs_test_feat=ev_df,
    correction_factors_file=correction_factors_file,
    seed=42
)

total_agents = (n_persons + n_students + n_seniors + n_evs
               + n_enterprises + n_offices + n_hospitals + n_malls + n_modern_residential + n_residential)

profiler = cProfile.Profile()
profiler.enable()

start = time.time()
for day in range(num_days):
    for hour in range(24):
        model.step()
    time_spent = time.time() - start
    mean_day_time = time_spent / (day + 1)
    print(f"--- Информация по шагам моделирования: времени прошло (мин):" 
          f"{round((time_spent / 60), 2)};  осталось (мин):" 
          f"{round(((num_days - day - 1) * mean_day_time / 60), 2)} ---")

profiler.disable()
with open('profile_stats.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats('cumtime')
    stats.print_stats(10)

os.makedirs("charts", exist_ok=True)
os.makedirs("simulations_results", exist_ok=True)

model_df = model.datacollector.get_model_vars_dataframe()
model_df.reset_index(inplace=True)
model_df.rename(columns={'index': 'Step'}, inplace=True)
model_df.to_csv('simulations_results/model_data.csv', index=False)

agent_df = model.datacollector.get_agent_vars_dataframe()
agent_df.reset_index(inplace=True)
agent_df = agent_df.merge(model_df[['Step', 'Date']], on='Step')
agent_df = agent_df[['Step', 'Date', 'Type', 'Consumption']]
agent_df.to_csv('simulations_results/agent_data.csv', index=False)

time_spent = time.time() - start
print("Затраченное время: ", time_spent)

with open('log.txt', 'a', encoding='utf-8') as f:
    f.write(f"\n--- Новая симуляция ---\n")
    f.write(f"n_persons={n_persons}, n_students={n_students}, n_seniors={n_seniors}, n_evs={n_evs}, n_days={num_days}\n")
    f.write(f"Затраченное время: {time_spent}\n")
