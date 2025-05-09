import os
import pandas as pd
from multiagent_model import MultiAgentModel
import time
import cProfile, pstats


start = time.time()

student_feature_file       = r"person\student\ML\student_test_features.csv"
senior_feature_file        = r"person\senior\ML\senior_test_features.csv"
person_feature_file        = r"person\ML\person_test_features.csv"
evs_feature_file           = r"transport\ev\ML\ev_test_features.csv"
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

num_days   = 180
start_date = pd.Timestamp("2021-01-01")

n_persons            = 60
n_students           = 60
n_seniors            = 60
n_evs                = 0
n_enterprises        = 0
n_offices            = 0
n_hospitals          = 0
n_malls              = 0
n_modern_residential = 0
n_residential        = 0

model = MultiAgentModel(
    n_persons=n_persons,
    n_students=n_students,
    n_seniors=n_seniors,
    n_evs=n_evs,
    n_days=num_days,
    n_enterprises=n_enterprises,
    n_offices=n_offices,
    n_hospitals=n_hospitals,
    n_malls=n_malls,
    n_modern_residential=n_modern_residential,
    n_residential=n_residential,
    global_features_file=global_features_file,
    senior_test_file=senior_feature_file,
    student_test_file=student_feature_file,
    person_test_file=person_feature_file,
    evs_test_file=evs_feature_file,
    correction_factors_file=correction_factors_file,
    seed=42
)

total_agents = (n_persons + n_students + n_seniors + n_evs
               + n_enterprises + n_offices + n_hospitals + n_malls + n_modern_residential + n_residential)

profiler = cProfile.Profile()
profiler.enable()

for day in range(num_days*24):
    model.step()
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
