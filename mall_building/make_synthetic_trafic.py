import pandas as pd
import numpy as np

np.random.seed(42)

SRC  = "MallBuilding/data/mall_traffic_2024.csv"
DEST = "MallBuilding/data/mall_traffic_synthetic.csv"

COLD_THRESHOLD  = -10   # °C
HEAT_THRESHOLD  =  28   # °C
COLD_RATE       =  0.4  # pp per °C below threshold
HEAT_RATE       =  0.2  # pp per °C above threshold
DAY_OFF_BONUS   =  0.15 # +15 % on days off
NOISE_STD       =  10    # standard deviation of noise
OPEN_HOURS      = range(10, 22)  # 10:00–21:59

def adjust_row(row):
    v = row.occupancy_rate

    # 1) бонус за выходной или праздник
    if row.day_off:
        v += DAY_OFF_BONUS * v

    # 2) температурный эффект
    if row.T_out <= COLD_THRESHOLD:
        v += COLD_RATE * (COLD_THRESHOLD - row.T_out)
    elif row.T_out >= HEAT_THRESHOLD:
        v += HEAT_RATE * (row.T_out - HEAT_THRESHOLD)

    # 3) белый шум
    v += np.random.normal(0, NOISE_STD)

    # 4) закрытые часы — в них 0
    if row.datetime.hour not in OPEN_HOURS:
        v = 0

    # 5) обрезаем до [0,100]
    v = max(0, min(v, 100))

    return round(v, 2)

# Основной блок
df = pd.read_csv(SRC, parse_dates=['datetime'])
df['occupancy_rate'] = df.apply(adjust_row, axis=1)

df.to_csv(DEST, index=False)
print(f"Synthetic dataset saved → {DEST} ({len(df):,} rows)")
