from datetime import datetime
import pandas as pd
import holidays
from meteostat import Stations, Hourly
import os

# 1. Интервал: с 2024-01-01 00:00 до 2024-12-31 23:00
start = datetime(2021, 1, 1, 0)
end   = datetime(2021, 12, 31, 23)

# 2. Координаты Воронежа
lat, lon = 51.6755, 39.2103

# 3. Находим ближайшую станцию к Воронежу
stations   = Stations()
station_df = stations.nearby(lat, lon).fetch(1)
station_id = station_df.index[0]

# 4. Загружаем почасовые данные
data = Hourly(station_id, start, end)
df   = data.fetch()

# 5. Оставляем и переименовываем колонку температуры
df = df[['temp']].rename(columns={'temp': 'T_out'})

# 6. Сбрасываем индекс, делаем datetime-столбец
df = df.reset_index().rename(columns={'time': 'datetime'})

# 7. Добавляем новые признаки из datetime
df['hour']         = df['datetime'].dt.hour
df['day_of_week']  = df['datetime'].dt.dayofweek
df['month']        = df['datetime'].dt.month

# 8. Признак выходного дня (суббота/воскресенье или праздник РФ)
ru_holidays = holidays.Russia(years=2021)
df['day_off'] = df['datetime'].apply(
    lambda x: (x.weekday() >= 5) or (x.date() in ru_holidays)
)

# 9. Оставляем только нужные столбцы и сохраняем
df = df[['datetime', 'hour', 'day_of_week', 'month', 'day_off', 'T_out']]
base = os.path.dirname(__file__)
data_path = os.path.join(base, 'data', 'environment_data.csv')
df.to_csv(data_path, index=False)

print("Файл voronezh_2024_weather.csv создан.")  
