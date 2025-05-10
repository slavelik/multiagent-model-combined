import pandas as pd
import os

# Пути к файлам
agent_path = os.path.join('simulations_results', 'agent_data.csv')
model_path = os.path.join('simulations_results', 'model_data.csv')
output_path = os.path.join('simulations_results', 'agent_data_with_apartments.csv')
# Чтение данных
df_agent = pd.read_csv(agent_path, parse_dates=['Date'])
df_model = pd.read_csv(model_path, parse_dates=['Date'])

# Словарь количества квартир
apt_counts = {
    'residentialbuilding': 72,
    'modernresidentialbuilding': 68,
}

# Оставляем только нужное из model_data
df_model = df_model[['Date', 'hourly_at_home']]

# Объединяем по дате-времени
df = df_agent.merge(df_model, on='Date', how='left')

# Функция добавления квартирного потребления
def add_apartment_consumption(row):
    count = apt_counts.get(row['Type'], 0)
    return row['Consumption'] + row['hourly_at_home'] * count

# Обновляем столбец Consumption
df['Consumption'] = df.apply(add_apartment_consumption, axis=1)

# Удаляем вспомогательный столбец
df = df.drop(columns=['hourly_at_home'])

# Сохраняем результат
df.to_csv(output_path, index=False)
print(f"Файл с обновлённым столбцом Consumption сохранён как: {output_path}")
