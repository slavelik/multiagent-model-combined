import pandas as pd

# 1. Загрузка исходных данных с минутным разрешением
path = 'EnterpriseBuilding/data/Steel_industry_data.csv'
df = (
    pd.read_csv(path, parse_dates=['datetime'], dayfirst=True)
      .set_index('datetime')
)

# 2. Отбираем последние 15 минут каждого часа (минуты 45–59)
df_last15 = df[(df.index.minute >= 45) & (df.index.minute < 60)].copy()

# 3. Циклический сдвиг на 2 дня внутри года + на год вперед
new_indices = []
for ts in df_last15.index:
    orig_year = ts.year
    plan_year = orig_year + 3

    # всего дней в плановом году (учитываем високосность)
    days_in_year = (pd.Timestamp(plan_year + 1, 1, 1) -
                    pd.Timestamp(plan_year, 1, 1)).days

    # номер дня в году (1–days_in_year)
    doy = ts.timetuple().tm_yday

    # сдвиг по кругу на 2 дня
    new_doy = ((doy + 2 - 1) % days_in_year) + 1

    # строим новую дату
    new_dt = pd.Timestamp(plan_year, 1, 1) + pd.Timedelta(days=new_doy - 1)
    # устанавливаем час и усечём до часа
    new_dt = new_dt.replace(hour=ts.hour, minute=0, second=0)
    new_indices.append(new_dt)

# 4. Присваиваем новые индексы
df_plan = df_last15.copy()
df_plan.index = new_indices
df_plan.index = df_plan.index.floor('H')

# 5. Сортируем по индексу, чтобы первые два дня (Jan 1 и Jan 2) оказались в начале
df_plan.sort_index(inplace=True)

# 6. Оставляем только нужные столбцы и сохраняем
df_plan = df_plan[['Load_Type', 'Motor_and_Transformer_Load_kVarh']]

output_path = 'EnterpriseBuilding/data/production_plan.csv'
df_plan.to_csv(output_path, index_label='datetime')

print(f"Production plan saved to {output_path}, total rows: {len(df_plan)}")
