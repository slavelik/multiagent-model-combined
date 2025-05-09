import os
import sys
import numpy as np
import pandas as pd
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
csv_file_path = os.path.join(project_root, 'datasets', 'GDP_UNEMP.csv')


def preprocess_economic_data(csv_file):
    # Загружаем данные из CSV
    df = pd.read_csv(csv_file)
    russia_df = df[df["Country Name"] == "Russian Federation"]
    gdp_row = russia_df[russia_df["Series Name"] == "GDP (constant 2015 US$)"]
    unemp_row = russia_df[russia_df["Series Name"] == "Unemployment, total (% of total labor force) (national estimate)"]
    
    # Извлекаем года и данные
    year_columns = [col for col in df.columns if "[YR" in col]
    years = [int(col.split()[0]) for col in year_columns]
    
    # Заменяем ".." на NaN
    gdp_data = gdp_row[year_columns].replace("..", np.nan).iloc[0]
    unemp_data = unemp_row[year_columns].replace("..", np.nan).iloc[0]
    
    # Преобразуем в float
    gdp_data = gdp_data.astype(float)
    unemp_data = unemp_data.astype(float)
    
    # Создаем DataFrame для удобства
    econ_df = pd.DataFrame({"GDP": gdp_data.values, "Unemployment": unemp_data.values}, index=years)
    
    # Интерполируем пропущенные значения
    econ_df = econ_df.interpolate(method='linear')
    
    # Вычисляем корректирующий коэффициент на основе предыдущего года
    correction_factors = {}
    for year in econ_df.index:
        if year - 1 in econ_df.index:
            previous_gdp = econ_df.loc[year - 1, "GDP"]
            previous_unemp = econ_df.loc[year - 1, "Unemployment"]
        else:
            # Для первого года используем текущие данные
            previous_gdp = econ_df.loc[year, "GDP"]
            previous_unemp = econ_df.loc[year, "Unemployment"]
        
        current_gdp = econ_df.loc[year, "GDP"]
        current_unemp = econ_df.loc[year, "Unemployment"]
        
        # Расчет отношения ВВП и разницы в безработице
        gdp_ratio = current_gdp / previous_gdp if previous_gdp != 0 else 1.0
        unemp_diff = current_unemp - previous_unemp
        
        # Корректирующий коэффициент
        correction_factor = gdp_ratio * (1 + 0.01 * unemp_diff)
        correction_factors[year] = correction_factor
    
    # Сохраняем словарь в файл
    with open("correction_factors.pkl", "wb") as f:
        pickle.dump(correction_factors, f)
    return correction_factors


preprocess_economic_data(csv_file_path)

# with open('correction_factors.pkl', "rb") as f:
#     correction_factors = pickle.load(f)
#     print(correction_factors)