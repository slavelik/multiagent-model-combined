import numpy as np
import matplotlib.pyplot as plt


yearly_consumption = {
    2000: 608526,
    2001: 618035,
    2002: 618237,
    2003: 632232,
    2004: 645532,
    2005: 649973,
    2006: 681401,
    2007: 700942,
    2008: 725460,
    2009: 686236,
    2010: 726683,
    2011: 728824,
    2012: 740285,
    2013: 744091,
    2014: 737830,
    2015: 732339,
    2016: 743971,
    2017: 760885,
    2018: 759582,
    2019: 755636,
    2020: 747462,
    2021: 807621,
    2022: 822819
}

coefficients = {}
previous_value = None

for year, value in yearly_consumption.items():
    if previous_value is None:
        coefficients[year] = np.float64(1.0)
    else:
        coefficients[year] = np.float64(value / previous_value)
    previous_value = value

print(coefficients)

# Словарь посчитанный
coefficients_1 = {
    2000: np.float64(1.0),
    2001: np.float64(1.0341529740457729),
    2002: np.float64(1.0354515099838468),
    2003: np.float64(1.0765940718563858),
    2004: np.float64(1.0672076410257174),
    2005: np.float64(1.057200696687822),
    2006: np.float64(1.0812541020746733),
    2007: np.float64(1.0735747300325609),
    2008: np.float64(1.0541352520165503),
    2009: np.float64(0.9413251821487689),
    2010: np.float64(1.0356576999921414),
    2011: np.float64(1.0342808094168483),
    2012: np.float64(1.0289334434093755),
    2013: np.float64(1.017849312211665),
    2014: np.float64(1.0043305105742777),
    2015: np.float64(0.9842233071571646),
    2016: np.float64(1.0016964358564318),
    2017: np.float64(1.014663450250924),
    2018: np.float64(1.0242994281993874),
    2019: np.float64(1.0183220660229662),
    2020: np.float64(0.9842396954364582),
    2021: np.float64(1.0469438990661053),
    2022: np.float64(0.9706654333037531),
    # 2023: np.float64(1.0278052399999986),
    # 2024: np.float64(1.0),
}

# Словарь из данных энергопотребления
coefficients_2 = {
    2000: np.float64(1.0),
    2001: np.float64(1.0156262838399674),
    2002: np.float64(1.000326842330936),
    2003: np.float64(1.0226369499075598),
    2004: np.float64(1.021036581508054),
    2005: np.float64(1.0068795969835733),
    2006: np.float64(1.0483527777307673),
    2007: np.float64(1.0286776802499555),
    2008: np.float64(1.0349786430260992),
    2009: np.float64(0.9459322360984754),
    2010: np.float64(1.058940364539313),
    2011: np.float64(1.0029462640518632),
    2012: np.float64(1.0157253328649989),
    2013: np.float64(1.0051412631621606),
    2014: np.float64(0.9915857065869631),
    2015: np.float64(0.9925579062927775),
    2016: np.float64(1.0158833545666692),
    2017: np.float64(1.0227347571343506),
    2018: np.float64(0.9982875204531565),
    2019: np.float64(0.9948050375074712),
    2020: np.float64(0.9891826223208),
    2021: np.float64(1.0804843590710966),
    2022: np.float64(1.0188182328096966),
}

# Calculate differences
absolute_differences = {year: abs(coefficients_1[year] - coefficients_2[year]) for year in coefficients_1}
percentage_differences = {year: (absolute_differences[year] / coefficients_1[year]) * 100 for year in coefficients_1}

# Calculate averages
average_absolute_difference = np.mean(list(absolute_differences.values()))
average_percentage_difference = np.mean(list(percentage_differences.values()))

print(f"Среднее абсолютное значение разницы: {average_absolute_difference}")
print(f"Средняя разница в процентах: {average_percentage_difference}%")

# Plotting
years = list(coefficients_1.keys())
values_1 = list(coefficients_1.values())
values_2 = list(coefficients_2.values())

plt.figure(figsize=(10, 6))
plt.plot(years, values_1, label="Посчитанные на основе мароэкономических параметров", marker="o")
plt.plot(years, values_2, label="Взятые из энергопотребления", marker="x")
plt.title("Сравнение годовых коэффициентов")
plt.xlabel("Год")
plt.ylabel("Значение коэффициента")
plt.legend()
plt.grid()
plt.show()

avg = 0
for v in coefficients_1.values():
    avg += v

avg = avg / len(coefficients_1)
print(f"Среднее значение: {avg}")