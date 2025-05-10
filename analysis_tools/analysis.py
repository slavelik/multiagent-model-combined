import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar

# Анализ энергопотребления по реальным агрегированным данным
# Паттерны:
# - День (0–23)
# - Неделя (0–167)
# - Месяц (1–31)
# - Год (1–12)

def ensure_dir(path):
    """Создаёт директорию, если её нет."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_plot_and_csv(series, xlabel, ylabel, title, csv_path, plot_path, xticks=None, xticklabels=None):
    """
    Сохраняет агрегированную серию в CSV и строит график линии по реальным значениям.
    """
    ensure_dir(os.path.dirname(csv_path))
    ensure_dir(os.path.dirname(plot_path))

    # Сохранение агрегированных данных
    series.to_csv(csv_path, header=[ylabel])

    # График по реальным точкам
    x = np.arange(len(series))
    y = series.values

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    if xticks is not None and xticklabels is not None:
        plt.xticks(xticks, xticklabels, rotation=45, ha='right', fontsize=10)
    else:
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def analyze_patterns(ts, name, output_dir, agg_func='mean'):
    """
    Анализ паттернов без сглаживания: агрегация и построение по реальным значениям.
    agg_func: 'sum' или 'mean'
    """
    ylabel = 'Consumption (Wh)' if 'consumption' in name else name

    # 1) День по часам
    daily = ts.groupby(ts.index.hour).agg(agg_func)
    hours = list(range(24))
    hour_labels = [f"{h}:00" for h in hours]
    save_plot_and_csv(
        daily, 'Hour of Day', ylabel, f'{name} — Daily Pattern',
        os.path.join(output_dir, f'{name}_daily.csv'),
        os.path.join(output_dir, f'{name}_daily.png'),
        xticks=hours, xticklabels=hour_labels
    )

    # 2) Неделя по часам
    weekly = ts.groupby(ts.index.dayofweek * 24 + ts.index.hour).agg(agg_func)
    ticks = [i * 24 for i in range(7)]
    day_labels = [calendar.day_abbr[i] for i in range(7)]
    save_plot_and_csv(
        weekly, 'Hour of Week', ylabel, f'{name} — Weekly Pattern',
        os.path.join(output_dir, f'{name}_weekly.csv'),
        os.path.join(output_dir, f'{name}_weekly.png'),
        xticks=ticks, xticklabels=day_labels
    )

    # 3) Месяц по дням
    monthly = ts.groupby(ts.index.day).agg(agg_func)
    days = list(monthly.index)
    day_labels = [str(d) for d in days]
    save_plot_and_csv(
        monthly, 'Day of Month', ylabel, f'{name} — Monthly Pattern',
        os.path.join(output_dir, f'{name}_monthly.csv'),
        os.path.join(output_dir, f'{name}_monthly.png'),
        xticks=days, xticklabels=day_labels
    )

    # 4) Год по месяцам
    yearly = ts.groupby(ts.index.month).agg(agg_func)
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]
    save_plot_and_csv(
        yearly, 'Month', ylabel, f'{name} — Yearly Pattern',
        os.path.join(output_dir, f'{name}_yearly.csv'),
        os.path.join(output_dir, f'{name}_yearly.png'),
        xticks=months, xticklabels=month_labels
    )


def analyze_contributions(df_agent, output_dir):
    """
    Строит графики и CSV с долей вклада каждого типа агента в общее потребление.
    """
    ensure_dir(output_dir)

    type_contrib = df_agent.groupby('Type')['Consumption'].sum()
    type_share = (type_contrib / type_contrib.sum()).sort_values(ascending=False)

    csv_types = os.path.join(output_dir, 'type_share.csv')
    png_types = os.path.join(output_dir, 'type_share.png')
    type_share.to_csv(csv_types, header=['share'])

    plt.figure(figsize=(8, 5))
    type_share.plot(kind='pie', autopct='%1.1f%%', ylabel='')
    plt.title('Contribution by Agent Type to Total Consumption', fontsize=14)
    plt.tight_layout()
    plt.savefig(png_types)
    plt.close()


def main():
    agent_path = os.path.join('simulations_results', 'agent_data.csv')
    model_path = os.path.join('simulations_results', 'model_data.csv')

    df_agent = pd.read_csv(agent_path, parse_dates=['Date']).set_index('Date')
    df_model = pd.read_csv(model_path, parse_dates=['Date']).set_index('Date')

    # 1) Общий расход энергии
    analyze_patterns(df_agent['Consumption'], 'total_consumption',
                     os.path.join('analysis', 'Consumption'), 'sum')

    # 2) По типам агентов (среднее)
    for agent_type, group in df_agent.groupby('Type'):
        analyze_patterns(
            group['Consumption'],
            f"{agent_type.lower().replace(' ', '_')}_consumption",
            os.path.join('analysis', agent_type.lower().replace(' ', '_') + '_consumption'),
            'mean'
        )

    # 3) Параметры модели
    for col in ['office_population', 'hospitalized', 'patients_total', 'presence_in_building']:
        analyze_patterns(df_model[col], col, os.path.join('analysis', col), 'mean')

    # 4) Процентные изменения параметров модели
    pct = df_model[['office_population', 'hospitalized', 'patients_total']].pct_change().dropna()
    changes_dir = os.path.join('analysis', 'changes')
    ensure_dir(changes_dir)
    pct.to_csv(os.path.join(changes_dir, 'model_params_pct_change.csv'))

    # 5) Анализ вклада
    analyze_contributions(df_agent, os.path.join('analysis', 'contribution'))

    print("Analysis completed. Результаты сохранены в папке 'analysis'.")

if __name__ == '__main__':
    main()
