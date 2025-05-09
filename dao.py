import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_simulation_results(results, file_path):
    """
    Сохраняет результаты симуляции в CSV-файл.
    """
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def create_hourly_bar(all_consumption, save_dir="charts"):
    """
    Строит столбчатую диаграмму, отображающую среднее суммарное энергопотребление
    агентов для каждого часа суток.
    """
    df = pd.DataFrame(all_consumption)
    hourly_avg = df.groupby("Hour")["Total Consumption"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(hourly_avg["Hour"], hourly_avg["Total Consumption"], edgecolor='black')
    plt.title("Среднее суммарное энергопотребление агентов по часам")
    plt.xlabel("Час суток")
    plt.ylabel("Среднее суммарное энергопотребление")
    plt.grid(True, axis='y')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "hourly_bar.png"))


def plot_daily_heatmap(all_consumption, save_dir="charts"):
    df = pd.DataFrame(all_consumption)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    pivot = df.pivot_table(
        index='Hour', columns='Weekday', values='Total Consumption', aggfunc='mean'
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        cbar_kws={'label': 'Среднее потребление'},
        xticklabels=['Пн','Вт','Ср','Чт','Пт','Сб','Вс']
    )
    plt.title("Среднее потребление по часам и дням недели")
    plt.xlabel("День недели")
    plt.ylabel("Час суток")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "daily_heatmap.png"))


def plot_weekday_vs_weekend(all_consumption, save_dir="charts"):
    df = pd.DataFrame(all_consumption)
    df['Date'] = pd.to_datetime(df['Date'])
    df['IsWeekend'] = df['Date'].dt.weekday >= 5
    agg = df.groupby(['IsWeekend', 'Hour'])['Total Consumption'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    for flag, grp in agg.groupby('IsWeekend'):
        label = 'Выходные' if flag else 'Будни'
        plt.plot(grp['Hour'], grp['Total Consumption'], marker='o', label=label)
    plt.title("Профиль энергопотребления: будни vs выходные")
    plt.xlabel("Час суток")
    plt.ylabel("Среднее потребление")
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "weekday_vs_weekend.png"))


def plot_hourly_boxplot(all_consumption, save_dir="charts"):
    df = pd.DataFrame(all_consumption)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Hour', y='Total Consumption', data=df)
    plt.title("Распределение потребления по часам (box‑plot)")
    plt.xlabel("Час суток")
    plt.ylabel("Total Consumption")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "hourly_boxplot.png"))


def plot_rolling_trend(all_consumption, window=24, save_dir="charts"):
    df = pd.DataFrame(all_consumption)
    df['Step'] = range(len(df))
    df['Rolling'] = df['Total Consumption'].rolling(window=window).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(df['Step'], df['Rolling'])
    plt.title(f"Сглаженный тренд энергопотребления (окно {window}ч)")
    plt.xlabel("Шаг симуляции")
    plt.ylabel("Rolling mean Total Consumption")
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "rolling_trend.png"))


# Бар-чарт суммарного потребления по типам
def bar_chart_by_type(per_type_consumption, save_dir="charts"):
    df_type = pd.DataFrame(per_type_consumption)
    total_by_type = df_type.groupby('type')['consumption'].sum()
    total_by_type.plot.bar()
    plt.title("Суммарное потребление по типам агентов")
    plt.ylabel("Вт*ч")
    plt.savefig(os.path.join(save_dir, "bar_chart_by_type.png"))


# Среднее потребление по часам в разбивке по типам
def avg_by_type(per_type_consumption, save_dir="charts"):
    df = pd.DataFrame(per_type_consumption)
    avg = df.groupby(['Hour','type'])['consumption'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg, x='Hour', y='consumption', hue='type', estimator=None)
    plt.title("Среднее почасовое потребление по типам агентов (сырые данные)")
    plt.xlabel("Час суток")
    plt.ylabel("Потребление (Вт*ч)")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "avg_by_type.png"))
    plt.close()


def create_hourly_line(all_consumption, save_dir="charts"):
    """
    Строит сглаженный линейный график среднего энергопотребления по часам.
    """
    import os
    df = pd.DataFrame(all_consumption)
    hourly_avg = df.groupby("Hour")["Total Consumption"].mean()
    # Скользящее среднее
    smoothed = hourly_avg.rolling(window=3, center=True, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed.index, smoothed.values, marker='o')
    plt.fill_between(smoothed.index, smoothed.values, alpha=0.2)
    # Подсветка ночных часов
    plt.axvspan(21, 24, color='navy', alpha=0.05)
    plt.axvspan(0, 5, color='navy', alpha=0.05)
    plt.xticks(range(0, 24))
    plt.title("Сглаженный профиль энергопотребления по часам")
    plt.xlabel("Час суток")
    plt.ylabel("Среднее энергопотребление (Вт*ч)")
    plt.grid(alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "hourly_line.png"))


def heatmap_by_type(per_type_consumption, agent_type, save_dir="charts"):
    df = pd.DataFrame(per_type_consumption)
    df_t = df[df['type']==agent_type].copy()
    if df_t.empty:
        print(f"Нет данных для типа агента: {agent_type}")
        return
    df_t['weekday'] = df_t['Date'].dt.weekday
    pivot = df_t.pivot_table(index='Hour', columns='weekday', values='consumption', aggfunc='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': 'Среднее потребление'})
    plt.title(f"{agent_type.capitalize()}: потребление по часам и дням недели")
    plt.xlabel("День недели")
    plt.ylabel("Час суток")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmap_{agent_type}.png"))
    plt.close()


def contribution_area_chart(per_type_consumption, save_dir="charts"):
    df = pd.DataFrame(per_type_consumption)
    pivot = df.pivot_table(index='Date', columns='type', values='consumption', aggfunc='sum')
    frac = pivot.divide(pivot.sum(axis=1), axis=0)
    plt.figure(figsize=(12, 6))
    frac.plot.area()
    plt.title("Доля вклада каждого типа агентов в суммарное потребление")
    plt.xlabel("Дата")
    plt.ylabel("Доля")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "contribution_area_chart.png"))
    plt.close()


def create_hourly_bar_per_agent(df, agent_type, save_dir="charts"):
    df = pd.DataFrame(df)
    df['Date'] = pd.to_datetime(df['Date'])
    df_t = df[df['Type'] == agent_type]
    if df_t.empty:
        print(f"Нет данных для типа агента: {agent_type}")
        return
    agent_hour = df_t.groupby(['Agent ID', 'Hour'])['Consumption'].mean().reset_index()
    hourly_avg = agent_hour.groupby('Hour')['Consumption'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(hourly_avg['Hour'], hourly_avg['Consumption'], edgecolor='black')
    plt.title(f"Среднее потребление на агента ({agent_type}) по часам")
    plt.xlabel("Час суток")
    plt.ylabel("Среднее потребление (Вт*ч)")
    plt.grid(True, axis='y')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"hourly_bar_per_{agent_type}.png"))
    plt.close()


def generate_all_plots(all_consumption, per_type_df, per_agent_consumption, agent_types, save_dir="charts"):
    """
    Вызывает все функции построения графиков для визуализации данных симуляции.
    """
    create_hourly_bar(all_consumption, save_dir)
    plot_daily_heatmap(all_consumption, save_dir)
    plot_weekday_vs_weekend(all_consumption, save_dir)
    plot_hourly_boxplot(all_consumption, save_dir)
    plot_rolling_trend(all_consumption, window=24, save_dir=save_dir)
    
    for atype in agent_types:
        type_data = per_type_df[per_type_df["type"] == atype]
        heatmap_by_type(type_data, atype, save_dir)
        agent_data = per_agent_consumption[per_agent_consumption["Type"] == atype]
        create_hourly_bar_per_agent(agent_data, agent_type=atype, save_dir=save_dir)
    
    avg_by_type(per_type_df, save_dir)
    create_hourly_line(all_consumption, save_dir)
    contribution_area_chart(per_type_df, save_dir)
    bar_chart_by_type(per_type_df, save_dir)
