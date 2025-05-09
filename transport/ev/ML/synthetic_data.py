import pandas as pd
import numpy as np

def generate_synthetic_electric_car_data(
    n_days: int = 365*5,
    hours_per_day: int = 24,
    seed: int = 42,
    output_path: str = "synthetic_ev_features.csv"
):
    """
    Генерация синтетических данных для обучения моделей прогнозирования состояний электромобиля.
    Файл содержит построчно для каждого часа:
      - day_of_week: 0-6
      - month: 1-12
      - hour: 0-23
      - battery_capacity_wh: вместимость батареи (Wh)
      - current_charge_wh: заряд в начале часа (Wh)
      - planned_distance_km: суммарный план пробега за день
      - efficiency_kwh_per_km: расход кВт·ч на км
      - charging_preference: 'overnight', 'midday', 'fast'
      - state: 'idle', 'driving', 'charging'
    charge динамически обновляется по часам, а вероятность состояний зависит от уровня заряда.
    """
    rng = np.random.default_rng(seed)
    records = []

    HOME_RATE = 7000    # Wh per hour
    FAST_RATE = 50000   # Wh per hour

    for day in range(n_days):
        capacity = rng.uniform(40e3, 80e3)  # Wh
        efficiency = rng.uniform(150, 250)
        pref = rng.choice(['overnight', 'midday', 'fast'])
        planned_dist = rng.uniform(20, 150)

        # Начальный заряд в начале дня
        charge = rng.uniform(0.2 * capacity, 0.8 * capacity)

        for hour in range(hours_per_day):
            dow = day % 7
            month = ((day // 30) % 12) + 1

            # Утренний и вечерний пик — в движении
            if hour in (7, 18):
                state = 'driving'
            else:
                # Расчет вероятностей на основе уровня заряда
                ratio = charge / capacity
                p_charge = np.clip((0.5 - ratio) * 2, 0, 1)
                p_drive = np.clip((ratio - 0.6) * 2.5, 0, 1)
                r = rng.random()
                if r < p_charge:
                    state = 'charging'
                elif r < p_charge + p_drive:
                    state = 'driving'
                else:
                    state = 'idle'

                # Предпочтение быстрой зарядки
                if state == 'charging':
                    if pref == 'midday' and hour != 13:
                        state = 'idle'
                    if pref == 'fast' and hour not in (19, 20):
                        state = 'idle'
                    if pref == 'overnight' and not (hour >= 22 or hour < 6):
                        state = 'idle'

            # Обновление уровня заряда
            if state == 'driving':
                # утренняя/вечерняя поездка: основная часть дневного пробега
                if hour in (7, 18):
                    dist = planned_dist * rng.uniform(0.4, 0.6)
                else:
                    dist = rng.uniform(0, 2)
                used = dist * efficiency * 1000  # Wh
                charge = max(0.0, charge - used)
            elif state == 'charging':
                rate = FAST_RATE if pref == 'fast' else HOME_RATE
                added = min(rate, capacity - charge)
                charge += added
            # idle — уровень заряда не меняется

            records.append({
                'day_of_week': dow,
                'holiday': dow >= 5,
                'month': month,
                'hour': hour,
                'battery_capacity_wh': capacity,
                'current_charge_wh': charge,
                'planned_distance_km': planned_dist,
                'efficiency_wh_per_km': efficiency,
                'charging_preference': pref,
                'state': state
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"{output_path} сохранён, строк: {len(df)}")

if __name__ == "__main__":
    generate_synthetic_electric_car_data()
