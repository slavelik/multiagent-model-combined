import os
import pickle
import numpy as np
import pandas as pd
from mesa import Agent


def load_clf(path: str):
    return pickle.load(open(path, 'rb'))

class MallAgent(Agent):
    """
    Агент торгово-развлекательного центра (ТРЦ).
    Расчёт энергопотребления на основе:
      - освещения,
      - оборудования арендаторов,
      - транспортных систем,
      - охлаждения,
      - вентиляции,
      - отопления.
    """

    HEAT_START = (10, 15)   # октябрь, 15 число
    HEAT_STOP  = (4, 15)    # апрель, 15 число

    def __init__(self, model,
                 floor_area: float = 12700,      # м² ТРЦ в Москве
                 escalator_count: int = 8,
                 opening_hour: int = 10,
                 closing_hour: int = 22):
        super().__init__(model)
        self.floor_area = floor_area
        self.escalator_count = escalator_count
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour

        # Плотности энергопотребления
        self.lighting_density = 18        # Вт/м² при работе (ASHRAE 90.1
        self.equipment_density = 15       # Вт/м² активного оборудования
        self.escalator_idle_power = 1800  # Вт (idle) 
        self.escalator_peak_power = 5000  # Вт (peak)
        self.cooling_density = 60         # Вт/м² при T_out>24°C
        self.ventilation_density = 4      # Вт/м² вентиляция 24/7
        self.it_density = 10.8
        self.other_density = 50
        # Минимальное ночное освещение (экстренное): 1 Вт/м²
        self.night_lighting_density = 1
        # Плотность тепловой нагрузки: 100 kWh/м²·год ≈ 11.4 Вт/м² в отопительный сезон
        self.heating_density = 11.4

        # Загружаем модель предсказания occupancy_rate
        clf_path = os.path.join(os.path.dirname(__file__), 'trained_models', 'best_mall_model.pkl')
        self.occ_clf = load_clf(clf_path)

        # Хранение результатов
        self.electric_consumption = 0.0  # Вт
        self.heat_consumption = 0.0      # Вт
        self.consumption = 0.0

    def predict_occupancy(self) -> float:
        """
        Предсказывает occupancy_rate по модели, используя признаки:
        T_out, day_off, циклические hour, day_of_week, month.
        """
        dt = self.model.current_datetime
        hour = dt.hour
        dow = dt.weekday()
        month = dt.month
        feats = {
            'T_out':     self.model.current_T_out,
            'day_off':   int(self.model.is_holiday),
            'hour_sin':  np.sin(2 * np.pi * hour / 24),
            'hour_cos':  np.cos(2 * np.pi * hour / 24),
            'dow_sin':   np.sin(2 * np.pi * dow / 7),
            'dow_cos':   np.cos(2 * np.pi * dow / 7),
            'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (month - 1) / 12)
        }
        X = pd.DataFrame([feats])
        return self.occ_clf.predict(X)[0]

    def in_heating_season(self, dt) -> bool:
        """Проверяет, в отопительном ли сезоне дата dt."""
        m, d = dt.month, dt.day
        sm, sd = self.HEAT_START
        em, ed = self.HEAT_STOP
        # от 15 октября до конца года
        if (m > sm) or (m == sm and d >= sd): return True
        # от начала года до 15 апреля
        if (m < em) or (m == em and d <= ed): return True
        return False

    def step(self):
        dt = self.model.current_datetime
        T_out = self.model.current_T_out
        it_load = self.it_density * self.floor_area
        other_density = self.other_density * self.floor_area
        # 1) Прогноз occupancy_rate
        occ = self.predict_occupancy() / 100

        # 2) Освещение
        if self.opening_hour <= dt.hour < self.closing_hour:
            light_d = self.lighting_density
        else:
            light_d = self.night_lighting_density
        lighting_load = light_d * self.floor_area

        # 3) Оборудование арендаторов
        equipment_load = self.equipment_density * self.floor_area * (0.2 if occ < 0.2 else occ) 

        # 4) Эскалаторы
        if self.opening_hour <= dt.hour < self.closing_hour:
            power = self.escalator_peak_power if occ > 0.1 else self.escalator_idle_power
            escalator_load = power * self.escalator_count
        else:
            escalator_load = 0

        # 5) Охлаждение
        cooling_load = self.cooling_density * self.floor_area if T_out > 24 else 0

        # 6) Вентиляция (круглосуточно)
        ventilation_load = self.ventilation_density * self.floor_area

        # Итоговое электричество
        self.electric_consumption = (
            lighting_load + equipment_load + escalator_load + 
            cooling_load + ventilation_load + it_load + other_density
        )

        # 7) Отопление (тепловая сеть)
        if self.in_heating_season(dt):
            self.heat_consumption = self.heating_density * self.floor_area
        else:
            self.heat_consumption = 0
        
        self.consumption = (self.electric_consumption + self.heat_consumption) / 1000.0  

        # # Логирование
        # print(f"[Mall {self.unique_id} | {dt}] Elec={self.electric_consumption:.0f} W, "
        #       f"Heat={self.heat_consumption:.0f} W, Occ={occ:.2f}, Total={self.consumption} W")
