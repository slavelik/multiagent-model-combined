import os
import pickle
import pandas as pd
from mesa import Agent
import numpy as np

class ElectricCarAgent(Agent):
    """
    Агент-электромобиль с прогнозируемыми состояниями:
      - 'driving': в пути
      - 'charging': зарядка
      - 'idle': простаивает

    Состояние меняется каждый час и зависит от признаков:
      - day_of_week, holiday, month, hour
      - battery_capacity_wh: float
      - current_charge_wh: float
      - planned_distance_km: float
      - efficiency_wh_per_km: float
      - charging_preference: str ['overnight', 'fast', 'midday']
    """

    P_HOME_CHARGE = 7000  # Домашняя зарядка
    P_FAST_CHARGE = 50000  # Быстрая зарядка
    P_DRIVE       = 10000  # Потребление при вождении
    P_IDLE        = 0

    def __init__(self, model, agent_id):
        super().__init__(model)
        self.agent_id = agent_id
        self.hourly_states = []
        self.consumption = 0.0

    def compute_hourly_consumption(self):
        hour = self.model.current_hour
        rev_state_map = {0: 'idle', 1: 'driving', 2: 'charging'}
        state = rev_state_map[self.hourly_states[hour]]

        if state == 'idle' and np.random.random() < 0.05:  # 5% шанс неожиданной поездки
            state = 'driving'
        elif state == 'charging' and np.random.random() < 0.1:  # 10% шанс отмены зарядки
            state = 'idle'

        if state == 'driving':
            # Дневной пробег на 3 поездки
            planned_distance = self.model.ev_test_features.loc[
                (self.model.ev_test_features['agent_id'] == self.agent_id) &
                (self.model.ev_test_features['datetime'] == self.model.current_datetime),
                'planned_distance_km'
            ].iloc[0]
            efficiency = self.model.ev_test_features.loc[
                (self.model.ev_test_features['agent_id'] == self.agent_id) &
                (self.model.ev_test_features['datetime'] == self.model.current_datetime),
                'efficiency_wh_per_km'
            ].iloc[0]
            planned_distance += np.random.normal(0, 5)
            efficiency += np.random.normal(0, 10)
            power = (planned_distance / 3) * efficiency + np.random.normal(0, 1000)
            power = max(0, power)
        elif state == 'charging':
            pref = self.model.ev_test_features.loc[
                (self.model.ev_test_features['agent_id'] == self.agent_id) &
                (self.model.ev_test_features['datetime'] == self.model.current_datetime),
                'charging_preference'
            ].iloc[0]
            if pref == 'fast':
                rate = self.P_FAST_CHARGE + np.random.normal(0, 2000)  # Шум ±2000 Вт
            else:
                rate = self.P_HOME_CHARGE + np.random.normal(0, 500)  # Шум ±500 Вт
            power = rate
        else:
            power = self.P_IDLE

        self.consumption = max(0, power)

    def step(self):
        self.compute_hourly_consumption()
        # print(
        #     f"[E-Car {self.agent_id}] hour={self.model.current_hour} | "
        #     f"state={self.hourly_states[self.model.current_hour]} | "
        #     f"cons={self.consumption:.1f} Wh"
        # )