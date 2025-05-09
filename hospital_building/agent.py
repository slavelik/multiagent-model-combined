from mesa import Agent
from datetime import datetime

class HospitalBuildingAgent(Agent):
    """
    Простой агент «Больница»:
    – Два расхода: тепло (равномерно в отопительный сезон) и электроэнергия (динамика по загруженности и операциям).
    – Итог в self.consumption (Вт·ч) за час.
    """

    # Параметры здания
    AREA_M2       = 8000                         # площадь, м²
    BEDS_TOTAL    = 320                          # койко-мест
    # Удельная энергия, кВт·ч/м²·год
    EUI_HEAT_BASE = 108.0  # СП 50: q = 0.024·ГСОП (ГСОП≈4500) → 108 кВт·ч/м²·год 
    EUI_EL_BASE   = 65.0   # среднее по РФ-аудитам (45–80) → 65 кВт·ч/м²·год 

    # Расчёт часов отопительного сезона: 15 окт–15 апр ≈ 183 дня → 183*24
    HEAT_HOURS = 183 * 24
    _HEAT_HOURLY_NORM = EUI_HEAT_BASE * AREA_M2 / HEAT_HOURS  # кВт·ч/ч

    def __init__(self, model):
        super().__init__(model)
        self.consumption = 0.0  # Вт·ч за последний час

    def step(self):
        dt       = self.model.current_datetime
        patients = getattr(self.model, 'num_unhealthy', 0)
        occ      = (patients + min(self.model.num_hospitalized, self.BEDS_TOTAL)) / self.BEDS_TOTAL

        # 1) Тепло: только в отопительный сезон, равномерно
        if  (dt.month == 10 and dt.day >= 15) or (11 <= dt.month <= 12) \
          or (1 <= dt.month <= 3) or (dt.month == 4 and dt.day <= 15):
            heat_kWh = self._HEAT_HOURLY_NORM
        else:
            heat_kWh = 0.0

        # 2) Электричество: базовый часовой уровень
        base_el = self.EUI_EL_BASE * self.AREA_M2 / 8760.0  # кВт·ч/ч

        #    – occupancy влияет на вентиляцию+охлаждение+освещение ≈56 % электропотребления
        #    – операции/оборудование — около 20 % электропотребления
        dynamic_factor = 1 + 0.56 * occ

        el_kWh = base_el * dynamic_factor

        # Сохраняем итог Вт·ч
        total_kWh       = heat_kWh + el_kWh
        self.consumption = total_kWh * 1000

        # print(f"[Hospital {self.unique_id} {dt:%Y-%m-%d %H}] "
        #       f"occ={occ:.2f}  patients={patients:.2f}  "
        #       f"heat={heat_kWh:.1f}kWh  el={el_kWh:.1f}kWh")

