import os, pickle, numpy as np, pandas as pd
from mesa import Agent

def load_clf(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

class MallAgent(Agent):
    """ТРЦ: электропотребление (освещение, арендаторы, вентиляция, эскалаторы, IT, насосы отопления)."""

    HEAT_START = (10, 15)   # 15 октября
    HEAT_STOP  = (4, 15)    # 15 апреля

    def __init__(
        self, model,
        floor_area: float = 6_990,
        escalator_count: int = 8,
        opening_hour: int = 10,
        closing_hour: int = 22
    ):
        super().__init__(model)
        self.floor_area       = floor_area
        self.escalator_count  = escalator_count
        self.opening_hour     = opening_hour
        self.closing_hour     = closing_hour

        # Удельные мощности, Вт/м²
        self.lighting_day      = 11
        self.lighting_night    = 1
        self.equipment_density = 12
        self.cooling_density   = 50
        self.vent_density      = 4
        self.it_density        = 2
        self.misc_density      = 5
        self.heating_pump_density = 0.05   # цирк. насосы отопления

        # Эскалаторы, Вт
        self.escalator_idle = 1_800
        self.escalator_peak = 5_000

        # Модель заполняемости
        pth = os.path.join(os.path.dirname(__file__), "trained_models", "best_mall_model.pkl")
        self.occ_clf = load_clf(pth)

        self.consumption = 0.0   # Wh за час

    # ---------- вспомогательные ----------
    def _heating_season(self, dt) -> bool:
        m, d = dt.month, dt.day
        return ((m > 10) or (m == 10 and d >= 15) or
                (m < 4)  or (m == 4  and d <= 15))

    def _predict_occ(self) -> float:
        dt = self.model.current_datetime
        feats = {
            "T_out":   self.model.current_T_out,
            "day_off": int(self.model.is_holiday),
            "hour_sin":  np.sin(2*np.pi*dt.hour/24),
            "hour_cos":  np.cos(2*np.pi*dt.hour/24),
            "dow_sin":   np.sin(2*np.pi*dt.weekday()/7),
            "dow_cos":   np.cos(2*np.pi*dt.weekday()/7),
            "month_sin": np.sin(2*np.pi*(dt.month-1)/12),
            "month_cos": np.cos(2*np.pi*(dt.month-1)/12)
        }
        return float(self.occ_clf.predict(pd.DataFrame([feats]))[0]) / 100.0

    # ---------- основной шаг ----------
    def step(self):
        dt    = self.model.current_datetime
        occ   = self._predict_occ() if self.opening_hour <= dt.hour < self.closing_hour else 0.0
        area  = self.floor_area

        # 1) Освещение
        light_dens = self.lighting_day if occ else self.lighting_night
        lighting   = light_dens * area

        # 2) Оборудование арендаторов
        equipment  = self.equipment_density * area * max(0.1, occ)

        # 3) Эскалаторы
        if self.opening_hour <= dt.hour < self.closing_hour:
            power_per_esc = self.escalator_peak if occ > 0.20 else self.escalator_idle
            escalators    = power_per_esc * self.escalator_count
        else:
            escalators = 0.0

        # 4) Вентиляция 24/7
        ventilation = self.vent_density * area

        # 5) Охлаждение (если T_out>24 °C)
        cooling = self.cooling_density * area if self.model.current_T_out > 24 else 0.0

        # 6) IT и прочие
        it_load   = self.it_density   * area
        misc_load = self.misc_density * area

        # 7) Насосы отопления (только сезон)
        pump_load = self.heating_pump_density * area if self._heating_season(dt) else 0.0

        # 8) Итоговая электрическая мощность, затем перевод в Wh
        kw_total = (lighting + equipment + escalators + ventilation +
                    cooling + it_load + misc_load + pump_load) / 1000.0
        self.consumption = kw_total * 1000.0  # Wh за час
