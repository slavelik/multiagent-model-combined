import numpy as np
from mesa import Agent

class OfficeBuildingAgent(Agent):
    """
    Электропотребление офисного здания:
      • цирк. насосы отопления (0,05 Вт/м², только в сезон)
      • вентиляция (1 Вт/м² днём, 30 % ночью)
      • освещение (10 Вт/м² днём, 1 Вт/м² ночью)
      • розеточная нагрузка ПК (150 Вт на сотрудника)
    """

    # ── удельные мощности ────────────────────────────────────────────────
    HEATING_PUMP_DENS = 0.05   # кВт → 50 Вт/м²? Actually W/m^2; but descending. (units should be W/m²)
    VENT_FAN_DENS     = 1.0
    LIGHT_DAY_DENS    = 10.0
    LIGHT_NIGHT_DENS  = 1.0
    PC_LOAD_W         = 150.0

    # ── отопительный сезон ──────────────────────────────────────────────
    HEAT_START = (10, 15)   # 15 октября
    HEAT_STOP  = (4, 15)    # 15 апреля

    def __init__(self, model):
        super().__init__(model)
        rng = np.random.RandomState(200 + self.unique_id)

        # Площадь и вместимость
        self.area     = float(rng.choice([500, 1000, 2000, 5000],
                                         p=[0.3, 0.4, 0.2, 0.1]))
        self.capacity = self.area / 6.5

        self.consumption = 0.0   # Wh за текущий час

    # ---------- служебные ----------
    @staticmethod
    def _in_heating_season(dt) -> bool:
        m, d = dt.month, dt.day
        return (
            (m > 10) or (m == 10 and d >= 15) or
            (m < 4)  or (m == 4  and d <= 15)
        )

    # ---------- основной шаг ----------
    def step(self):
        dt   = self.model.current_datetime
        hour = dt.hour

        # 1) численность сотрудников
        ppl_now = int(self.model.num_at_office / self.model.num_office_agents)
        ppl     = min(ppl_now, int(self.capacity))

        # 2) насосы отопления
        pump_load = self.area * self.HEATING_PUMP_DENS if self._in_heating_season(dt) else 0.0

        # 3) ночные коэффициенты
        night       = (hour >= 22 or hour < 7)
        vent_factor = 0.3 if night else 1.0
        light_dens  = self.LIGHT_NIGHT_DENS if night else self.LIGHT_DAY_DENS

        # 4) расчёт нагрузок (Вт)
        ventilation = self.area * self.VENT_FAN_DENS * vent_factor
        lighting    = self.area * light_dens
        plug_load   = ppl * self.PC_LOAD_W

        # 5) итог (Вт → Wh за час)
        self.consumption = (pump_load + ventilation + lighting + plug_load)
