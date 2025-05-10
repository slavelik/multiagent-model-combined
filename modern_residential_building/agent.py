"""
Reference block
---------------
reference: 17-storey “П-44ТМ”, 2015
17 floors · 2 entrances · 136 flats · ≈ 310 residents  СП 54.13330-2016
3 gearless regen lifts · LED + PIR corridor lights · EC exhaust fans
VFD circulation pumps · full IP-BMS

Constants (average kW):
LIGHT_STANDBY  0.80   # 10 % of 8 kW full corridor load
LIGHT_DELTA    2.20   # additional when Δpresence = 1
FAN_KW         0.50   # 2 × EC fan 250 W
IT_KW          0.20
PUMP_KW        0.15   # 2 VFD pumps @ 75 W (winter)
ELEV_TRIP_KWH  0.06   # KONE MonoSpace DX, ISO A-class
FULL_PRES_TR   180    # ISO 25745 cat-3
"""

import datetime as _dt
from mesa import Agent

class ModernResidentialBuildingAgent(Agent):
    LIGHT_STBY_KW = 0.80
    LIGHT_DELTA_KW = 2.20
    FAN_KW   = 0.50
    IT_KW    = 0.20
    PUMP_KW  = 0.15
    ELEV_TRIP_KWH = 0.06
    FULL_PRES_TRIPS = 180
    HEATING_DENSITY_W_PER_M2 = 22.8  # Вт/м² в отопительный сезон
    HEAT_AREA_M2             = 7500  # отапливаемая площадь здания в м² (брутто по проекту)
    HEAT_START, HEAT_STOP = (10,15), (4,15)

    def __init__(self, model):
        super().__init__(model)
        self._last_p: float | None = None
        self.consumption = 0.0  # Wh

    def _heating(self, dt: _dt.datetime):
        m_d = (dt.month, dt.day)
        return (m_d >= ModernResidentialBuildingAgent.HEAT_START) or (
               m_d <= ModernResidentialBuildingAgent.HEAT_STOP)


    def _lift_kw(self, p_now):
        if self._last_p is None or self.model.current_datetime.hour == 23 or self.model.current_datetime.hour < 4:
            self._last_p = p_now
            return 0.0
        trips = abs(p_now - self._last_p) * self.FULL_PRES_TRIPS
        self._last_p = p_now
        return trips * self.ELEV_TRIP_KWH


    def _light_kw(self, delta_p):
        return self.LIGHT_STBY_KW + self.LIGHT_DELTA_KW * delta_p


    def step(self):
        dt = self.model.current_datetime
        prs = self.model.num_home / self.model.num_people_agents
        dprs = 1 if self._last_p is None else abs(prs - self._last_p)

        heating_on = self._heating(dt)
        heating_kw = (self.HEAT_AREA_M2 * self.HEATING_DENSITY_W_PER_M2 / 1000.0) if heating_on else 0.0

        # Электрическая нагрузка (кВт)
        kw_elec = (
            self._light_kw(dprs)
            + self.FAN_KW
            + self.IT_KW
            + (self.PUMP_KW if heating_on else 0.0)
            + self._lift_kw(prs)
        )
        # Суммарная мощность: электрическая + тепловая (кВт)
        kw = kw_elec + heating_kw
        self.consumption = kw * 1000.0
        # print(f"[Modern {self.unique_id} {dt:%F %H}] pres={prs:.2f} "
        #         f"Δp={dprs:.2f} load={kw:.2f} kW")
