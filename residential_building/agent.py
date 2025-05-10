"""
reference: 9-storey “II-68” panel block)
12 floors  · 84 flats · 190 residents  СП 54.13330-2016
2 rope lifts (non-regen) · stairwell fans · fixed-output corridor lights

Constants (average kW)
LIGHT_KW  = 3.00   # 6 kWh m-2 yr-1 OДН lighting norm × 5 000 m² / 8 760 h
FAN_KW    = 0.40   # 2 axial VЦ-200 fans @ 200 W
IT_KW     = 0.15   # CCTV + intercom + fire/BMS
PUMP_KW   = 0.24   # 2 × UPS 25-60 pumps @ 0.12 kW (winter only)

Elevator model
ISO 25745-2 cat.-2 ⇒ 110-130 trips day-¹ (two cars).  
"""

import datetime as _dt
from mesa import Agent

class ResidentialBuildingAgent(Agent):
    LIGHT_KW = 3.00
    FAN_KW   = 0.40
    IT_KW    = 0.15
    PUMP_KW  = 0.24
    FULL_PRESENCE_TRIPS = 120       # ISO cat-2 mid-value
    ELEV_TRIP_KWH       = 0.10      # KONE MonoSpace EPD
    HEATING_DENSITY_W_PER_M2 = 22.8  # Вт/м² в отопительный сезон (~100 кВт·ч/м²·год)
    HEAT_AREA_M2 = 4000              # отапливаемая площадь здания в м² (по проекту)
    HEAT_START, HEAT_STOP = (10,15), (4,15)

    def __init__(self, model):
        super().__init__(model)
        self._last_p = None
        self.consumption = 0.0  # W

    def _heating(self, dt: _dt.datetime):
        m_d = (dt.month, dt.day)
        return (m_d >= ResidentialBuildingAgent.HEAT_START) or (
               m_d <= ResidentialBuildingAgent.HEAT_STOP)

    def _lift_kw(self, p_now):
        if self._last_p is None or self.model.current_datetime.hour == 23 or self.model.current_datetime.hour < 4:
            self._last_p = p_now
            return 0.0
        
        trips = abs(p_now - self._last_p) * self.FULL_PRESENCE_TRIPS
        self._last_p = p_now
        return trips * self.ELEV_TRIP_KWH  # kWh/1 h == kW

    def step(self):
        dt = self.model.current_datetime
        prs = self.model.num_home / self.model.num_people_agents

        # Определяем состояние отопительного сезона
        heating_on = self._heating(dt)
        # Мощность на отопление (кВт)
        heating_kw = (self.HEAT_AREA_M2 * self.HEATING_DENSITY_W_PER_M2 / 1000.0) if heating_on else 0.0

        # Электрическая нагрузка (кВт)
        kw = (
            self.LIGHT_KW
            + self.FAN_KW
            + self.IT_KW
            + (self.PUMP_KW if heating_on else 0.0)
            + heating_kw  # учёт тепловой нагрузки как эквивалент электрической энергии
            + self._lift_kw(prs)
        )
        # Переводим в Wh за час
        self.consumption = kw * 1000.0

        # print(f"[Old {self.unique_id} {dt:%F %H}] pres={prs:.2f} lastpres={self._last_p} dpres= {abs(prs - self._last_p)} load={kw:.2f} kW")
