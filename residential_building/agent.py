"""
reference: 9-storey “II-68” panel block)
9 floors · 3 entrances · 72 flats · ≈180 residents
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
    HEAT_START, HEAT_STOP = (10,15), (4,15)

    def __init__(self, model):
        super().__init__(model)
        self._last_p: float | None = None
        self.consumption = 0.0  # W

    def _heating(self, dt: _dt.datetime):
        m_d = (dt.month, dt.day)
        return (m_d >= ResidentialBuildingAgent.HEAT_START) or (
               m_d <= ResidentialBuildingAgent.HEAT_STOP)

    def _lift_kw(self, p_now):
        if self._last_p is None:
            self._last_p = p_now
            return 0.0
        trips = abs(p_now - self._last_p) * self.FULL_PRESENCE_TRIPS
        self._last_p = p_now
        return trips * self.ELEV_TRIP_KWH  # kWh/1 h == kW

    def step(self):
        dt  = self.model.current_datetime
        prs = max(0.0, min(getattr(self.model, "presence_in_building", 1.0), 1.0))
        kw  = (self.LIGHT_KW + self.FAN_KW + self.IT_KW +
               (self.PUMP_KW if self._heating(dt) else 0.0) +
               self._lift_kw(prs))
        self.consumption = kw * 1_000
        if getattr(self.model, "verbose", False):
            print(f"[Old {self.unique_id} {dt:%F %H}] pres={prs:.2f} load={kw:.2f} kW")
