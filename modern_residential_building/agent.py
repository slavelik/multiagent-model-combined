import datetime as _dt
from mesa import Agent

class ModernResidentialBuildingAgent(Agent):
    """
    Современный 17-этажный дом (серия П-44ТМ),  310 жителей.
    Учитываются электрические нагрузки:
      • коридорное освещение (LED + PIR);            LIGHT_STBY_KW + LIGHT_DELTA_KW·Δpresence
      • вытяжные вентиляторы (EC);                   FAN_KW
      • ИТ-система, лифтовой БМС;                    IT_KW
      • циркуляционные насосы (VFD);                 PUMP_KW (только отопительный сезон)
      • лифты (рег-привод, реген.);                  trips·ELEV_TRIP_KWH
    """

    # ── удельные мощности, кВт ───────────────────────────────────────────
    LIGHT_STBY_KW   = 0.80         # дежурное освещение
    LIGHT_DELTA_KW  = 2.20         # добавка при Δpresence = 1
    FAN_KW          = 0.50
    IT_KW           = 0.20
    PUMP_KW         = 0.15         # 2 × 75 Вт

    # ── лифты ────────────────────────────────────────────────────────────
    ELEV_TRIP_KWH   = 0.06         # энергозатраты на поездку
    FULL_PRES_TRIPS = 180          # поездок при Δpresence = 1

    # ── отопительный сезон: 15 окт – 15 апр ─────────────────────────────
    HEAT_START = (10, 15)
    HEAT_STOP  = (4, 15)

    def __init__(self, model):
        super().__init__(model)
        self._last_presence: float | None = None   # доля жителей «дома» час назад
        self.consumption = 0.0                     # Wh за текущий час

    # ---------- вспомогательные ----------
    @staticmethod
    def _in_heating_season(dt: _dt.datetime) -> bool:
        m, d = dt.month, dt.day
        return (
            (m > 10) or (m == 10 and d >= 15) or
            (m < 4)  or (m == 4  and d <= 15)
        )

    def _lift_energy_kwh(self, pres_now: float) -> float:
        """Расход лифтов за час, кВт·ч."""
        h = self.model.current_datetime.hour
        if self._last_presence is None or 0 <= h < 4 or h == 23:
            self._last_presence = pres_now
            return 0.0

        trips = abs(pres_now - self._last_presence) * self.FULL_PRES_TRIPS
        self._last_presence = pres_now
        return trips * self.ELEV_TRIP_KWH

    # ---------- главный шаг ----------
    def step(self):
        dt   = self.model.current_datetime
        pres = self.model.num_home / self.model.num_people_agents

        # 1) фиксированные и сезонные нагрузки (кВт)
        kw_fixed = self.FAN_KW + self.IT_KW + self.LIGHT_STBY_KW
        kw_pump  = self.PUMP_KW if self._in_heating_season(dt) else 0.0

        # 2) динамика освещения (Δpresence) и лифтов
        delta_p  = 1.0 if self._last_presence is None else abs(pres - self._last_presence)
        kw_light = self.LIGHT_DELTA_KW * delta_p
        kw_lifts = self._lift_energy_kwh(pres)

        # 3) итоговая мощность → потребление за час (Вт·ч)
        kw_total         = kw_fixed + kw_pump + kw_light + kw_lifts
        self.consumption = kw_total * 1_000
