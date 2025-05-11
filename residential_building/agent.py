import datetime as _dt
from mesa import Agent

class ResidentialBuildingAgent(Agent):
    """
    12-этажный панельный дом серии II-68, 190 жителей
    Учитываются только электрические нагрузки:
      • коридорное освещение, ИТ-оборудование, вытяжные вентиляторы
      • циркуляционные насосы (только в отопительный сезон)
      • лифты — энергозатраты пропорциональны изменению доли жителей дома
    """

    # ── постоянные мощности, кВт ──────────────────────────────────────────
    LIGHT_KW = 3.00   # ОДН-освещение
    FAN_KW   = 0.40   # 2×VЦ-200
    IT_KW    = 0.15   # видеонабл., домофон, пожарка
    PUMP_KW  = 0.24   # 2×UPS 25-60 (0,12 кВт каждый)

    # ── лифты ─────────────────────────────────────────────────────────────
    FULL_PRES_TRIPS = 120     # поездок при Δpresence = 1, ISO 25745-2
    ELEV_TRIP_KWH   = 0.10    # энергозатраты на одну поездку

    # ── отопительный сезон: 15 окт – 15 апр ──────────────────────────────
    HEAT_START = (10, 15)
    HEAT_STOP  = (4, 15)

    def __init__(self, model):
        super().__init__(model)
        self._last_presence = None   # доля жителей «дома» час назад
        self.consumption    = 0.0    # Wh за текущий час

    # ---------- утилиты ----------
    @staticmethod
    def _in_heating_season(dt: _dt.datetime) -> bool:
        m, d = dt.month, dt.day
        return (
            (m > 10) or (m == 10 and d >= 15) or
            (m < 4)  or (m == 4  and d <= 15)
        )

    def _lift_energy_kwh(self, pres_now: float) -> float:
        """
        Возвращает кВт·ч, израсходованные лифтами за час.
        В ночной «тихий» интервал 23:00–03:59 энергозатраты не считаются.
        """
        hour = self.model.current_datetime.hour
        if self._last_presence is None or 0 <= hour < 4 or hour == 23:
            self._last_presence = pres_now
            return 0.0

        trips = abs(pres_now - self._last_presence) * self.FULL_PRES_TRIPS
        self._last_presence = pres_now
        return trips * self.ELEV_TRIP_KWH   # кВт·ч (за 1-часовой шаг ≈ кВт)

    # ---------- основной шаг ----------
    def step(self):
        dt        = self.model.current_datetime
        pres      = self.model.num_home / self.model.num_people_agents

        kw_fixed  = self.LIGHT_KW + self.FAN_KW + self.IT_KW
        kw_pumps  = self.PUMP_KW if self._in_heating_season(dt) else 0.0
        kw_lifts  = self._lift_energy_kwh(pres)

        kw_total  = kw_fixed + kw_pumps + kw_lifts
        self.consumption = kw_total * 1_000     # Wh за час
