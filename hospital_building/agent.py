from mesa import Agent
import datetime as _dt

class HospitalBuildingAgent(Agent):
    """
    Агент «Больница» (электропотребление):
      – self.consumption — Wh за последний час.
      – Электропотребление = базовый норматив + насосы отопления (в сезон) +
        динамические надбавки:
          • от доли занятых коек (inpatients),
          • от доли амбулаторных приёмов (outpatients), нормированных на число приёмных боксов.
    """
    AREA_M2     = 8_000           # отапливаемая площадь, м²
    BEDS_TOTAL  = 320             # число коек (максимум госпитализаций)

    # Годовой норматив электроэнергии: 65 кВт·ч/м²·год → _EL_KWH_H кВт·ч/ч
    EUI_EL_BASE = 65.0
    _EL_KWH_H   = EUI_EL_BASE * AREA_M2 / 8_760.0

    # Насосы отопления
    PUMP_KW = 0.60                # кВт в отопительный сезон

    # Норма времени приёма одного пациента: 15 мин (Минздрав РФ № 290н от 02.06.2015) 
    VISIT_TIME_MIN = 15
    VISITS_PER_BOX_PER_HOUR = 60 / VISIT_TIME_MIN  # = 4 визита/ч

    def __init__(self, model):
        super().__init__(model)
        self.consumption = 0.0  # Вт·ч за последний час
        self._last_date      = None
        self._daily_occ      = 0.0
        self._daily_amb      = 0.0

    @staticmethod
    def _in_heating_season(dt: _dt.datetime) -> bool:
        m, d = dt.month, dt.day
        return (
            (m == 10 and d >= 15) or (11 <= m <= 12) or
            (1 <= m <= 3) or (m == 4 and d <= 15)
        )

    def step(self):
        dt    = self.model.current_datetime
        today = dt.date()

        # 1) При первом вызове или при смене даты — пересчитываем "дневные" коэффициенты.
        if self._last_date is None or today != self._last_date:
            # 1.1) Доля занятых коек
            self._daily_occ = int(self.model.num_hospitalized / 20) / self.BEDS_TOTAL

            # 1.2) Считаем число приёмно-смотровых боксов по СП 158.13330.2014 §6.7.1.10
            if   self.BEDS_TOTAL <=  60: n_boxes = 2
            elif self.BEDS_TOTAL <= 100: n_boxes = 3        
            else:                        n_boxes = 3 + ((self.BEDS_TOTAL - 100) // 50)

            # 1.3) Максимальная пропускная способность амбулаторного приёма
            HOURS_AMB_OPERATION = 12  # часы работы с 9 до 21
            VISITS_PER_BOX_PER_HOUR = 60 / 15  # 15 мин на приём → 4 визита/ч
            max_daily_amb = n_boxes * VISITS_PER_BOX_PER_HOUR * HOURS_AMB_OPERATION

            # 1.4) Доля амбулаторных приёмов от максимума
            self._daily_amb = min(int(self.model.num_unhealthy / 20), max_daily_amb) / max_daily_amb

            self._last_date = today

        # 2) В течение дня для каждого часового шага берём сохранённые "дневные" коэффициенты
        occ_ratio = self._daily_occ
        # амбулаторная нагрузка только в рабочие часы
        amb_ratio = self._daily_amb if 9 <= dt.hour < 21 else 0.0

        # 3) Базовое электропотребление + насосы (если отопительный сезон)
        pump_kWh = self.PUMP_KW if self._in_heating_season(dt) else 0.0
        base_el  = self._EL_KWH_H + pump_kWh  # кВт·ч за час

        # 4) Динамический множитель:
        #    – 56% от HVAC/освещения → зависит от occ_ratio
        #    – 20% от мед-оборудования     → зависит от amb_ratio
        dyn = 1.0 + 0.56 * occ_ratio + 0.20 * amb_ratio

        # 5) Итог (перевод кВт·ч → Вт·ч)
        el_kWh           = base_el * dyn
        self.consumption = el_kWh * 1_000.0

        print(f"amb_ratio = {amb_ratio} occ_ratio={occ_ratio} self.model.num_unhealthy={int(self.model.num_unhealthy / 10)} self.model.num_hospitalized={int(self.model.num_hospitalized / 10)} ")
