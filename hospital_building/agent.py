from mesa import Agent

class HospitalBuildingAgent(Agent):
    """
    Агент «Больница»:
      – self.consumption хранит итоговое энергопотребление за последний час (Вт·ч).
      – Отопление рассчитывается по удельному годовому нормативу (СП 50).
      – Электропотребление — базовый уровень (аудиты РФ) + динамика от загрузки койко-мест и числа пациентов.
    """
    # Параметры здания
    AREA_M2       = 8000    # площадь больницы, м²
    BEDS_TOTAL    = 320     # всего коек

    # Удельное годовое потребление тепла:
    #   СП 50 «Требования к теплоснабжению»: qₒₚ = 0.024·ГСОП, где ГСОП≈4500 → 108 кВт·ч/м²·год
    EUI_HEAT_BASE = 108.0   # кВт·ч/м²·год

    # Удельное годовое потребление эл-энергии:
    #   Российские энергоаудиты: 45–80 → берем среднее 65 кВт·ч/м²·год (EIA CBECS 2012)
    EUI_EL_BASE   = 65.0    # кВт·ч/м²·год

    # Часы отопительного сезона (15 окт–15 апр ≈183 дн × 24 ч)
    HEAT_HOURS    = 183 * 24

    # Часовые нормативы, кВт·ч/ч
    _HEAT_KWH_H   = EUI_HEAT_BASE * AREA_M2 / HEAT_HOURS
    _EL_KWH_H     = EUI_EL_BASE   * AREA_M2 / 8760.0

    def __init__(self, model):
        super().__init__(model)
        self.consumption = 0.0  # Вт·ч за последний час

    def step(self):
        dt = self.model.current_datetime

        # --- Загрузка койко-мест и пациентов ---
        # occ_ratio = доля занятых коек
        hospitalized = self.model.num_hospitalized
        occ_ratio    = hospitalized / self.BEDS_TOTAL

        # pat_ratio = доля «активных» больных (операции, диагностика)
        patients     = self.model.num_unhealthy
        pat_ratio    = min(patients, self.BEDS_TOTAL) / self.BEDS_TOTAL

        # --- Отопление ---
        month, day = dt.month, dt.day
        in_heat = (
            (month == 10 and day >= 15) or
            (11 <= month <= 12) or
            (1 <= month <= 3) or
            (month == 4 and day <= 15)
        )
        # равномерно по всем часам сезона → _HEAT_KWH_H
        heat_kWh = self._HEAT_KWH_H if in_heat else 0.0

        # --- Электропотребление базовое ---
        base_el = self._EL_KWH_H

        # --- Динамические коэффициенты ---
        # 1) ОВиК + освещение: ~56 % от общего электропотребления больницы
        #    (EIA CBECS 2012, см. HVAC share ≈55–60 %)
        #    → нагрузка пропорциональна ocuppancy (occ_ratio)
        # 2) Медоборудование и операции: ~20 % от электропотребления
        #    (ISO EPIA, мед.ингредиенты) → пропорц. числу пациентов (pat_ratio)
        dynamic_factor = 1.0 + 0.56 * occ_ratio + 0.20 * pat_ratio

        el_kWh = base_el * dynamic_factor

        # --- Итог и перевод в Вт·ч ---
        total_kWh        = heat_kWh + el_kWh
        self.consumption = total_kWh * 1000.0

        # подробный лог (для отладки)
        # print(
        #     f"[Hospital {self.unique_id} | {dt:%Y-%m-%d %H}] "
        #     f"occ={occ_ratio:.2f}  pat={patients}  "
        #     f"heat={heat_kWh:.2f}kWh  el={el_kWh:.2f}kWh  "
        #     f"total={total_kWh:.2f}kWh"
        # )
