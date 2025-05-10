import numpy as np
from mesa import Agent

class OfficeBuildingAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        rng = np.random.RandomState(200 + self.unique_id)

        # Фиксированная площадь офиса (м²)
        # Норма на одного: 6.5 м²/чел (СНиП)
        self.area = float(rng.choice([500, 1000, 2000, 5000], p=[0.3,0.4,0.2,0.1]))
        self.capacity = self.area / 6.5  # вместимость в чел

        # Базовые удельные плотности (W/m²)
        self.heating_pump_density   = 0.05   # циркуляционный насос отопления
        self.heating_thermal_density = 22.8
        self.vent_fan_density       = 1.0    # вентиляционные вентиляторы 
        self.lighting_day_density   = 10.0   # офисное освещение днём 
        self.lighting_night_density = 1.0  # аварийное/безопасное освещение ночью
        self.per_pc_load            = 150.0  # W на ПК в рабочее время 

        self.consumption = 0

    def step(self):
        dt = self.model.current_datetime
        hour = dt.hour

        # 2) Определяем ppl
        ppl = int(self.model.num_at_office / self.model.num_office_agents)
        ppl = int(self.capacity) if ppl > self.capacity else ppl
        # 3) Отопительный сезон: 15 октября–15 апреля
        m, d = dt.month, dt.day
        heating_active = ((m == 10 and d >= 15) or (m > 10) or (m < 4) or (m == 4 and d <= 15))
        heating_load = self.area * self.heating_pump_density if heating_active else 0.0

        # 4) Ночная переработка (22:00–7:00) – потребление падает
        night = True if (hour >= 22 or hour < 7) else False
        vent_factor = 0.3 if night else 1.0   # 30% мощности вентиляторов ночью
        light_density = self.lighting_night_density if night else self.lighting_day_density


        # 5) Расчёт нагрузок
        ventilation_load = self.area * self.vent_fan_density * vent_factor
        lighting_load    = self.area * light_density
        plug_load        = ppl * self.per_pc_load

        # 6) Суммарная мгновенная мощность (W)
        self.consumption = heating_load + ventilation_load + lighting_load + plug_load

        # print(
        #     f"[Office {self.unique_id} | {dt}] "
        #     f"Heat={'on' if heating_active else 'off'}({heating_load:.0f}W), "
        #     f"Vent={ventilation_load:.0f}W, Light={lighting_load:.0f}W, "
        #     f"Plug={plug_load:.0f}W Total={self.consumption:.0f}W"
        #     f"people in ofice: {ppl}"
        # )
