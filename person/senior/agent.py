import os
import pickle
import pandas as pd
from mesa import Agent


class SeniorAgent(Agent):
    """
    Пожилой человек с прогнозируемыми состояниями и адаптивным расписанием.
    Предсказанные атрибуты:
      - healthy: bool
      - hospitalized: bool
      - mobility_level: int
      - tariff_change: float (% изменения тарифа)

    Шаг разделен на три этапа:
      1. predict_states()      — прогноз признаков ML-моделями
      2. build_schedule()      — построение расписания на основе состояний
      3. compute_consumption() — расчет потребления за текущий час
    """

    # Классовый кэш — модели будут загружены только при первом создании PersonAgent
    _models_loaded = False
    clf_healthy = None
    clf_hosp = None
    clf_mob = None
    reg_tariff = None

    def __init__(self, model, agent_id: int):
        super().__init__(model=model)
        self.agent_id = agent_id

        if not SeniorAgent._models_loaded:
            base = os.path.dirname(__file__)
            tm = os.path.join(base, "ML", "trained_models")
            SeniorAgent.clf_healthy = pickle.load(open(os.path.join(tm, "health_event/XGBoost.pkl"), "rb"))
            SeniorAgent.clf_hosp    = pickle.load(open(os.path.join(tm, "hospitalized/XGBoost.pkl"), "rb"))
            SeniorAgent.clf_mob     = pickle.load(open(os.path.join(tm, "mobility_level/LogisticRegression.pkl"), "rb"))
            SeniorAgent.reg_tariff  = pickle.load(open(os.path.join(tm, "tariff_change/RandomForest.pkl"), "rb"))
            SeniorAgent._models_loaded = True

        self.clf_healthy = SeniorAgent.clf_healthy
        self.clf_hosp    = SeniorAgent.clf_hosp
        self.clf_mob     = SeniorAgent.clf_mob
        self.reg_tariff  = SeniorAgent.reg_tariff

        # Прогнозируемые атрибуты
        self.healthy = False
        self.hospitalized = False
        self.mobility_level = 0
        self.tariff_change = 0.0

        # Мощности приборов (Вт)
        self.P_KETTLE    = 1500                                 # средний чайник
        self.P_MICROWAVE = self.random.randint(600, 1000)       # средняя микроволновка 600 - 1000 Вт
        self.P_TV        = 100                                  # телевизор
        self.P_PHONE     = 5                                    # зарядка телефона
        self.P_LIGHTS    = 60                                   # освещение
        self.P_FRIDGE    = self.random.randint(350, 780) * 0.3  # холодильник
        self.P_WASHER    = self.random.randint(400, 1300)       # стиральная машина
        if self.random.random() < 0.6:
            self.P_STOVE     = self.random.randint(1000, 3000)  # электроплита
        else:
            self.P_STOVE     = self.random.randint(200, 400)

        # Расписание и коэффициент
        self.schedule = []
        self.econ_coef = 1
        self.consumption = 0

    def predict_states(self):
        return

    def compute_consumption_coef(self):
        """
        Вычисление экономического коэффициента на основе tariff_change.
        """
        tc = self.tariff_change
        if tc > 10: self.econ_coef = 0.85
        elif tc > 5: self.econ_coef = 0.90
        elif tc > 0: self.econ_coef = 0.95
        else: self.econ_coef = 1.00
        self.econ_coef = self.econ_coef * self.model.economic_coeff

    def build_schedule(self):
        """
        Построение расписания на день в виде списка событий
        (start_minute, duration_min, power, label) на основе состояний.
        """
        ev = []
        anomaly_roll = self.random.random()
        self.anomaly_type = None  # по умолчанию — нет отклонений
        minute = int(self.random.normalvariate(7.5*60, 90))  # подъём ~6:00±1.5ч

        def add(dur, power, label):
            nonlocal minute
            ev.append((minute, dur, power, label))
            minute += dur

        if self.hospitalized:
            add(18*60, self.P_PHONE*0.2, "hospital_rest")
            self.model.num_hospitalized += 1
        elif not self.healthy:
            add(2*60, self.P_PHONE, "wake_charge")
            add(2*60, 0, "sleep_rest")
            add(4*60, self.P_TV, "TV_rest")
            add(3*60, self.P_PHONE, "extended_rest")
            add(2*60, 0, "sleep_rest")
            self.model.num_unhealthy += 1
        else:
            if anomaly_roll < 0.02:
                # Редкий случай бессонницы
                self.anomaly_type = "insomnia"
                add(2*60, self.P_PHONE, "wake_charge")
                add(6*60, self.P_TV, "night_insomnia_TV")
                add(60, self.P_STOVE, "midnight_cook")
                add(6*60, 0, "late_sleep")
            elif anomaly_roll < 0.04:
                # Отказ от прогулки и больше готовки
                self.anomaly_type = "no_walk_extra_cooking"
                add(60, 0, "sleep_rest")
                add(60, self.P_STOVE, "extra_cook")
                add(60, self.P_PHONE, "phone_call")
                add(3*60, self.P_TV, "afternoon_TV")
                add(60, self.P_STOVE, "dinner")
                add(3*60, self.P_TV, "evening_TV")
                add(6*60, 0, "night_sleep")
            elif anomaly_roll < 0.06:
                # Полный день в постели
                self.anomaly_type = "bed_rest"
                minute = 0
                add(16*60, self.P_PHONE*0.5, "bed_rest")
            else:
                # Нормальное расписание с гибким временем
                minute = int(self.random.normalvariate(360, 30))  # Пробуждение около 6:00 ± 30 минут

                # Утро: постепенный рост с 6 до 15
                add(30, 0, "wake_up")  # Пробуждение
                add(30, self.P_STOVE, "breakfast_cook")  # Завтрак
                add(30, 0, "breakfast_eat")

                # Утренняя прогулка: длиннее в выходные
                if self.model.is_holiday:
                    walk_dur = {0: 0, 1: 45, 2: 90, 3: 120}[self.mobility_level]
                else:
                    walk_dur = {0: 0, 1: 30, 2: 60, 3: 90}[self.mobility_level]
                if walk_dur > 0:
                    add(walk_dur, 0, "morning_walk")

                # Утренний ТВ: меньше в выходные
                if self.model.is_holiday:
                    tv_dur = int(self.random.normalvariate(90, 15))
                else:
                    tv_dur = int(self.random.normalvariate(120, 20))
                add(tv_dur, self.P_TV, "morning_TV")

                # Отдых до обеда
                rest_dur = int(self.random.normalvariate(60, 15))  # около 1 часа
                add(rest_dur, 0, "rest_before_lunch")

                # Обед: время варьируется
                cook_dur = int(self.random.normalvariate(30, 5))  # около 30 минут
                add(cook_dur, self.P_STOVE, "lunch_cook")
                add(30, 0, "lunch_eat")

                # Дневной ТВ
                tv_dur = int(self.random.normalvariate(120, 30))  # около 2 часов
                add(tv_dur, self.P_TV, "afternoon_TV")

                # Стирка: чаще в выходные
                if self.random.random() < 0.5:
                    add(60, self.P_WASHER, "laundry")
                else:
                    add(60, 0, "afternoon_rest")

                # Отдых до ужина
                rest_dur = int(self.random.normalvariate(60, 15))  # около 1 часа
                add(rest_dur, 0, "rest_before_dinner")

                # Ужин: время варьируется
                cook_dur = int(self.random.normalvariate(60, 20))
                add(cook_dur, self.P_STOVE, "dinner_cook")
                add(30, 0, "dinner")

                # Вечерний ТВ: меньше в выходные
                if self.model.is_holiday:
                    tv_dur = int(self.random.normalvariate(120, 30))
                else:
                    tv_dur = int(self.random.normalvariate(160, 30))
                add(tv_dur, self.P_TV, "evening_TV")

                # Зарядка телефона
                add(60, self.P_PHONE, "phone_charge")

                # Сон: до конца дня
                remaining = 1440 - minute
                if remaining > 0:
                    add(remaining, 0, "night_sleep")

        exclude_labels = {
            'hospital_rest', 'night_sleep', 'late_sleep', 'walk',
            'phone_before_bed', 'sleep_rest', 'rest_before_bed', 'morning_TV',
            'afternoon_TV', 'evening_TV', 'TV_rest', 'rest_before_lunch', 'afternoon_rest'
        }

        hourly = [0.0] * 24
        labels = [None]*24
        for start_min, dur, power, label in ev:
            # для каждого события распределяем энергию по пересечению с каждым часом
            ev_end = start_min + dur
            for hour in range(24):
                h_start = hour * 60
                h_end   = h_start + 60
                # сколько минут из события попадает в этот час
                overlap = max(0, min(ev_end, h_end) - max(start_min, h_start))
                if overlap > 0:
                    total_wh = power * (overlap / 60)
                    if label not in exclude_labels:
                        total_wh += self.P_LIGHTS * (overlap / 60)
                    hourly[hour] += total_wh
                    if not labels[hour]:
                        labels[hour] = label
        for hour in range(24):
            hourly[hour] += self.P_FRIDGE
            if not labels[hour]:
                labels[hour] = 'idle'
        self.hourly_profile = hourly
        self.hourly_label   = labels

    def compute_consumption(self):
        """
        Расчет потребления (Wh) за текущий час по self.schedule и econ_coef.
        """
        hour = self.model.current_hour
        self.consumption = self.hourly_profile[hour] * self.econ_coef

    def step(self):
        """
        В модели на current_hour == 0 уже:
          1) рассчитываются состояния в batch_predict_seniors()
          2) вызываются compute_econ_coef() и _build_schedule()
        """
        hour = self.model.current_hour
        self.consumption = self.hourly_profile[hour] * self.econ_coef

        label = self.hourly_label[hour]
        if label not in {"hospital_rest", "walk", "morning_walk"}:
            self.model.num_home += 1
            self.model.hourly_at_home += self.consumption

        # print(f"Senior {self.agent_id}  | healthy={self.healthy} | "
        #       f"hosp={self.hospitalized} | mobility={self.mobility_level} | "
        #       f"tariff={self.tariff_change:.1f} | anomaly={self.anomaly_type} | "
        #       f"hour={self.model.current_hour} | cons={self.consumption:.1f} Wh")
