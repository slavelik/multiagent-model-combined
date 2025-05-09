import os
import pickle
import numpy as np
import pandas as pd
from mesa import Agent


class StudentAgent(Agent):
    """
    Агент-студент с прогнозируемыми состояниями и адаптивным расписанием.
    Агент получает готовый набор признаков из CSV (student_test_features.csv),
    загруженного в модель, и прогнозирует по ним.
    Прогнозируемые признаки:
      - hw_duration:            float (часы, домашнее задание)
      - sleep_start_hour:       int (час начала сна)
      - leisure_duration:       float (часы вечернего досуга)
      - commute_duration:       int (минуты поездки)
      - healthy:                bool (здоров или нет)
      - evening_study_duration: int (минуты вечернего обучения или работы)

    Этапы шага:
      1. predict_states()           — прогноз признаков ML-моделями
      2. compute_consumption_coef() — вычисление коэффициента энергопотребления (учитывая on_campus)
      3. build_schedule()           — построение расписания на основе прогнозов
      4. compute_consumption()      — расчет потребления за текущий час
    """
    # Классовый кэш: модели загружаем только один раз
    _models_loaded = False
    reg_hw = reg_sleep = reg_leisure = reg_commute = reg_evening = clf_healthy = None

    FEATURES = {
            "hw_duration":           ["course", "on_campus", "diligence", "exam_period", "holiday_flag"],
            "sleep_start_hour":      ["on_campus", "course", "exam_period", "holiday_flag"],
            "leisure_duration":      ["on_campus", "holiday_flag", "exam_period", "diligence"],
            "commute_duration":      ["on_campus", "course"],
            "evening_study_duration":["course", "exam_period", "diligence"],
            "healthy":               ["hw_duration", "evening_study_duration", "sleep_start_hour", "sleep_duration", "T_out"],
    }
    def __init__(self, model, agent_id, feature_cols=None):
        super().__init__(model)
        
        self.agent_id = agent_id
        self.feature_cols = feature_cols

        if not StudentAgent._models_loaded:
            base = os.path.dirname(__file__)
            tm = os.path.join(base, "ML", "trained_models")
            StudentAgent.reg_hw      = pickle.load(open(os.path.join(tm, "hw_duration/XGBoost.pkl"), "rb"))
            StudentAgent.reg_sleep   = pickle.load(open(os.path.join(tm, "sleep_start_hour/RandomForest.pkl"), "rb"))
            StudentAgent.reg_leisure = pickle.load(open(os.path.join(tm, "leisure_duration/ElasticNet.pkl"), "rb"))
            StudentAgent.reg_commute = pickle.load(open(os.path.join(tm, "commute_duration/XGBoost.pkl"), "rb"))
            StudentAgent.reg_evening = pickle.load(open(os.path.join(tm, "evening_study_duration/ElasticNet.pkl"), "rb"))
            StudentAgent.clf_healthy = pickle.load(open(os.path.join(tm, "healthy/XGBoost.pkl"), "rb"))
            StudentAgent._models_loaded = True

        self.reg_hw       = StudentAgent.reg_hw
        self.reg_sleep    = StudentAgent.reg_sleep
        self.reg_leisure  = StudentAgent.reg_leisure
        self.reg_commute  = StudentAgent.reg_commute
        self.reg_evening  = StudentAgent.reg_evening
        self.clf_healthy  = StudentAgent.clf_healthy

        # Изначальные прогнозируемые атрибуты со значениями по умолчанию
        self.hw_duration = None               # часы домашнего задания
        self.sleep_start_hour = None          # час начала сна
        self.leisure_duration = None          # часы вечернего досуга
        self.commute_duration = None          # минуты дороги
        self.evening_study_duration = None    # минуты вечернего обучения
        self.healthy = None                   # здоровье
        self.hospitalized = None              # госпитализация
        self.diligence = None                 # усердие
        self.consumption_coef = 1.0

        # Мощности приборов (Вт)
        self.P_LAPTOP    = self.random.randint(60, 100)         # ноутбук
        self.P_PC        = self.random.randint(100, 120)        # персональный компьютер
        self.P_KETTLE    = 1500                                 # средний чайник
        self.P_MICROWAVE = self.random.randint(600, 1000)       # средняя микроволновка 600 - 1000 Вт
        self.P_TV        = 100                                  # телевизор
        self.P_PHONE     = 5                                    # зарядка телефона
        self.P_LIGHTS    = 60                                   # освещение
        self.P_FRIDGE    = self.random.randint(350, 780) * 0.3  # холодильник
        self.P_WASHER    = self.random.randint(400, 1300)       # стиральная машина
        self.P_ADD       = self.random.randint(20, 50)          # ночник / лампа
        if self.random.random() < 0.6:
            self.P_STOVE     = self.random.randint(1000, 3000)  # электроплита
        else:
            self.P_STOVE     = self.random.randint(200, 400)

        # Коэффициент, влияющий на потребление (например, студенты на кампусе могут расходовать меньше за счет общих ресурсов)
        self.consumption_coef = 1.0

        self.schedule = []  # список событий: (начало в минутах, длительность, мощность, метка)
        self.consumption = 0

        self.course = None # курс
        self.on_campus = None # очно / онлайн

    def predict_states(self):
        return

    def compute_consumption_coef(self):
        """
        Устанавливает коэффициент энергопотребления.
        Например, студенты на кампусе могут иметь немного более экономное потребление.
        """
        self.econ_coef = 0.9 if self.on_campus else 1.0
        self.econ_coef *= self.model.economic_coeff

    def build_schedule(self):
        """
        Формирование расписания студента на день.
        Расписание состоит из событий (начало в минутах, длительность, мощность, метка)
        с учётом прогнозируемых признаков.
        """
        events = []
        wake_hour = (self.sleep_start_hour + int(self.random.normalvariate(9 - 0.5 * self.diligence, 1.5))) % 24
        minute = wake_hour * 60

        def add(dur, power, label):
            nonlocal minute
            power *= self.random.uniform(0.9, 1.1)
            events.append((minute, dur, power, label))
            minute += dur
        
        hourly = [0.0] * 24
        labels = [None] * 24
        schedule_complete = False
        total_duration = 0

        use_stove = (self.random.random() < 0.5)
        cook_power = self.P_STOVE if use_stove else self.P_MICROWAVE
        cook_label = "cook_on_stove" if use_stove else "microwave_prep"
        prep_duration = (
            self.random.randint(30, 91) if use_stove
            else 5
        )
        cook_label = "cook_on_stove" if use_stove else "microwave_prep"

        self.anomaly_type = None  # по умолчанию — нет отклонений
        
        if self.hospitalized:
            add(20*60, self.P_PHONE, "hospital_rest")
            self.schedule = events
            self.model.num_hospitalized += 1
            schedule_complete = True
            total_duration += 20*60
        elif not self.healthy:
            add(5, self.P_MICROWAVE, "breakfast_prep")
            add(self.random.randint(15, 60), 0, "light_breakfast")
            add(4*60, self.P_LAPTOP, "rest")
            add(2*60, self.P_PHONE, "rest_phone")
            add(10*60, 0, "extended_rest")
            self.model.num_unhealthy += 1
            total_duration += 16*60
            schedule_complete = True
            
        # Случайные аномалии (5% случаев)
        if self.random.random() < 0.05:
            anomaly_type = self.random.choice(["full_pc", "active_day", "sports_day"])
            start_min = max(minute, 9*60)
            # завтрак‑готовка
            add(prep_duration, cook_power, cook_label)
            add(self.random.randint(15, 60), 0, "breakfast")
            if anomaly_type == "full_pc":
                minute = start_min
                add(8 * 60, self.P_PC, "full_day_pc_session")
                total_duration += 8*60
            elif anomaly_type == "active_day":
                add(30, 0, "morning_walk")
                add(240, 0, "city_walk")
                add(90, self.P_PHONE + self.P_PC, "cafe_and_chill")
                total_duration += 6*60
            else:
                add(90, 0, "gym")
                add(60, self.P_PC, "post_gym_rest")
                total_duration += 150
            # ужин‑готовка
            add(prep_duration, cook_power, cook_label)
            add(60, 0, "dinner")
            add(2 * 60, self.P_PC, "evening_rest")
            self.anomaly_type = anomaly_type
            total_duration += 3*60
            # Подготовка ко сну
            sleep_minute = self.sleep_start_hour * 60
            if minute < sleep_minute:
                add(sleep_minute - minute, 0, "pre_sleep_prep")
            schedule_complete = total_duration >= 720
    
        if not schedule_complete:
            # Обычный день
            if self.random.random() < 0.7:  # 70% вероятность готовки
                add(prep_duration, cook_power, cook_label)
                add(self.random.randint(15, 60), 0, "breakfast")
            else:
                add(15, self.P_PHONE, "quick_breakfast")  # быстрый перекус с телефоном

            # Выходной день
            if self.model.is_holiday:
                # Больше свободного времени
                if self.random.random() < 0.5:
                    add(120, 0, "morning_walk")
                if self.random.random() < 0.3:
                    add(90, 0, "sports_time")
                leisure_minutes = int(self.leisure_duration * 60)
                add(leisure_minutes, self.P_PC, "weekend_leisure")
                add(prep_duration, cook_power, cook_label)
                add(60, 0, "dinner")
                add(120, 0, "evening_walk")
                add(120, self.P_WASHER, "laundry")
                sleep_minute = self.sleep_start_hour * 60
                if minute < sleep_minute:
                    add(sleep_minute - minute, 0, "pre_sleep_prep")

            # Учебный день
            if self.on_campus:
                n_pairs = 4 if self.course <= 2 else 3
                add(self.commute_duration, 0, "commute")
                for i in range(n_pairs):
                    add(90, 0, f"pairs")
                    add(60, self.P_LAPTOP, "break")
                add(self.commute_duration, 0, "commute")
            else:
                online_dur = int(8 * 60 * (1 + 0.2 * self.course/4))
                add(online_dur, self.P_LAPTOP, "online_lectures")

            # Ланч/обеденный перерыв
            add(60, self.P_PHONE, "lunch")

            # Выполнение домашнего задания (прогнозируемая длительность переводится в минуты)
            hw_minutes = int(self.hw_duration * 60 * 1.2)
            add(hw_minutes, self.P_LAPTOP, "homework")

            # Ужин
            add(prep_duration, cook_power, cook_label)
            if self.random.random() < 0.5:
                add(5, self.P_KETTLE, "boil_water")
            add(60, 0, "dinner")

            if self.random.random() < 0.5:
                add(60, self.P_PC, "evening_leisure")
            if not self.healthy:
                add(2*60, self.P_PC, "evening_rest")
            else:
                if self.on_campus:
                    add(self.evening_study_duration, self.P_PC, "evening_study")
                else:
                    add(int(self.leisure_duration * 60), self.P_PC, "leisure_evening")

            # Подготовка ко сну
            sleep_minute = self.sleep_start_hour * 60
            if minute < sleep_minute:
                add(sleep_minute - minute, 0, "pre_sleep_prep")
        
        exclude_labels = {'hospital_rest', 'morning_walk', 'gym', 'city_walk', 'cafe_and_chill', 'commute', 'sports_time', 'evening_walk'}
        hourly = [0.0] * 24
        labels = [None]*24
        for start_min, dur, power, label in events:
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
            hourly[hour] += self.P_FRIDGE + self.P_ADD
            if not labels[hour]:
                labels[hour] = 'idle'
        self.hourly_profile = hourly
        self.hourly_label   = labels

    def step(self):
        """
        В модели на current_hour == 0 уже:
          1) рассчитываются состояния в batch_predict_students()
          2) вызываются compute_econ_coef() и _build_schedule()
        """
        hour = self.model.current_hour
        self.consumption = self.hourly_profile[hour] * self.econ_coef

        label = self.hourly_label[hour]
        if label not in {
            "hospital_rest", "morning_walk", "gym", "city_walk", "cafe_and_chill",
            "commute", "sports_time", "evening_walk", 'pairs', 'commute', 'break'
        }:
            self.model.hourly_at_home += self.consumption
            self.model.num_home += 1

        # print(f"Student {self.agent_id} | course={self.course} | on_campus={self.on_campus} | anomaly={self.anomaly_type} | "
        #       f"leisure={self.leisure_duration:.2f} | sleep_start={self.sleep_start_hour} | commute={self.commute_duration} | "
        #       f"ev_st={self.evening_study_duration} | hw={self.hw_duration:.2f} | healthy={self.healthy} | hosp={self.hospitalized} | "
        #       f"hour {self.model.current_hour} | cons={self.consumption:.1f} Wh")