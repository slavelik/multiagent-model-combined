import os
import pickle
import pandas as pd
from mesa import Agent
import numpy as np


class PersonAgent(Agent):
    """
    Агент, моделирующий обычного человека с прогнозируемыми параметрами и адаптивным расписанием.
    Агент получает готовый набор признаков из CSV-файла, прогнозирует по ним свои параметры.
    Прогнозируемые параметры:
      - occupation: 'office' | 'remote' | 'shift' | 'retired'
      - family_size: int
      - has_kids: bool
      - movie_enthusiasm: float [0,1]
      - socialness: float [0,1]
      - weekend_relax_factor: float >=1
      - healthy: float [0,1]
      - hospitalized: bool
    """
    # Классовый кэш — модели будут загружены только при первом создании PersonAgent
    _models_loaded = False
    models = {}

    FEATURES = {
        "has_kids":             ["age","family_size","region","income_level"],
        "weekend_relax_factor": ["socialness","occupation","family_size"],
        "movie_enthusiasm":     ["socialness","tv_time","evening_activity_duration"],
        "socialness":           ['age', 'sport_activity', 'weekend_outdoor_time'],
        "occupation":           ['age','income_level','region','socialness', 'family_size', 'has_kids', 'commute_duration'],
        "family_size":          ["age","region","income_level","socialness"],
        "healthy":              ["age", "socialness", "occupation", "T_out", "region", "sport_activity"],
        "hospitalized":         ["healthy","age","T_out","sin_month","cos_month"]
    }

    def __init__(self, model, agent_id):
        super().__init__(model)
        self.agent_id = agent_id

        self.classification_targets = ["occupation", "has_kids", "hospitalized", "healthy"]
        self.regression_targets = ["family_size", "movie_enthusiasm", "socialness", "weekend_relax_factor"]

        if not PersonAgent._models_loaded:
            base = os.path.dirname(__file__)
            tm   = os.path.join(base, "ML", "trained_models")
            PersonAgent.models["occupation_pipeline"]  = pickle.load(open(os.path.join(tm, "occupation","occupation_pipeline.pkl"), "rb"))
            PersonAgent.models["occupation_encoder"]   = pickle.load(open(os.path.join(tm, "occupation","occupation_encoder.pkl"), "rb"))
            PersonAgent.models["hospitalized"]         = pickle.load(open(os.path.join(tm, "hospitalized","XGBoost.pkl"), "rb"))
            PersonAgent.models["has_kids"]             = pickle.load(open(os.path.join(tm, "has_kids","RandomForest.pkl"), "rb"))
            PersonAgent.models["movie_enthusiasm"]     = pickle.load(open(os.path.join(tm, "movie_enthusiasm","Ridge.pkl"), "rb"))
            PersonAgent.models["socialness"]           = pickle.load(open(os.path.join(tm, "socialness","GradientBoosting.pkl"), "rb"))
            PersonAgent.models["weekend_relax_factor"] = pickle.load(open(os.path.join(tm, "weekend_relax_factor","CatBoost.pkl"), "rb"))
            PersonAgent.models["healthy_pipeline"]  = pickle.load(open(os.path.join(tm, "healthy","healthy_pipeline.pkl"), "rb"))
            PersonAgent.models["healthy_encoder"]   = pickle.load(open(os.path.join(tm, "healthy","healthy_encoder.pkl"), "rb"))
            PersonAgent._models_loaded = True

        self.models = PersonAgent.models

        # self.occupation_map = {0: 'office', 1: 'remote', 2: 'shift', 3: 'retired', 4: 'consultant'}
        self.occupation = None
        self.hospitalized = None
        self.has_kids = None
        self.healthy = None
        self.family_size = None
        self.movie_enthusiasm = None
        self.socialness = None
        self.weekend_relax_factor = None
        
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
        
        if self.random.random() < 0.6:
            self.P_STOVE     = self.random.randint(1000, 3000)  # электроплита
        else:
            self.P_STOVE     = self.random.randint(200, 400)

        # Состояние агента
        self.schedule = []
        self.consumption = 0

    def predict_states(self):
        return

    def compute_consumption_coef(self):
        """
        Вычисление коэффициента энергопотребления.
        Используется экономический коэффициент из модели.
        """
        self.consumption_coef = self.model.economic_coeff

    def build_schedule(self):
        events = []
        base_wake = int(self.random.normalvariate(8*60, 120)) 
        if self.model.is_holiday:
            wake_shift = self.random.randint(0, 240)
        else:
            wake_shift = 0
        minute = int(self.random.normalvariate(base_wake + wake_shift, 60))

        def add(duration_min, power, label):
            nonlocal minute
            power *= self.random.uniform(0.9, 1.1)
            events.append((minute, duration_min, power, label))
            minute += duration_min
        
        hourly = [0.0] * 24
        labels = [None] * 24
        schedule_complete = False
        total_duration = 0

        if self.hospitalized:
            add(18*60, self.P_PHONE*0.2, "hospital_rest")
            self.healthy = False
            self.model.num_hospitalized += 1
            total_duration += 18*60
            schedule_complete = True
        # Нездоров
        elif not self.healthy:
            self.model.num_unhealthy += 1
            add(4*60, self.P_PHONE,     "sleep_rest")
            add(2*60, self.P_TV,        "TV_rest")
            add(7*60, self.P_PHONE,     "extended_rest")
            add(2*60, self.P_PHONE,     "sleep_rest")
            total_duration += (4*60 + 2*60 + 7*60 + 2*60)
            schedule_complete = total_duration >= 720
        # выбросы в 8% случаев
        if self.random.random() < 0.08:
            choice = self.random.choice(['insomnia', 'nap_afternoon', 'sport_day', 'gaming_day'])
            if choice == 'insomnia':
                # бессонница: всю ночь смотрит ТВ
                add(6*60, self.P_TV, 'late night TV')
                total_duration += 6*60
            elif choice == 'nap_afternoon':
                # дневной сон после завтрака
                add(30 + 3*self.family_size, self.P_STOVE, 'morning cooking')
                add(2*60, 0, 'afternoon nap')
                total_duration += (30 + 3*self.family_size + 2*60)
            elif choice == 'sport_day':
                # спорт вместо работы
                add(60, 0, 'morning warmup')
                add(3*60, 0, 'gym workout')
                total_duration += (60 + 3*60)
            else:
                # игровой марафон
                add(8*60, self.P_PC, 'gaming_day')
                total_duration += 8*60
            schedule_complete = total_duration >= 720
        
        if not schedule_complete:
            # 1) Утро
            add(20 + 3*self.family_size, self.P_STOVE, 'morning cooking')
            add(3, self.P_KETTLE, 'boil water')
            if self.has_kids and self.random.random()<0.5:
                add(20, 0, 'prepare kids')

            # 2) Работа или учёба
            if not self.model.is_holiday:
                if self.occupation == 'office':
                    add(self.random.randint(30, 60), 0, 'commute to office')
                    add(8 * 60, 0, 'office work')
                elif self.occupation == 'remote':
                    add(8 * 60, self.P_PC, 'remote work')
                elif self.occupation == 'shift':
                    shift_start = self.random.choice([0, 6, 14]) * 60
                    if minute < shift_start:
                        add(shift_start - minute, 0, 'idle before shift')
                    add(8 * 60, self.P_PC, 'shift work')
                else:  # retired
                    add(8 * 60, 0, 'retired activities')
            else:
                # Выходной
                choice = self.random.choices(['tv', 'pc', 'outdoor'], weights=[0.4,0.3,0.3])
                if choice == 'tv':
                    add(self.random.randint(60,120), self.P_TV, 'weekend TV')
                elif choice == 'pc':
                    add(self.random.randint(60,120), self.P_PC, 'weekend PC')
                else:
                    add(self.random.randint(60,120), 0, 'weekend outdoor activities')

                if self.weekend_relax_factor > 1.15:
                    add(self.random.randint(60, 120), self.P_TV, 'weekend TV')
                elif self.socialness > 0.5:
                    add(self.random.randint(120, 240), 0, 'weekend outdoor activities')
                if self.random.random() < 0.5:
                    add(self.random.randint(0, 90), self.P_TV, 'weekend TV')
                else:
                    add(self.random.randint(60, 180), 0, 'weekend hobbies')

            # 3) Обеденный перерыв
            add(60, self.P_PHONE, 'lunch')

            # Короткие дневные активности
            if self.random.random() < 0.2:
                add(30, self.P_TV, 'daytime TV')
            if self.random.random() < 0.1:
                add(30, 0, 'social call')
            if self.random.random() < 0.1:
                add(45, 0, 'afternoon walk')

            # 4) Вечерний досуг
            leisure_duration = int(90 * (self.weekend_relax_factor if self.model.is_holiday else 1.0))
            if self.random.random() < 0.3:
                add(90, self.P_WASHER, 'laundry')
            if self.movie_enthusiasm > 0.55:
                add(leisure_duration, self.P_TV, 'movie time')
            elif self.socialness > 0.50 and self.model.is_holiday:
                add(leisure_duration, 0, 'social outing')
            else:
                add(leisure_duration, self.P_STOVE, 'evening cooking')

            # 5) Подготовка ко сну и ночь
            prep_duration = 30
            add(prep_duration, 0, 'pre-sleep routine')

        exclude_labels = {
            'hospital_rest', 'social outing', 'afternoon walk',
            'idle before shift', 'gym workout', 'gaming_day',
            'afternoon nap', 'weekend outdoor activities', 'boil water'
        }

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
                    if label == 'laundry':
                        total_wh += power * 0.5 * (overlap / 60)
                    elif 'cooking' in label:
                        total_wh += power * 0.6 * (overlap / 60)
                    hourly[hour] += total_wh
                    if not labels[hour]:
                        labels[hour] = label
        for hour in range(24):
            hourly[hour] += self.P_FRIDGE
            if not labels[hour]:
                labels[hour] = 'idle'

        self.hourly_profile = hourly
        self.hourly_label   = labels
        # print(f"DEBUG occupation for agent={self.agent_id}")
        # print(self.occupation)

    def step(self):
        hour = self.model.current_hour
        self.consumption = self.hourly_profile[hour] * self.consumption_coef
        
        label = self.hourly_label[hour]
        if label == "office work":
            self.model.num_at_office += 1
        elif label not in {
            "hospital_rest", "social outing", "afternoon walk",
            "idle before shift", "gym workout",
            "weekend outdoor activities"
        }:
            self.model.hourly_at_home += self.consumption
            self.model.num_home += 1

        # occ = self.occupation_map.get(self.occupation, self.occupation)
        # print(
        #     f"Person {self.agent_id}  | occupation={occ} | hospitalized = {self.hospitalized} | healthy={self.healthy} | "
        #     f"family_size={self.family_size} | kids={self.has_kids} | hour={self.model.current_hour} | movie_ent={self.movie_enthusiasm:.2f} | "
        #     f"socialness={self.socialness:.2f} | wrf={self.weekend_relax_factor} | cons={self.consumption:.1f} Wh"
        # )