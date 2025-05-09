import os
import pickle
import random
import time
from mesa import DataCollector, Model
from mesa.agent import AgentSet
import numpy as np
import pandas as pd

from person.ML.create_features_for_testing import generate_person_features
from person.agent import PersonAgent
from person.senior.ML.create_features_for_testing import generate_senior_test_features
from person.senior.agent import SeniorAgent
from person.student.ML.create_features_for_testing import generate_student_test_features
from person.student.agent import StudentAgent
from transport.ev.ML.create_features_for_testing import generate_ev_test_features
from transport.ev.agent import ElectricCarAgent
from generate_hourly_features import generate_hourly_features

from enterprise_building.agent import EnterpriseBuildingAgent
from office_building.agent import OfficeBuildingAgent
from hospital_building.agent import HospitalBuildingAgent
from mall_building.agent import MallAgent
from modern_residential_building.agent import ModernResidentialBuildingAgent
from residential_building.agent import ResidentialBuildingAgent

class MultiAgentModel(Model):
    """
    Мультиагентная модель, где каждый агент получает персональные фичи
    и глобальные параметры времени для синхронизации, с учетом экономического корректирующего коэффициента.  
    """
    def __init__(
        self,
        n_persons: int,
        n_students: int,
        n_seniors: int,
        n_evs: int,
        n_enterprises: int,
        n_offices: int,
        n_hospitals: int,
        n_malls: int,
        n_modern_residential: int,
        n_residential: int,
        n_days: int,
        senior_test_file: str,
        student_test_file: str,
        person_test_file: str,
        evs_test_file: str,
        correction_factors_file: str,
        global_features_file: int,
        seed=None,
        hours_per_day: int = 24
    ):
        super().__init__(seed=seed)

        # 1) глобальный временной ряд и переменные для синхронизации шагов
        self.feature_df = pd.read_csv(global_features_file, parse_dates=["datetime"])
        self.feature_len = len(self.feature_df)
        self.step_count = 0

        # 2) генерация фич для каждого типа агентов
        generate_senior_test_features(
            n_days=n_days,
            hours_per_day=hours_per_day,
            num_seniors=n_seniors,
            seed=seed or 42,
            output_path=senior_test_file
        )
        generate_student_test_features(
            n_days=n_days,
            hours_per_day=hours_per_day,
            num_students=n_students,
            seed=seed or 42,
            output_path=student_test_file
        )
        generate_person_features(
            n_days=n_days,
            hours_per_day=hours_per_day,
            num_persons=n_persons,
            seed=seed or 42,
            output_path=person_test_file
        )
        generate_ev_test_features(
            n_days=n_days,
            hours_per_day=hours_per_day,
            num_evs=n_evs,
            seed=seed or 42,
            output_path=evs_test_file
        )

        self.senior_test_features = pd.read_csv(senior_test_file, parse_dates=["datetime"])
        self.student_test_features = pd.read_csv(student_test_file, parse_dates=["datetime"])
        self.person_test_features = pd.read_csv(person_test_file, parse_dates=["datetime"])
        self.ev_test_features = pd.read_csv(evs_test_file, parse_dates=["datetime"])

        # 3) инициируем глобальные
        print("Инициализация глобальных переменных.")
        first = self.feature_df.iloc[0]
        self.current_datetime    = first["datetime"]
        self._last_date          = self.current_datetime.date()
        self.current_hour        = int(first["hour"])
        self.current_day_of_week = int(first["day_of_week"])
        self.current_month       = int(first["month"])
        self.is_holiday          = bool(first["day_off"])
        self.current_T_out       = float(first["T_out"])
        self.num_hospitalized    = 0     # количество госпитализированных
        self.num_unhealthy       = 0     # количество нездоровых
        self.num_at_office       = 0     # сколько сейчас в офисе
        self.num_home            = 0     # сколько сейчас дома
        self._last_hour          = None  # для сброса раз в час
        self.num_office_agents   = n_offices
        self.total_consumption   = 0
        self.hourly_at_home      = 0

        base = os.path.dirname(__file__)
        tm = os.path.join(base, 'transport', 'ev', 'ML', 'enhanced_model')
        self.ev_state_model = pickle.load(open(os.path.join(tm, 'XGBoost.pkl'), 'rb'))
        
        print("Инициализация агентов.")
        students = [StudentAgent(self, i) for i in range(n_students)]
        persons  = [PersonAgent(self, i) for i in range(n_persons)]
        seniors  = [SeniorAgent(self, i) for i in range(n_seniors)]
        evs      = [ElectricCarAgent(self, i) for i in range(n_evs)]
        self.by_type = {
            'student': students,
            'person':  persons,
            'senior':  seniors,
            'ev':      evs,
        }

        print("Инициализация людей и транспорта завершена.")
        enterprises = [EnterpriseBuildingAgent(self) for _ in range(n_enterprises)]
        offices     = [OfficeBuildingAgent(self) for _ in range(n_offices)]
        hospitals   = [HospitalBuildingAgent(self) for _ in range(n_hospitals)]
        malls       = [MallAgent(self) for _ in range(n_malls)]
        modern_residential = [ModernResidentialBuildingAgent(self) for _ in range(n_modern_residential)]
        residential = [ResidentialBuildingAgent(self) for _ in range(n_residential)]
        print("Инициализация зданий завершена.")

        self.agents_buildings = AgentSet(agents=enterprises + offices + hospitals + malls + modern_residential + residential, random=self.random)
        self.population = AgentSet(agents=persons + students + seniors + evs, random=self.random)
        print("Созданы агент-сеты")

        # 4) Загружаем предвычисленные корректирующие коэффициенты
        print("Загрузка корректирующих коэффициентов.")
        with open(correction_factors_file, "rb") as f:
            self.correction_factors = pickle.load(f)
        self.current_year = None
        self.economic_coeff = 1.0

        # 5) Инициализация DataCollector
        print("Инициализация DataCollector.")
        self.datacollector = DataCollector(
            model_reporters={
                "Date": lambda m: m.current_datetime,
                "Hour": lambda m: m.current_hour,
                "Total Consumption": lambda m: m.total_consumption,
                "hospitalized": lambda m: m.num_hospitalized,
                "patients_total": lambda m: m.num_unhealthy,
                "office_population": lambda m: m.num_at_office,
                "presence_in_building": lambda m: m.num_home,
                "hourly_at_home": lambda m: m.hourly_avg_home
            },
            agent_reporters={
                "Type": lambda a: type(a).__name__.lower().replace("agent", ""),
                "Consumption": lambda a: a.consumption,
            }
        )

    def step(self):
        print("Дата:", self.current_datetime)
        row = self.feature_df.iloc[self.step_count % self.feature_len]
        self.current_datetime    = row["datetime"]
        self.current_hour        = int(row["hour"])
        self.current_day_of_week = int(row["day_of_week"])
        self.current_month       = int(row["month"])
        self.is_holiday          = bool(row["day_off"])
        self.current_T_out       = float(row["T_out"])

        # Сброс параметров раз в день
        current_date = self.current_datetime.date()
        if current_date != self._last_date:
            self.num_hospitalized = 0
            self.num_unhealthy    = 0
            self.num_at_office    = 0
            self._last_date       = current_date
        
        # Чтобы обновлять каждый час
        self.num_home = 0
        self.hourly_at_home = 0
        
        # Обновляем год и корректирующий коэффициент, если год изменился
        current_year = self.current_datetime.year
        if current_year != self.current_year:
            self.current_year = current_year
            # Используем коэффициент для текущего года или 1.0, если года нет в данных
            self.economic_coeff = self.correction_factors.get(current_year, 1.0)

        if self.current_hour == 0:
            self.batch_predict_seniors()
            self.batch_predict_persons()
            self.batch_predict_students()
            self.batch_predict_evs()
            for typ, agents in self.by_type.items():
                if typ != 'ev':
                    for agent in agents:
                        agent.compute_consumption_coef()
                        agent.build_schedule()

        self.step_count += 1
        self.population.do("step")
        self.agents_buildings.do("step")
        self.hourly_avg_home = self.hourly_at_home / self.num_home
        self.total_consumption = sum(a.consumption for a in self.population) + sum(a.consumption for a in self.agents_buildings)
        self.datacollector.collect(self)
        print(f"num_at_office={self.num_at_office}")
        # print(f"num_at_office={self.num_office} | num_home={self.num_home} | hourly_at_home={self.hourly_at_home} | hourly_avg={self.hourly_avg_home}")
        # print(f"num_hospitalized={self.num_hospitalized} | num_unhealthy={self.num_unhealthy} | num_office={self.num_office} | num_home={self.num_home}")

    def batch_predict_seniors(self):
        if not self.by_type.get('senior') or len(self.by_type['senior']) == 0:
            return
        now = self.current_datetime
        # отфильтровываем строки для всех агентов на этот час
        df = self.senior_test_features
        df0 = df[df['datetime'] == now].set_index('agent_id')
        
        # 1) unhealthy
        feat_h = [
            "T_out", "monthly_income", "debt_to_income_ratio",
            "diabetes", "hypertension", "cvd",
            "diabetes_meds", "hypertension_meds", "cvd_meds",
            "living_alone", "has_caregiver"
        ]
        X_h = df0[feat_h].copy()
        X_h['T_out'] = self.current_T_out
        probs_unhealthy = SeniorAgent.clf_healthy.predict_proba(X_h)[:, 1]
        
        # 2) hospitalized
        df_hp = X_h.copy()
        df_hp['health_event'] = (probs_unhealthy < 0.05).astype(int)
        cols_hp = [
            'health_event', 'T_out', 'diabetes', 'hypertension', 'cvd',
            'living_alone', 'has_caregiver'
        ]
        X_hp = df_hp[cols_hp]
        probs_hosp = SeniorAgent.clf_hosp.predict_proba(X_hp)[:,1]
        
        # 3) mobility
        df_m = df_hp.copy()
        df_m['hospitalized'] = (probs_hosp >= 0.05).astype(int)
        df_m['day_of_week'] = self.current_day_of_week
        cols_m = [
            'day_of_week', 'T_out', 'health_event', 'hospitalized',
            'living_alone', 'has_caregiver'
        ]
        X_m = df_m[cols_m]
        probs_m = SeniorAgent.clf_mob.predict_proba(X_m)
        
        # 4) tariff_change
        X_t = df0[['month']]
        base_tariffs = SeniorAgent.reg_tariff.predict(X_t)
        
        # Раздаём результаты агентам
        for agent in self.by_type['senior']:
            aid = agent.agent_id
            p_un = probs_unhealthy[aid]
            agent.healthy = (p_un < 0.05)
            
            p_hp = probs_hosp[aid]
            agent.hospitalized = (p_hp >= 0.05)
            if agent.hospitalized:
                agent.healthy = False
            
            # mobility: рандомный выбор по весам
            weights = probs_m[aid]
            classes = SeniorAgent.clf_mob.classes_
            agent.mobility_level = int(self.random.choices(classes, weights=weights, k=1)[0])
            
            t0 = float(base_tariffs[aid])
            agent.tariff_change = max(0.0, t0 + self.random.normalvariate(0,1.0))
    
    def batch_predict_persons(self):
        if not self.by_type.get('person') or len(self.by_type['person']) == 0:
            return
        now = self.current_datetime
        df = self.person_test_features
        df0 = df[df['datetime'] == now].set_index('agent_id')

        base_cols = [
            'age', 'gender', 'income_level', 'family_size', 'region',
            'socialness', 'education_level', 'occupation', 'sport_activity',
            'weekend_outdoor_time', 'has_kids', 'tv_time',
            'evening_activity_duration', 'sleep_start_hour', 'healthy',
            'commute_duration', 'cooking_time', 'movie_enthusiasm',
            'weekend_relax_factor', 'hospitalized'
        ]
        dem = pd.get_dummies(df0[base_cols], drop_first=True)
        dem['T_out']     = self.current_T_out
        dem['sin_month'] = np.sin(2 * np.pi * (self.current_month - 1) / 12)
        dem['cos_month'] = np.cos(2 * np.pi * (self.current_month - 1) / 12)

        results = {}
        for param, md in PersonAgent.models.items():
            if param in ["occupation_pipeline", "occupation_encoder"]:
                continue
            feats = PersonAgent.FEATURES[param]
            for f in feats:
                if f not in dem:
                    dem[f] = 0
            Xp = dem[feats]
            if param in {'occupation', 'has_kids', 'healthy', 'hospitalized'}:
                results[param] = md.predict(Xp)
            else:
                results[param] = md.predict(Xp)
        
        occ_pipe = PersonAgent.models["occupation_pipeline"]
        occ_le   = PersonAgent.models["occupation_encoder"]
        occ_feats = PersonAgent.FEATURES["occupation"]
        occ_pred_raw = occ_pipe.predict(df0[occ_feats])
        results["occupation"] = occ_le.inverse_transform(occ_pred_raw)

        for agent in self.by_type['person']:
            aid = agent.agent_id
            # классификация
            agent.occupation          = results['occupation'][aid]
            agent.has_kids            = bool(results['has_kids'][aid])
            agent.healthy             = bool(results['healthy'][aid])
            agent.hospitalized        = bool(results['hospitalized'][aid])
            # шум
            agent.movie_enthusiasm     = float(np.clip(results['movie_enthusiasm'][aid]
                                               + self.random.normalvariate(0,0.05), 0,1))
            agent.socialness           = float(np.clip(results['socialness'][aid]
                                               + self.random.normalvariate(0,0.05), 0,1))
            agent.weekend_relax_factor = max(1.0, results['weekend_relax_factor'][aid]
                                               + self.random.normalvariate(0,0.1))
            agent.family_size = self.random.choices(
                [1, 2, 3, 4, 5, 6],
                weights=[0.35, 0.3, 0.2, 0.05, 0.05, 0.05], k=1
            )[0]
    
    def batch_predict_students(self):
        if not self.by_type.get('student') or len(self.by_type['student']) == 0:
            return
        now = self.current_datetime
        df = self.student_test_features
        df0 = df[df['datetime'] == now].set_index('agent_id')

        # основные колонки, без дамми
        X = df0[['exam_period', 'course', 'on_campus', 'diligence', 'sleep_duration']].copy()
        X['T_out']        = self.current_T_out
        X['holiday_flag'] = self.is_holiday
        X['hour']         = self.current_hour
        X['day_of_week']  = self.current_day_of_week

        # 1) hw_duration
        X_hw = X[StudentAgent.FEATURES['hw_duration']]
        base_hw = self.by_type['student'][0].reg_hw.predict(X_hw)
        hw = [ max(0.5, b + self.random.normalvariate(0,0.2)) for b in base_hw ]

        # 2) leisure_duration
        X_le = X[StudentAgent.FEATURES['leisure_duration']]
        base_le = self.by_type['student'][0].reg_leisure.predict(X_le)
        le = [ max(0.5, b + self.random.normalvariate(0,0.2)) for b in base_le ]

        # 3) commute_duration
        X_cm = X[StudentAgent.FEATURES['commute_duration']]
        base_cm = self.by_type['student'][0].reg_commute.predict(X_cm)
        cm = [ max(0, int(round(b + self.random.normalvariate(0,5)))) for b in base_cm ]

        # 4) evening_study_duration
        X_ev = X[StudentAgent.FEATURES['evening_study_duration']]
        base_ev = self.by_type['student'][0].reg_evening.predict(X_ev)
        evd = [ max(10, int(round(b + self.random.normalvariate(0,10)))) for b in base_ev ]

        # 5) sleep_start_hour
        X_sl = X[StudentAgent.FEATURES['sleep_start_hour']]        
        base_sl = self.by_type['student'][0].reg_sleep.predict(X_sl)
        ssh = [ int(round(b + self.random.normalvariate(0,1))) % 24 for b in base_sl ]

        # 6) healthy
        hl_data = {
            'hw_duration': hw,
            'evening_study_duration': evd,
            'sleep_start_hour': ssh,
            'sleep_duration': df0['sleep_duration'].values,
            'T_out': [self.current_T_out] * len(df0)
        }
        hl = pd.DataFrame(hl_data, index=df0.index)
        healthy = self.by_type['student'][0].clf_healthy.predict(hl) 

        for agent in self.by_type['student']:
            aid = agent.agent_id
            agent.hw_duration            = hw[aid]
            agent.leisure_duration       = le[aid]
            agent.commute_duration       = cm[aid]
            agent.evening_study_duration = evd[aid]
            agent.sleep_start_hour       = ssh[aid]
            agent.healthy                = bool(healthy[aid])
            agent.hospitalized           = bool(df0.loc[aid, 'hospitalized'])
            agent.course                 = int(df0.loc[aid, 'course'])
            agent.on_campus              = bool(df0.loc[aid, 'on_campus'])
            agent.diligence              = float(df0.loc[aid, 'diligence'])

    def batch_predict_evs(self):
        if not self.by_type.get('ev') or len(self.by_type['ev']) == 0:
            return
        today = self.current_datetime.normalize()
        mask = (
            (self.ev_test_features['datetime'] >= today) &
            (self.ev_test_features['datetime'] < today + pd.Timedelta(days=1))
        )
        day_df = self.ev_test_features.loc[mask].sort_values(['agent_id', 'datetime'])
        
        X = pd.DataFrame({
            'month': self.current_month,
            'holiday': int(self.is_holiday),
            'hour': day_df['datetime'].dt.hour,
            'battery_capacity_wh': day_df['battery_capacity_wh'],
            'current_charge_wh': day_df['current_charge_wh'],
            'planned_distance_km': day_df['planned_distance_km'],
            'efficiency_wh_per_km': day_df['efficiency_wh_per_km'],
            'charging_preference_fast': (day_df['charging_preference'] == 'fast').astype(int),
            'charging_preference_midday': (day_df['charging_preference'] == 'midday').astype(int),
            'charging_preference_overnight': (day_df['charging_preference'] == 'overnight').astype(int),
        }, index=day_df.index)
        
        predictions = self.ev_state_model.predict(X)
        
        for agent in self.by_type['ev']:
            agent_id = agent.agent_id
            agent_mask = (day_df['agent_id'] == agent_id)
            agent_predictions = predictions[agent_mask]
            agent.hourly_states = list(agent_predictions)