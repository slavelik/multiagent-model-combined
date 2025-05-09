import os
import pickle
import numpy as np
import pandas as pd
from mesa import Agent


from .train_models import train_enterprise_models

class EnterpriseBuildingAgent(Agent):
    """
    Агент предприятия, предсказывающий почасовое энергопотребление,
    используя заранее сгенерированный план и обученную регрессию.
    """
    def __init__(self, model):
        
        super().__init__(model)
        base_dir   = os.path.dirname(__file__)
        tm_dir     = os.path.join(base_dir, 'trained_models')
        plan_path  = os.path.join(base_dir, 'data', 'production_plan.csv')
        model_path = os.path.join(tm_dir, 'best_enterprise_model.pkl')

        # Убедимся, что папка есть и модель обучена
        os.makedirs(tm_dir, exist_ok=True)
        if not os.path.isfile(model_path):
            train_enterprise_models()

        # Загружаем регрессор
        with open(model_path, 'rb') as f:
            self.regressor = pickle.load(f)

        # Загружаем плановый годовой датасет
        self.plan_df = (pd.read_csv(plan_path, parse_dates=['datetime']).set_index('datetime'))

        # Генерируем список признаков динамически (без внешних файлов)
        # числовые признаки
        base_feats = [
            'Motor_and_Transformer_Load_kVarh',
            'is_weekend',
            'hour_sin',
            'hour_cos',
        ]
        # категории Load_Type из плана
        cats = sorted(self.plan_df['Load_Type'].dropna().unique())
        # создаём dummy-признаки, пропуская первую категорию
        dummy_feats = [f"Load_Type_{cat}" for cat in cats[1:]]
        self.feature_columns = base_feats + dummy_feats

        self.consumption = 0.0

    def predict_usage(self) -> float:
        """
        Собирает все признаки для текущего часа и возвращает предсказание в kWh.
        Включает:
          - Motor_and_Transformer_Load_kVarh и Load_Type из плана
          - is_weekend (будни/выходные)
          - циклические признаки часа (sin/cos)
          - погодный признак T_out из модели
        """
        dt = self.model.current_datetime

        # 1) Признак выходного дня
        is_weekend = int(self.model.is_holiday)

        # 2) Циклические признаки времени
        hour     = dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # 4) Специальные параметры из годового плана
        plan_row = self.plan_df.loc[dt]
        mtl      = plan_row['Motor_and_Transformer_Load_kVarh']
        lt       = plan_row['Load_Type']

        # 5) Собираем все «сырые» признаки
        raw = {
            'Motor_and_Transformer_Load_kVarh': mtl,
            'is_weekend'                      : is_weekend,
            'hour_sin'                        : hour_sin,
            'hour_cos'                        : hour_cos,
            'Load_Type'                       : lt
        }

        # 6) One-hot кодирование Load_Type и выравнивание по feature_columns
        df = pd.get_dummies(
            pd.DataFrame([raw]),
            columns=['Load_Type'],
            prefix='Load_Type',
            drop_first=True
        )
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_columns]

        # 7) Предсказание (kWh)
        return float(self.regressor.predict(df)[0])

    def step(self):
        """
        Шаг агента: предсказывает энергопотребление в kWh,
        сохраняет в Wh и выводит лог.
        """
        usage_kwh = self.predict_usage()
        self.consumption = usage_kwh * 1000.0  # перевод в ватт-часы

        # print(
        #     f"[Enterprise {self.unique_id} | {self.model.current_datetime}]  "
        #     f"Weekend={self.model.is_holiday}  "
        #     f"T_out={self.model.current_T_out}°C  "
        #     f"Load_Type={self.plan_df.loc[self.model.current_datetime, 'Load_Type']}  "
        #     f"Usage={usage_kwh:.3f} kWh ({self.consumption:.1f} Wh)"
        # )
