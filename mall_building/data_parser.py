#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_mall_traffic_dataset.py

Генерирует часовой датасет трафика ТРЦ за период
с 2024-01-01 до 2025-01-01 (локальное время МСК) с признаками:
    datetime, T_out, occupancy_rate, day_off

    - T_out: наружная температура (°C)
    - occupancy_rate: относительная заполняемость ТРЦ (0–100 %)
    - day_off: 1 для выходных и официальных праздников РФ, иначе 0

Требует:
    pip install requests pandas meteostat holidays python-dateutil pytz
"""

import os
import sys
import requests
import pandas as pd
import pytz
from datetime import datetime
from meteostat import Point, Hourly
import holidays

# ---------------------------- КОНФИГ ----------------------------------------------------
API_PRIVATE = "pri_7aa9210061d14cb2b2ae903d24a9dd87"
API_PUBLIC  = "pub_ed5d288795a54f1ca48d4848d84a0a64"

MALL_NAME    = "ТРЦ Авиапарк"
MALL_ADDRESS = "Ходынский бульвар 4, Москва, Россия"

# Интервал выборки (локальное время МСК)
START_LOCAL = datetime(2024, 1, 1, 0, 0)
END_LOCAL   = datetime(2025, 1, 1, 0, 0)

TIMEZONE = "Europe/Moscow"
TZ = pytz.timezone(TIMEZONE)

OUT_DIR = "./MallBuilding/data"
LAT, LON = 55.75, 37.62  # для Meteostat

# Календарь праздников РФ
RU_HOLI = holidays.RU(years=[2024, 2025])

# ------------------------ BestTime API helpers -----------------------------------------
def new_forecast(name: str, address: str) -> str:
    resp = requests.post(
        "https://besttime.app/api/v1/forecasts",
        params={
            "api_key_private": API_PRIVATE,
            "venue_name": name,
            "venue_address": address
        },
        timeout=30
    ).json()
    if resp.get("status") == "OK":
        return resp["venue_info"]["venue_id"]
    if "venue_id" in resp:
        return resp["venue_id"]
    raise RuntimeError(f"Forecast error: {resp}")

def get_week_raw_and_window(venue_id: str) -> tuple[list[int], int]:
    resp = requests.get(
        "https://besttime.app/api/v1/forecasts/week/raw",
        params={"api_key_public": API_PUBLIC, "venue_id": venue_id},
        timeout=30
    ).json()
    wr = resp.get("analysis", {}).get("week_raw")
    ws = resp.get("window", {}).get("time_window_start")
    if not wr or len(wr) != 168 or ws is None:
        raise RuntimeError(f"Bad /week/raw response: {resp}")
    return wr, int(ws)

# ---------------------------- УТИЛИТЫ ----------------------------------------------------
def build_time_series(
    week_raw: list[int],
    window_start: int,
    start_local: datetime,
    end_local: datetime
) -> pd.DataFrame:
    # Генерируем часовую шкалу с tz-aware, потом сбрасываем зону
    total_hours = int((end_local - start_local).total_seconds() // 3600) + 1
    idx_aware = pd.date_range(
        start=TZ.localize(start_local),
        periods=total_hours,
        freq="H",
        tz=TZ
    )
    idx = idx_aware.tz_localize(None)

    # Для каждой метки рассчитываем индекс в week_raw
    visits = []
    for dt in idx:
        N = dt.weekday() * 24 + dt.hour
        j = (N - window_start) % 168
        visits.append(week_raw[j])

    return pd.DataFrame({"datetime": idx, "occupancy_rate": visits})

def fetch_temperature(df: pd.DataFrame) -> pd.DataFrame:
    start, end = df.datetime.min(), df.datetime.max()
    wx = Hourly(Point(LAT, LON), start, end).fetch().reset_index()
    temp_col = next(c for c in ("temp", "tavg") if c in wx.columns)
    wx = wx[["time", temp_col]].rename(columns={"time": "datetime", temp_col: "T_out"})
    return df.merge(wx, on="datetime", how="left")

# ---------------------------- ОСНОВНОЙ СЦЕНАРИЙ ------------------------------------------
def build_dataset():
    # 1) Получаем прогноз
    venue_id = new_forecast(MALL_NAME, MALL_ADDRESS)
    week_raw, window_start = get_week_raw_and_window(venue_id)

    # 2) Строим временной ряд occupancy_rate
    df = build_time_series(week_raw, window_start, START_LOCAL, END_LOCAL)

    # 3) Добавляем температуру и признак выходного/праздника
    df = fetch_temperature(df)
    df["day_off"] = (
        df.datetime.dt.weekday.isin([5, 6]) |
        df.datetime.dt.date.isin(RU_HOLI)
    ).astype(int)

    # 4) Оставляем только нужные признаки
    df = df[["datetime", "T_out", "occupancy_rate", "day_off"]]

    # 5) Сохраняем результат
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "mall_traffic_2024.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset saved ({len(df):,} rows) → {out_path}")
