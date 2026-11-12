"""Inkrementalni update budva_{MODEL}_detailed.csv tabela.

Dopunjava samo modele koje koristi glavni pipeline (forecast_48h_v3.MODELS),
od zadnjeg zabiljezenog sata do danas. Postojeci redovi se ne diraju.
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from advanced_model_analysis import (
    LAT, LON, BASE_URL, OBS_CSV,
    load_observed, categorize_weather_conditions,
)

MODELS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",
    "UKMO_SEAMLESS": "ukmo_seamless",
    "ECMWF_IFS": "ecmwf_ifs",
    "KNMI_SEAMLESS": "knmi_seamless",
    "DMI_SEAMLESS": "dmi_seamless",
}

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
]


def fetch_range(model_id, start_date, end_date):
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": HOURLY_VARS, "timezone": "auto",
        "temperature_unit": "celsius", "wind_speed_unit": "ms",
        "precipitation_unit": "mm", "models": model_id,
    }
    for attempt in range(3):
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)
            if r.status_code == 429:
                print("    429 - cekam 60s...", flush=True)
                time.sleep(60)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    greska: {e}", flush=True)
            time.sleep(5)
            continue
        if "hourly" not in data:
            return None
        h = data["hourly"]
        cols = {"datetime": pd.to_datetime(h["time"])}
        for v in HOURLY_VARS:
            if v in h:
                cols[f"{v}_model"] = h[v]
        return pd.DataFrame(cols)
    return None


def update_one(name, model_id, obs):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"budva_{name}_detailed.csv")
    if not os.path.exists(path):
        print(f"{name}: nema fajla, preskacem")
        return None

    old = pd.read_csv(path, parse_dates=['datetime'], low_memory=False)
    schema = list(old.columns)
    last = old['datetime'].max()
    start = (last.date()).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    if start >= end:
        print(f"{name}: vec azuran ({last})")
        return None

    print(f"{name}: {last} -> {end}", flush=True)
    model_df = fetch_range(model_id, start, end)
    if model_df is None or model_df.empty:
        print(f"{name}: FETCH FAILED")
        return (name, 'failed')

    new = pd.merge(obs, model_df, on='datetime', how='inner')
    new = new[new['datetime'] > last]
    if new.empty:
        print(f"{name}: nema novih preklapajucih sati")
        return None

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['datetime'], keep='first')
    combined = combined.sort_values('datetime').reset_index(drop=True)

    combined['temp_obs'] = combined['temperature_2m_obs']
    combined['wind_ms'] = combined['wind_speed_10m_obs']
    combined['solar_wm2'] = combined.get('shortwave_radiation_obs', 0)
    combined['precip_rate_mm'] = combined.get('precipitation_rate_obs', 0)
    combined['precip_accum_mm'] = combined.get('precipitation_obs', 0)
    combined = categorize_weather_conditions(combined)

    for c in schema:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[schema]

    combined.to_csv(path, index=False)
    added = len(combined) - len(old)
    print(f"{name}: {len(old)} -> {len(combined)} (+{added}), do {combined['datetime'].max()}")
    return (name, added)


def main():
    obs = load_observed(OBS_CSV)
    results = []
    for name, mid in MODELS.items():
        try:
            r = update_one(name, mid, obs)
        except Exception as e:
            print(f"{name}: GRESKA {e}")
            r = (name, 'error')
        if r:
            results.append(r)
        time.sleep(5)

    print("\n" + "=" * 60)
    for name, n in results:
        print(f"  {name:22s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
