"""
XGBoost +48h forecast correction for Budva.
Historical bias tables + multi-model ensemble + XGBoost per parameter.
Run: .venv/Scripts/python.exe forecast_48h_v2.py
Author: Matija Ivanović (@matko-iv)
"""

import sys, io, os, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "forecast_output")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models_v2")
PREV_RUNS_DIR = os.path.join(BASE_DIR, "previous_runs_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LAT, LON = 42.29, 18.84  # E viva!!

MODELS = ["ARPEGE_EUROPE", "GFS_SEAMLESS", "ICON_SEAMLESS", "METEOFRANCE", "ECMWF_IFS025", "ITALIAMETEO_ICON2I", "UKMO_SEAMLESS", "BOM_ACCESS", "ECMWF_IFS", "KNMI_SEAMLESS", "DMI_SEAMLESS"]
MODEL_IDS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",
    "UKMO_SEAMLESS": "ukmo_seamless",
    "BOM_ACCESS": "bom_access_global",
    "ECMWF_IFS": "ecmwf_ifs",
    "KNMI_SEAMLESS": "knmi_seamless",
    "DMI_SEAMLESS": "dmi_seamless",
}

PREV_RUNS_MODELS = [m for m in MODELS if m not in ("ITALIAMETEO_ICON2I", "ECMWF_IFS")]
PREV_RUNS_VARS = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_gusts_10m", "pressure_msl",
    "cloud_cover", "precipitation",
]
PREV_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation"
]

TARGET_PARAMS = {
    "temperature_2m":       {"obs": "temperature_2m_obs",       "unit": "\u00b0C",   "display": "Temperatura"},
    "dew_point_2m":         {"obs": "dew_point_2m_obs",         "unit": "\u00b0C",   "display": "Tacka rose"},
    "relative_humidity_2m": {"obs": "relative_humidity_2m_obs", "unit": "%",    "display": "Vlaznost"},
    "wind_speed_10m":       {"obs": "wind_speed_10m_obs",       "unit": "m/s",  "display": "Brzina vjetra"},
    "wind_gusts_10m":       {"obs": "wind_gusts_10m_obs",       "unit": "m/s",  "display": "Udari vjetra"},
    "pressure_msl":         {"obs": "pressure_msl_obs",         "unit": "hPa",  "display": "Pritisak"},
    "cloud_cover":          {"obs": "_derived_cloud_obs",       "unit": "%",    "display": "Oblacnost"},
    "precipitation":        {"obs": "_derived_precip_obs",      "unit": "mm",   "display": "Padavine"},
    "shortwave_radiation":  {"obs": "shortwave_radiation_obs",  "unit": "W/m\u00b2", "display": "Solar. radijacija"},
}

SPLIT_DATE = pd.Timestamp('2025-07-01')

print("=" * 72)
print("  XGBoost +48h v3 --- Bias Correction Pipeline --- Budva")
print("  Models:", len(MODELS), "| Obs: merged (2020-2026) | Split:", SPLIT_DATE.date())
print("  Previous Runs: +Day1/Day2 forecasts for", len(PREV_RUNS_MODELS), "models")
print("=" * 72)


def compute_clear_sky(dt_series):
    doy = dt_series.dt.dayofyear
    hour = dt_series.dt.hour + dt_series.dt.minute / 60.0
    lat_rad = np.radians(LAT)
    dec = np.radians(23.45 * np.sin(np.radians(360 / 365.25 * (doy - 81))))
    ha = np.radians(15 * (hour - 12))
    sin_e = (np.sin(lat_rad) * np.sin(dec) +
             np.cos(lat_rad) * np.cos(dec) * np.cos(ha)).clip(lower=0)
    return (1361 * sin_e * 0.75).clip(lower=0)


def load_historical_data():
    print("\n[1/6] Ucitavanje istorijskih podataka...")
    all_dfs = {}
    available_models = []
    for m in MODELS:
        path = os.path.join(BASE_DIR, f"budva_{m}_detailed.csv")
        if not os.path.exists(path):
            print(f"  {m}: NEMA FAJLA - preskačem (pokreni fetch_new_models.py)")
            continue
        all_dfs[m] = pd.read_csv(path, parse_dates=['datetime'])
        available_models.append(m)
        print(f"  {m}: {all_dfs[m].shape[0]} redova")

    if not available_models:
        raise RuntimeError("Nema nijednog model fajla!")

    forecast_cols = [c for c in all_dfs[available_models[0]].columns if c.endswith('_model')]
    base = all_dfs[available_models[0]].copy()
    base.rename(columns={c: f"{available_models[0]}_{c}" for c in forecast_cols}, inplace=True)
    for m in available_models[1:]:
        other_cols = [c for c in all_dfs[m].columns if c.endswith('_model')]
        other = all_dfs[m][['datetime'] + other_cols].copy()
        other.rename(columns={c: f"{m}_{c}" for c in other_cols}, inplace=True)
        base = base.merge(other, on='datetime', how='left')
    base.sort_values('datetime', inplace=True)
    base.reset_index(drop=True, inplace=True)

    solar = pd.to_numeric(base.get('shortwave_radiation_obs', pd.Series(dtype=float)), errors='coerce')
    clear = compute_clear_sky(base['datetime'])
    clarity = (solar / clear.clip(lower=1)).clip(0, 1.5)
    cloud = (1 - clarity).clip(0, 1) * 100
    cloud[clear < 20] = np.nan
    base['_derived_cloud_obs'] = cloud
    print(f"  Cloud cover derived: {cloud.notna().sum()} valid (daytime)")

    precip_derived = False
    for precip_col, multiplier in [('precip_rate_in', 25.4), ('precipitation_rate_obs', 1.0), ('precip_rate_mm', 1.0)]:
        if precip_col in base.columns:
            base['_derived_precip_obs'] = pd.to_numeric(base[precip_col], errors='coerce') * multiplier
            n_nonzero = (base['_derived_precip_obs'] > 0).sum()
            print(f"  Hourly precip derived from '{precip_col}': {n_nonzero} non-zero hours")
            precip_derived = True
            break
    if not precip_derived:
        base['_derived_precip_obs'] = np.nan
        print("  WARNING: No precip obs column found")

    print("  Ucitavanje previous runs podataka (Day1/Day2)...")
    prev_merged = 0
    for m in PREV_RUNS_MODELS:
        prev_path = os.path.join(PREV_RUNS_DIR, f"{m}_previous_runs.csv")
        if not os.path.exists(prev_path):
            print(f"    {m}: nema fajla - preskačem")
            continue
        prev = pd.read_csv(prev_path, parse_dates=['datetime'])
        rename_map = {}
        for v in PREV_RUNS_VARS:
            for lag in ['previous_day1', 'previous_day2']:
                old_col = f"{v}_{lag}"
                new_col = f"{m}_{v}_{lag}"
                if old_col in prev.columns:
                    rename_map[old_col] = new_col
        prev_keep = ['datetime'] + list(rename_map.keys())
        prev = prev[[c for c in prev_keep if c in prev.columns]].rename(columns=rename_map)
        base = base.merge(prev, on='datetime', how='left')
        prev_merged += 1
        n_valid = base[f'{m}_temperature_2m_previous_day1'].notna().sum() if f'{m}_temperature_2m_previous_day1' in base.columns else 0
        print(f"    {m}: merged ({n_valid} valid Day1 rows)")
    print(f"  Previous runs: {prev_merged} modela merged")

    print(f"  Merged: {base.shape[0]} x {base.shape[1]}")
    return base


def compute_bias_tables(df):
    print("\n  Kreiranje tabela istorijskog biasa (samo na train podacima)...")
    train = df[df['datetime'] < SPLIT_DATE].copy()
    train['month'] = train['datetime'].dt.month
    train['hour'] = train['datetime'].dt.hour

    bias_tables = {}
    for param, info in TARGET_PARAMS.items():
        obs_col = info['obs']
        if obs_col not in train.columns:
            continue
        obs = pd.to_numeric(train[obs_col], errors='coerce')

        for m in MODELS:
            fcst_col = f"{m}_{param}_model"
            if fcst_col not in train.columns:
                continue
            fcst = pd.to_numeric(train[fcst_col], errors='coerce')
            err = fcst - obs
            tmp = pd.DataFrame({'err': err, 'month': train['month'], 'hour': train['hour']})
            table = tmp.groupby(['month', 'hour'])['err'].agg(['mean', 'std']).reset_index()
            table.columns = ['month', 'hour', 'bias_mean', 'bias_std']
            key = f"{m}_{param}"
            bias_tables[key] = table

    print(f"  Tabele biasa: {len(bias_tables)} (model x param kombinacija)")
    return bias_tables


def apply_bias_features(df, bias_tables):
    df = df.copy()
    df['_month'] = df['datetime'].dt.month
    df['_hour'] = df['datetime'].dt.hour

    for key, table in bias_tables.items():
        merged = df[['_month', '_hour']].merge(
            table, left_on=['_month', '_hour'], right_on=['month', 'hour'], how='left'
        )
        df[f'{key}_hist_bias'] = merged['bias_mean'].values
        df[f'{key}_hist_bias_std'] = merged['bias_std'].values

    df.drop(columns=['_month', '_hour'], inplace=True)
    return df


def engineer_features(df):
    out = df.copy()

    model_cols = [c for c in out.columns if c.endswith('_model')]
    for c in model_cols:
        if out[c].dtype == 'object':
            out[c] = pd.to_numeric(out[c], errors='coerce')

    out['hour'] = out['datetime'].dt.hour
    out['month'] = out['datetime'].dt.month
    out['day_of_year'] = out['datetime'].dt.dayofyear
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    out['doy_sin'] = np.sin(2 * np.pi * out['day_of_year'] / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * out['day_of_year'] / 365.25)
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out['season'] = out['month'].map(season_map)

    clear = compute_clear_sky(out['datetime'])
    out['is_daytime'] = (clear > 20).astype(float)
    out['clear_sky_rad'] = clear

    ensemble_params = ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
                       'apparent_temperature', 'wind_speed_10m', 'wind_gusts_10m',
                       'wind_direction_10m', 'pressure_msl', 'surface_pressure',
                       'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                       'cloud_cover_high', 'precipitation', 'shortwave_radiation',
                       'direct_radiation', 'diffuse_radiation', 'rain']

    for param in ensemble_params:
        mcols = [f"{m}_{param}_model" for m in MODELS if f"{m}_{param}_model" in out.columns]
        if len(mcols) < 2:
            continue
        vals = out[mcols].apply(pd.to_numeric, errors='coerce')
        out[f'{param}_ens_mean'] = vals.mean(axis=1)
        out[f'{param}_ens_std'] = vals.std(axis=1)
        out[f'{param}_ens_range'] = vals.max(axis=1) - vals.min(axis=1)
        out[f'{param}_ens_median'] = vals.median(axis=1)
        out[f'{param}_ens_min'] = vals.min(axis=1)
        out[f'{param}_ens_max'] = vals.max(axis=1)

        if len(mcols) >= 4:
            sorted_vals = np.sort(vals.values, axis=1)
            out[f'{param}_ens_trimmed_mean'] = np.nanmean(sorted_vals[:, 1:-1], axis=1)
        for m in MODELS:
            c = f"{m}_{param}_model"
            if c in out.columns:
                out[f'{m}_{param}_dev'] = pd.to_numeric(out[c], errors='coerce') - out[f'{param}_ens_mean']

    if 'temperature_2m_ens_mean' in out.columns and 'dew_point_2m_ens_mean' in out.columns:
        out['temp_dew_spread'] = out['temperature_2m_ens_mean'] - out['dew_point_2m_ens_mean']
    if 'wind_speed_10m_ens_mean' in out.columns and 'pressure_msl_ens_mean' in out.columns:
        out['wind_pressure_idx'] = out['wind_speed_10m_ens_mean'] / (out['pressure_msl_ens_mean'] / 1013.25).clip(lower=0.9)
    if 'cloud_cover_ens_mean' in out.columns:
        out['cloud_solar_discrepancy'] = out.get('shortwave_radiation_ens_mean', 0) / (out['cloud_cover_ens_mean'].clip(lower=1))
    if 'pressure_msl_ens_mean' in out.columns:
        out['pres_tend_3h'] = out['pressure_msl_ens_mean'].diff(3)
        out['pres_tend_6h'] = out['pressure_msl_ens_mean'].diff(6)
    if 'temperature_2m_ens_mean' in out.columns:
        out['temp_tend_3h'] = out['temperature_2m_ens_mean'].diff(3)
        out['temp_tend_6h'] = out['temperature_2m_ens_mean'].diff(6)

    rain_mcols = [f"{m}_precipitation_model" for m in MODELS if f"{m}_precipitation_model" in out.columns]
    if rain_mcols:
        rain_vals = out[rain_mcols].apply(pd.to_numeric, errors='coerce')
        out['rain_model_count'] = (rain_vals > 0.1).sum(axis=1)
        out['rain_agreement'] = out['rain_model_count'] / max(len(rain_mcols), 1)

    wc_feat_cols = [f"{m}_weather_code_model" for m in MODELS if f"{m}_weather_code_model" in out.columns]
    if wc_feat_cols:
        wc_vals = out[wc_feat_cols].apply(pd.to_numeric, errors='coerce')
        out['rain_wc_count'] = (wc_vals >= 51).sum(axis=1)
        out['storm_wc_count'] = (wc_vals >= 95).sum(axis=1)

    if 'precipitation_ens_mean' in out.columns and 'temperature_2m_ens_mean' in out.columns:
        out['precip_x_temp'] = out['precipitation_ens_mean'] * out['temperature_2m_ens_mean']
        out['precip_x_temp_std'] = out['precipitation_ens_mean'] * out.get('temperature_2m_ens_std', 0)

    if 'cloud_cover_ens_mean' in out.columns:
        out['cloud_tend_3h'] = out['cloud_cover_ens_mean'].diff(3)
        out['cloud_tend_6h'] = out['cloud_cover_ens_mean'].diff(6)

    if 'precipitation_ens_mean' in out.columns:
        out['precip_tend_3h'] = out['precipitation_ens_mean'].diff(3)
        out['precip_tend_6h'] = out['precipitation_ens_mean'].diff(6)

    if 'relative_humidity_2m_ens_mean' in out.columns:
        out['humidity_tend_3h'] = out['relative_humidity_2m_ens_mean'].diff(3)

    if 'temp_dew_spread' in out.columns:
        out['dew_spread_tend_3h'] = out['temp_dew_spread'].diff(3)
        out['dew_spread_tend_6h'] = out['temp_dew_spread'].diff(6)

    for m in MODELS:
        wd = f"{m}_wind_direction_10m_model"
        ws = f"{m}_wind_speed_10m_model"
        if wd in out.columns and ws in out.columns:
            d = pd.to_numeric(out[wd], errors='coerce')
            s = pd.to_numeric(out[ws], errors='coerce')
            out[f'{m}_bura'] = (((d >= 315) | (d <= 45)) & (s >= 8)).astype(float)
    bura_cols = [f'{m}_bura' for m in MODELS if f'{m}_bura' in out.columns]
    if bura_cols:
        out['bura_agreement'] = out[bura_cols].sum(axis=1)

    if rain_mcols:
        rain_vals = out[rain_mcols].apply(pd.to_numeric, errors='coerce').fillna(0)
        ens_precip = rain_vals.mean(axis=1)

        out['precip_running_6h'] = ens_precip.rolling(6, min_periods=1).sum()
        out['precip_running_12h'] = ens_precip.rolling(12, min_periods=1).sum()
        out['precip_running_24h'] = ens_precip.rolling(24, min_periods=1).sum()

        rain_hours = (rain_vals > 0.1).sum(axis=1)
        out['rain_hours_6h'] = rain_hours.rolling(6, min_periods=1).sum()
        out['rain_hours_12h'] = rain_hours.rolling(12, min_periods=1).sum()
        out['rain_hours_24h'] = rain_hours.rolling(24, min_periods=1).sum()

        agreement = (rain_vals > 0.1).sum(axis=1) / max(len(rain_mcols), 1)
        out['rain_agreement_6h'] = agreement.rolling(6, min_periods=1).mean()
        out['rain_agreement_12h'] = agreement.rolling(12, min_periods=1).mean()

        out['persistent_rain'] = (
            (out['rain_hours_6h'] >= 3) &
            (agreement >= 0.5)
        ).astype(float)

        out['sustained_rain_12h'] = (
            (out['rain_hours_12h'] >= 6) &
            (out['rain_agreement_12h'] >= 0.4)
        ).astype(float)

    is_winter = out['month'].isin([11, 12, 1, 2, 3]).astype(float)
    out['is_winter'] = is_winter

    if 'temp_dew_spread' in out.columns:
        out['dew_saturated'] = (out['temp_dew_spread'] < 2.0).astype(float)
        if rain_mcols:
            out['winter_rain_signal'] = (
                is_winter *
                out.get('persistent_rain', 0) *
                out['dew_saturated']
            )

    if 'relative_humidity_2m_ens_mean' in out.columns:
        rh = out['relative_humidity_2m_ens_mean']
        out['humidity_above_90'] = (rh > 90).astype(float)
        out['humidity_above_90_6h'] = out['humidity_above_90'].rolling(6, min_periods=1).sum()

        if rain_mcols:
            out['humid_rain_persistence'] = (
                out['humidity_above_90_6h'] *
                out.get('rain_agreement_6h', 0)
            )

    if 'cloud_cover_ens_mean' in out.columns:
        cc = out['cloud_cover_ens_mean']
        out['overcast_6h'] = (cc > 80).rolling(6, min_periods=1).mean()
        out['overcast_12h'] = (cc > 80).rolling(12, min_periods=1).mean()

    if 'precipitation_ens_mean' in out.columns:
        pem = out['precipitation_ens_mean']
        out['precip_ens_running_6h'] = pem.rolling(6, min_periods=1).sum()
        out['precip_ens_nonzero_6h'] = (pem > 0.05).rolling(6, min_periods=1).sum()
        out['precip_ens_nonzero_12h'] = (pem > 0.05).rolling(12, min_periods=1).sum()
        out['precip_ens_intensity'] = pem.rolling(6, min_periods=1).mean()

    if 'precipitation_ens_std' in out.columns:
        out['precip_model_certainty'] = 1.0 / (1.0 + out['precipitation_ens_std'])

    
    if 'pressure_msl_ens_mean' in out.columns:
        pres = out['pressure_msl_ens_mean']

        pres_change_3h = pres.diff(3)
        pres_change_6h = pres.diff(6)
        pres_change_12h = pres.diff(12)

        out['pres_change_12h'] = pres_change_12h

        out['pres_rapidly_falling'] = (pres_change_3h < -3.0).astype(float)
        out['pres_falling'] = (pres_change_3h < -1.0).astype(float)
        out['pres_rising'] = (pres_change_3h > 1.0).astype(float)
        out['pres_rapidly_rising'] = (pres_change_3h > 3.0).astype(float)

        out['pres_anomaly'] = pres - 1015.0

        out['low_pressure_regime'] = (pres < 1010.0).astype(float)
        out['very_low_pressure'] = (pres < 1005.0).astype(float)
        out['high_pressure_regime'] = (pres > 1020.0).astype(float)

        out['pres_stability_6h'] = pres.rolling(6, min_periods=2).std()
        out['pres_stability_12h'] = pres.rolling(12, min_periods=3).std()

        if 'pressure_msl_ens_std' in out.columns:
            out['pres_model_disagreement'] = out['pressure_msl_ens_std']
            out['pres_high_uncertainty'] = (out['pressure_msl_ens_std'] > 1.5).astype(float)

        if 'cloud_cover_ens_mean' in out.columns:
            out['frontal_signal'] = (
                (pres_change_6h < -2.0) &
                (out['cloud_cover_ens_mean'] > 70)
            ).astype(float)

            if rain_mcols:
                out['active_front'] = (
                    out['frontal_signal'] *
                    out.get('rain_agreement', 0)
                )

    if 'relative_humidity_2m_ens_mean' in out.columns:
        rh = out['relative_humidity_2m_ens_mean']

        out['humidity_above_80'] = (rh > 80).astype(float)
        out['humidity_above_95'] = (rh > 95).astype(float)
        out['sustained_humid_6h'] = out['humidity_above_80'].rolling(6, min_periods=1).sum()
        out['sustained_humid_12h'] = out['humidity_above_80'].rolling(12, min_periods=1).sum()
        out['sustained_humid_24h'] = out['humidity_above_80'].rolling(24, min_periods=1).sum()

        out['rh_tend_1h'] = rh.diff(1)
        out['rh_tend_3h'] = rh.diff(3)
        out['rh_tend_6h'] = rh.diff(6)
        out['rh_rising'] = (out['rh_tend_3h'] > 3.0).astype(float)
        out['rh_falling'] = (out['rh_tend_3h'] < -3.0).astype(float)

        if 'relative_humidity_2m_ens_std' in out.columns:
            out['rh_model_disagreement'] = out['relative_humidity_2m_ens_std']

        if 'cloud_cover_ens_mean' in out.columns:
            cc = out['cloud_cover_ens_mean']
            out['humid_overcast'] = (rh * cc / 100.0)
            out['humid_overcast_flag'] = ((rh > 80) & (cc > 80)).astype(float)
            out['dry_clear_flag'] = ((rh < 50) & (cc < 30)).astype(float)

        if 'wind_speed_10m_ens_mean' in out.columns:
            out['rh_wind_interaction'] = rh / (1.0 + out['wind_speed_10m_ens_mean'])

        out['night_humid'] = (
            out.get('is_daytime', pd.Series(0, index=out.index)).eq(0) &
            (rh > 75)
        ).astype(float)

    if 'temperature_2m_ens_mean' in out.columns and 'dew_point_2m_ens_mean' in out.columns:
        temp = out['temperature_2m_ens_mean']
        dew = out['dew_point_2m_ens_mean']
        spread = temp - dew

        out['near_saturation'] = (spread < 1.5).astype(float)
        out['moderate_spread'] = ((spread >= 1.5) & (spread < 5.0)).astype(float)
        out['dry_spread'] = (spread >= 8.0).astype(float)

        out['near_sat_6h'] = out['near_saturation'].rolling(6, min_periods=1).sum()
        out['near_sat_12h'] = out['near_saturation'].rolling(12, min_periods=1).sum()

        if 'wind_speed_10m_ens_mean' in out.columns:
            out['fog_risk'] = (
                (spread < 2.0) &
                (out['wind_speed_10m_ens_mean'] < 3.0)
            ).astype(float)

        out['spread_tend_3h'] = spread.diff(3)
        out['spread_closing'] = (out['spread_tend_3h'] < -1.0).astype(float)
        out['spread_opening'] = (out['spread_tend_3h'] > 1.0).astype(float)

    if 'temperature_2m_ens_mean' in out.columns:
        temp = out['temperature_2m_ens_mean']

        if 'temperature_2m_ens_std' in out.columns:
            out['temp_high_uncertainty'] = (out['temperature_2m_ens_std'] > 1.0).astype(float)

        if 'is_daytime' in out.columns:
            out['dtr_proxy'] = temp.rolling(24, min_periods=6).max() - temp.rolling(24, min_periods=6).min()

            if 'cloud_cover_ens_mean' in out.columns:
                out['dtr_x_cloud'] = out['dtr_proxy'] * (1.0 - out['cloud_cover_ens_mean'] / 100.0)

        out['temp_near_zero'] = ((temp > -2.0) & (temp < 5.0)).astype(float)

        if 'precipitation_ens_mean' in out.columns:
            pem = out['precipitation_ens_mean']
            out['temp_x_precip'] = temp * pem.clip(upper=5.0)
            out['cold_rain'] = ((temp < 8.0) & (pem > 0.1)).astype(float)
            out['warm_rain'] = ((temp > 20.0) & (pem > 0.1)).astype(float)

        month = out['month']
        sea_temp_approx = 13.0 + 6.0 * np.sin(2 * np.pi * (month - 3) / 12)
        out['sea_air_diff'] = sea_temp_approx - temp
        out['marine_warming'] = (out['sea_air_diff'] > 3.0).astype(float)  # sea warms air
        out['marine_cooling'] = (out['sea_air_diff'] < -3.0).astype(float)  # sea cools air

    if all(c in out.columns for c in ['is_winter', 'cloud_cover_ens_mean',
           'relative_humidity_2m_ens_mean']):
        cc = out['cloud_cover_ens_mean']
        rh = out['relative_humidity_2m_ens_mean']

        winter_overcast = (
            (out['is_winter'] > 0) &
            (cc > 75) &
            (rh > 70)
        ).astype(float)
        out['winter_overcast_regime'] = winter_overcast

        out['winter_overcast_6h'] = winter_overcast.rolling(6, min_periods=1).sum()
        out['winter_overcast_12h'] = winter_overcast.rolling(12, min_periods=1).sum()

        if 'precipitation_ens_mean' in out.columns:
            pem = out['precipitation_ens_mean']
            out['winter_overcast_rain'] = (
                winter_overcast *
                (pem > 0.1).astype(float)
            )
            out['winter_overcast_rain_12h'] = out['winter_overcast_rain'].rolling(12, min_periods=1).sum()

        if 'pressure_msl_ens_mean' in out.columns:
            pres = out['pressure_msl_ens_mean']
            out['winter_pres_above_1020'] = (
                (out['is_winter'] > 0) &
                (pres > 1020.0)
            ).astype(float)

            out['winter_low_pres_rain'] = (
                (out['is_winter'] > 0) &
                (pres < 1010.0) &
                (out.get('rain_agreement', pd.Series(0, index=out.index)) > 0.3)
            ).astype(float)

    pres_mcols = [f"{m}_pressure_msl_model" for m in MODELS
                  if f"{m}_pressure_msl_model" in out.columns]
    if len(pres_mcols) >= 2 and 'pressure_msl_ens_mean' in out.columns:
        pres_ens = out['pressure_msl_ens_mean']
        for m in MODELS:
            pc = f"{m}_pressure_msl_model"
            if pc in out.columns:
                dev = pd.to_numeric(out[pc], errors='coerce') - pres_ens
                out[f'{m}_pres_bias'] = dev

        pres_vals = out[pres_mcols].apply(pd.to_numeric, errors='coerce')
        out['pres_max_spread'] = pres_vals.max(axis=1) - pres_vals.min(axis=1)

    rh_mcols = [f"{m}_relative_humidity_2m_model" for m in MODELS
                if f"{m}_relative_humidity_2m_model" in out.columns]
    if len(rh_mcols) >= 2 and 'relative_humidity_2m_ens_mean' in out.columns:
        rh_ens = out['relative_humidity_2m_ens_mean']
        for m in MODELS:
            rhc = f"{m}_relative_humidity_2m_model"
            if rhc in out.columns:
                dev = pd.to_numeric(out[rhc], errors='coerce') - rh_ens
                out[f'{m}_rh_bias'] = dev

        rh_vals = out[rh_mcols].apply(pd.to_numeric, errors='coerce')
        out['rh_max_spread'] = rh_vals.max(axis=1) - rh_vals.min(axis=1)

        out['rh_above85_count'] = (rh_vals > 85).sum(axis=1)
        out['rh_above85_ratio'] = out['rh_above85_count'] / len(rh_mcols)

    if all(c in out.columns for c in ['cloud_cover_ens_mean',
           'relative_humidity_2m_ens_mean', 'shortwave_radiation_ens_mean']):
        cc = out['cloud_cover_ens_mean']
        rh = out['relative_humidity_2m_ens_mean']
        sw = out['shortwave_radiation_ens_mean']
        clear = out.get('clear_sky_rad', pd.Series(1, index=out.index))

        out['cloud_rh_inconsistent'] = ((cc > 80) & (rh < 60)).astype(float)

        out['humid_clear_sky'] = ((cc < 30) & (rh > 85)).astype(float)

        solar_ratio = sw / clear.clip(lower=1)
        out['solar_cloud_mismatch'] = (
            (cc > 80) & (solar_ratio > 0.5) |
            (cc < 20) & (solar_ratio < 0.3)
        ).astype(float)

    if 'precipitation_ens_mean' in out.columns:
        pem = out['precipitation_ens_mean']

        out['precip_24h_total'] = pem.rolling(24, min_periods=1).sum()
        out['precip_48h_total'] = pem.rolling(48, min_periods=1).sum()

        out['heavy_rain_event'] = (pem > 3.0).astype(float)
        out['heavy_rain_6h'] = out['heavy_rain_event'].rolling(6, min_periods=1).sum()

        out['precip_decreasing'] = (pem.diff(3) < -0.5).astype(float)
        out['post_rain_clearing'] = (
            (out['precip_24h_total'] > 5.0) &
            (pem < 0.1) &
            out['precip_decreasing'].astype(bool)
        ).astype(float)

    wd_mcols = [f"{m}_wind_direction_10m_model" for m in MODELS
                if f"{m}_wind_direction_10m_model" in out.columns]
    ws_mcols = [f"{m}_wind_speed_10m_model" for m in MODELS
                if f"{m}_wind_speed_10m_model" in out.columns]
    if wd_mcols and ws_mcols:
        wd_vals = out[wd_mcols].apply(pd.to_numeric, errors='coerce')
        ws_vals = out[ws_mcols].apply(pd.to_numeric, errors='coerce')

        wd_mean = np.degrees(np.arctan2(
            np.sin(np.radians(wd_vals)).mean(axis=1),
            np.cos(np.radians(wd_vals)).mean(axis=1)
        )) % 360
        ws_mean = ws_vals.mean(axis=1)

        out['is_jugo'] = (
            ((wd_mean >= 100) & (wd_mean <= 170)) &
            (ws_mean > 5.0)
        ).astype(float)

        out['is_maestral'] = (
            ((wd_mean >= 280) & (wd_mean <= 340)) &
            (ws_mean > 3.0) &
            (out['month'].isin([5, 6, 7, 8, 9]).astype(float) > 0)
        ).astype(float)

        out['jugo_6h'] = out['is_jugo'].rolling(6, min_periods=1).sum()

        out['winter_jugo'] = (
            out['is_jugo'] * out.get('is_winter', pd.Series(0, index=out.index))
        )

    for v in PREV_RUNS_VARS:
        day1_cols = [f"{m}_{v}_previous_day1" for m in PREV_RUNS_MODELS
                     if f"{m}_{v}_previous_day1" in out.columns]
        day2_cols = [f"{m}_{v}_previous_day2" for m in PREV_RUNS_MODELS
                     if f"{m}_{v}_previous_day2" in out.columns]

        if len(day1_cols) >= 2:
            d1_vals = out[day1_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_prev_day1_ens_mean'] = d1_vals.mean(axis=1)
            out[f'{v}_prev_day1_ens_std'] = d1_vals.std(axis=1)

        if len(day2_cols) >= 2:
            d2_vals = out[day2_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_prev_day2_ens_mean'] = d2_vals.mean(axis=1)

        rev_cols = []
        for m in PREV_RUNS_MODELS:
            d0 = f"{m}_{v}_model"
            d1 = f"{m}_{v}_previous_day1"
            if d0 in out.columns and d1 in out.columns:
                col_name = f'{m}_{v}_revision'
                out[col_name] = (pd.to_numeric(out[d0], errors='coerce') -
                                 pd.to_numeric(out[d1], errors='coerce'))
                rev_cols.append(col_name)

        d1d2_rev_cols = []
        for m in PREV_RUNS_MODELS:
            d1 = f"{m}_{v}_previous_day1"
            d2 = f"{m}_{v}_previous_day2"
            if d1 in out.columns and d2 in out.columns:
                col_name = f'{m}_{v}_d1d2_revision'
                out[col_name] = (pd.to_numeric(out[d1], errors='coerce') -
                                 pd.to_numeric(out[d2], errors='coerce'))
                d1d2_rev_cols.append(col_name)

        if len(rev_cols) >= 2:
            rv = out[rev_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_revision_ens_mean'] = rv.mean(axis=1)
            out[f'{v}_revision_ens_std'] = rv.std(axis=1)
            out[f'{v}_revision_ens_abs_mean'] = rv.abs().mean(axis=1)

        if len(d1d2_rev_cols) >= 2:
            d1d2 = out[d1d2_rev_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_d1d2_revision_ens_mean'] = d1d2.mean(axis=1)

        if f'{v}_prev_day1_ens_mean' in out.columns and f'{v}_ens_mean' in out.columns:
            out[f'{v}_day0_vs_day1_ens'] = out[f'{v}_ens_mean'] - out[f'{v}_prev_day1_ens_mean']

    for param in ['temperature_2m', 'dew_point_2m', 'pressure_msl', 'wind_speed_10m',
                  'relative_humidity_2m', 'cloud_cover']:
        ens_col = f'{param}_ens_mean'
        if ens_col in out.columns:
            ser = out[ens_col]
            out[f'{param}_ens_lag1'] = ser.shift(1)
            out[f'{param}_ens_lag3'] = ser.shift(3)
            out[f'{param}_ens_ma6'] = ser.rolling(6, min_periods=1).mean()
            out[f'{param}_ens_ma12'] = ser.rolling(12, min_periods=1).mean()
            out[f'{param}_ens_ma24'] = ser.rolling(24, min_periods=1).mean()
            out[f'{param}_ens_std6'] = ser.rolling(6, min_periods=2).std()
            out[f'{param}_ens_std24'] = ser.rolling(24, min_periods=3).std()
            out[f'{param}_ens_anom24'] = ser - out[f'{param}_ens_ma24']

    if 'precipitation_ens_mean' in out.columns:
        pem = out['precipitation_ens_mean']
        out['precip_sqrt'] = np.sqrt(pem.clip(lower=0))
        out['precip_log1p'] = np.log1p(pem.clip(lower=0))
        out['precip_is_zero'] = (pem < 0.05).astype(float)
        out['precip_dry_hours'] = out['precip_is_zero'].rolling(12, min_periods=1).sum()

    for param in ['temperature_2m', 'precipitation', 'wind_speed_10m', 'cloud_cover']:
        mcols = [f"{m}_{param}_model" for m in MODELS if f"{m}_{param}_model" in out.columns]
        if len(mcols) >= 4:
            vals = out[mcols].apply(pd.to_numeric, errors='coerce')
            q25 = vals.quantile(0.25, axis=1)
            q75 = vals.quantile(0.75, axis=1)
            out[f'{param}_ens_iqr'] = q75 - q25
            out[f'{param}_ens_skew'] = vals.skew(axis=1)

    if 'hour_sin' in out.columns and 'season' in out.columns:
        out['hour_sin_x_season'] = out['hour_sin'] * out['season']
        out['hour_cos_x_season'] = out['hour_cos'] * out['season']
    if 'hour_sin' in out.columns and 'doy_sin' in out.columns:
        out['hour_x_doy'] = out['hour_sin'] * out['doy_sin']

    if 'temperature_2m_ens_std' in out.columns and 'temperature_2m_ens_mean' in out.columns:
        out['temp_cv'] = out['temperature_2m_ens_std'] / (out['temperature_2m_ens_mean'].abs().clip(lower=0.1))

    return out


def get_feature_columns(df):
    exclude = set([
        'datetime', 'date',
        'temp_f', 'dewpoint_f', 'wind_mph', 'gust_mph', 'pressure_in',
        'precip_rate_in', 'precip_accum_in', 'precip_rate_mm', 'precip_accum_mm',
        'temp_obs', 'wind_ms', 'solar_wm2', 'uv',
        'temp_c', 'dewpoint_c', 'humidity_pct', 'wind_dir', 'gust_ms', 'pressure_hpa',
        'day_night', 'time_of_day', 'has_rain', 'light_rain', 'heavy_rain',
        'strong_wind', 'very_strong_wind', 'is_bura', 'winter_bura',
        'cloudy', 'extreme_cold', 'extreme_hot',
        '_derived_cloud_obs', '_derived_precip_obs',
        'date_str', 'time_str', '_h',
    ])
    obs_suffix = '_obs'

    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.endswith(obs_suffix):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def _make_val_split(X_tr, y_tr, val_frac=0.05):
    """Split last val_frac of training data as validation (respects time order)."""
    n = len(X_tr)
    split_idx = int(n * (1 - val_frac))
    return (X_tr.iloc[:split_idx], y_tr.iloc[:split_idx],
            X_tr.iloc[split_idx:], y_tr.iloc[split_idx:])


def _train_xgb(X_tr, y_tr, X_val, y_val, hp):
    """Two-pass training: find best n_estimators on val, retrain on all data."""
    model_val = xgb.XGBRegressor(**hp)
    model_val.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_n = model_val.best_iteration + 1
    if best_n < 10:
        best_n = hp.get('n_estimators', 500)

    hp_final = {k: v for k, v in hp.items() if k != 'early_stopping_rounds'}
    hp_final['n_estimators'] = best_n
    X_full = pd.concat([X_tr, X_val], axis=0)
    y_full = pd.concat([y_tr, y_val], axis=0)
    model = xgb.XGBRegressor(**hp_final)
    model.fit(X_full, y_full, verbose=False)
    return model, list(X_tr.columns)


def _find_optimal_blend(y_pred, y_te, ens_te):
    """Find optimal alpha: final = alpha*xgb + (1-alpha)*ensemble."""
    base_mae = mean_absolute_error(y_te, y_pred)
    best_alpha, best_mae = 1.0, base_mae
    best_pred = y_pred.copy()
    for alpha in np.arange(0.50, 1.01, 0.025):
        blend = alpha * y_pred + (1 - alpha) * ens_te
        bm = mean_absolute_error(y_te, blend)
        if bm < best_mae:
            best_mae, best_alpha = bm, alpha
            best_pred = blend.copy()
    return best_alpha, best_mae, best_pred


def _train_residual_blended(X_tr, y_tr, X_te, y_te, hp, param, ens_col, df_v_tr, df_v_te):
    """Train direct + residual (Huber) models on ALL data, find optimal blend.
    No splits — uses full 50K training set."""
    ens_tr = pd.to_numeric(df_v_tr[ens_col], errors='coerce').fillna(0) if ens_col in df_v_tr.columns else pd.Series(0, index=y_tr.index)
    ens_te = pd.to_numeric(df_v_te[ens_col], errors='coerce').fillna(0) if ens_col in df_v_te.columns else pd.Series(0, index=y_te.index)

    X_train_a, y_train_a, X_val_a, y_val_a = _make_val_split(X_tr, y_tr)
    direct_model, _ = _train_xgb(X_train_a, y_train_a, X_val_a, y_val_a, hp)
    direct_pred = direct_model.predict(X_te)

    y_resid_tr = y_tr - ens_tr.values
    X_train_b, y_train_b, X_val_b, y_val_b = _make_val_split(X_tr, y_resid_tr)
    hp_resid = hp.copy()
    hp_resid['objective'] = 'reg:pseudohubererror'
    resid_model, _ = _train_xgb(X_train_b, y_train_b, X_val_b, y_val_b, hp_resid)
    resid_correction = resid_model.predict(X_te)
    resid_pred = ens_te.values + resid_correction

    best_alpha, best_blend_mae = 1.0, float('inf')
    for alpha in np.arange(0.5, 1.01, 0.05):
        blend = alpha * direct_pred + (1 - alpha) * ens_te.values
        bm = mean_absolute_error(y_te, blend)
        if bm < best_blend_mae:
            best_blend_mae, best_alpha = bm, alpha
    blend_pred = best_alpha * direct_pred + (1 - best_alpha) * ens_te.values

    mae_direct = mean_absolute_error(y_te, direct_pred)
    mae_resid = mean_absolute_error(y_te, resid_pred)
    mae_blend = best_blend_mae

    methods = {'direct': (mae_direct, direct_pred, direct_model, False),
               'residual': (mae_resid, resid_pred, resid_model, True),
               'blend': (mae_blend, blend_pred, direct_model, False)}

    best_name = min(methods, key=lambda k: methods[k][0])
    best_mae, best_pred, best_model, is_residual = methods[best_name]
    best_rmse = np.sqrt(mean_squared_error(y_te, best_pred))

    info_str = f"direct={mae_direct:.3f}, residual={mae_resid:.3f}, blend({best_alpha:.2f})={mae_blend:.3f} → {best_name}"

    return {
        'model': best_model, 'direct_model': direct_model, 'resid_model': resid_model,
        'method': best_name, 'is_residual': is_residual,
        'blend_alpha': best_alpha if best_name == 'blend' else None,
        'mae': best_mae, 'rmse': best_rmse,
        'info_str': info_str,
    }


def _train_precipitation_twostage(X_tr, y_tr, X_te, y_te, X_val, y_val, feature_cols):
    """Two-stage precipitation: classifier + regressor. Uses ALL data, no splits."""
    RAIN_THRESH = 0.1

    y_cls_tr = (y_tr >= RAIN_THRESH).astype(int)
    y_cls_val = (y_val >= RAIN_THRESH).astype(int)
    rain_ratio = y_cls_tr.mean()

    cls_hp = dict(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=10, gamma=0.1,
        scale_pos_weight=max(1.0, (1 - rain_ratio) / max(rain_ratio, 0.01)),
        objective='binary:logistic', eval_metric='logloss',
        random_state=42, n_jobs=-1, early_stopping_rounds=40
    )
    cls_val_model = xgb.XGBClassifier(**cls_hp)
    cls_val_model.fit(X_tr, y_cls_tr, eval_set=[(X_val, y_cls_val)], verbose=False)
    cls_best_n = max(cls_val_model.best_iteration + 1, 50)

    proba_val = cls_val_model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.05):
        pred_cls = (proba_val >= t).astype(int)
        f1 = f1_score(y_cls_val, pred_cls, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    cls_hp_final = {k: v for k, v in cls_hp.items() if k != 'early_stopping_rounds'}
    cls_hp_final['n_estimators'] = cls_best_n
    X_cls_full = pd.concat([X_tr, X_val], axis=0)
    y_cls_full = pd.concat([y_cls_tr, y_cls_val], axis=0)
    cls_model = xgb.XGBClassifier(**cls_hp_final)
    cls_model.fit(X_cls_full, y_cls_full, verbose=False)

    cls_proba_te = cls_model.predict_proba(X_te)[:, 1]

    rain_mask_tr = y_tr >= RAIN_THRESH
    rain_mask_val = y_val >= RAIN_THRESH

    reg_hp = dict(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=3.0,
        min_child_weight=10, gamma=0.15,
        objective='reg:absoluteerror', random_state=42, n_jobs=-1,
        early_stopping_rounds=50
    )

    if rain_mask_tr.sum() >= 100 and rain_mask_val.sum() >= 20:
        reg_val_model = xgb.XGBRegressor(**reg_hp)
        y_rain_tr_sqrt = np.sqrt(y_tr[rain_mask_tr])
        y_rain_val_sqrt = np.sqrt(y_val[rain_mask_val])
        reg_val_model.fit(X_tr[rain_mask_tr], y_rain_tr_sqrt,
                          eval_set=[(X_val[rain_mask_val], y_rain_val_sqrt)], verbose=False)
        reg_best_n = max(reg_val_model.best_iteration + 1, 50)

        reg_hp_final = {k: v for k, v in reg_hp.items() if k != 'early_stopping_rounds'}
        reg_hp_final['n_estimators'] = reg_best_n
        y_full = pd.concat([y_tr, y_val], axis=0)
        X_full = pd.concat([X_tr, X_val], axis=0)
        rain_mask_full = y_full >= RAIN_THRESH
        reg_model = xgb.XGBRegressor(**reg_hp_final)
        reg_model.fit(X_full[rain_mask_full], np.sqrt(y_full[rain_mask_full]), verbose=False)
        reg_pred_te_sqrt = reg_model.predict(X_te)
        reg_pred_te = np.square(np.clip(reg_pred_te_sqrt, 0, None))
    else:
        reg_val_model = xgb.XGBRegressor(**reg_hp)
        reg_val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        reg_best_n = max(reg_val_model.best_iteration + 1, 50)
        reg_hp_final = {k: v for k, v in reg_hp.items() if k != 'early_stopping_rounds'}
        reg_hp_final['n_estimators'] = reg_best_n
        X_full = pd.concat([X_tr, X_val], axis=0)
        y_full = pd.concat([y_tr, y_val], axis=0)
        reg_model = xgb.XGBRegressor(**reg_hp_final)
        reg_model.fit(X_full, y_full, verbose=False)
        reg_pred_te = np.clip(reg_model.predict(X_te), 0, None)

    single_hp = dict(
        n_estimators=1000, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=3.0,
        min_child_weight=15, gamma=0.2,
        objective='reg:absoluteerror', random_state=42, n_jobs=-1,
        early_stopping_rounds=50
    )
    single_val_model = xgb.XGBRegressor(**single_hp)
    single_val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    single_best_n = max(single_val_model.best_iteration + 1, 50)
    single_hp_final = {k: v for k, v in single_hp.items() if k != 'early_stopping_rounds'}
    single_hp_final['n_estimators'] = single_best_n
    X_full_s = pd.concat([X_tr, X_val], axis=0)
    y_full_s = pd.concat([y_tr, y_val], axis=0)
    single_model = xgb.XGBRegressor(**single_hp_final)
    single_model.fit(X_full_s, y_full_s, verbose=False)
    single_pred = np.clip(single_model.predict(X_te), 0, None)
    single_pred[single_pred < RAIN_THRESH] = 0.0

    hard_pred = np.where(cls_proba_te >= best_thresh, reg_pred_te, 0.0)
    soft_pred = cls_proba_te * reg_pred_te
    sharp_pred = np.where(
        cls_proba_te >= best_thresh,
        0.7 * reg_pred_te + 0.3 * single_pred,
        single_pred * cls_proba_te
    )
    confidence = np.abs(cls_proba_te - 0.5) * 2
    adaptive_pred = np.where(
        cls_proba_te >= best_thresh,
        confidence * reg_pred_te + (1 - confidence) * single_pred,
        (1 - confidence) * single_pred * 0.5
    )

    methods = {
        'single': (np.clip(single_pred, 0, None), single_model),
        'hard': (np.clip(hard_pred, 0, None), None),
        'soft': (np.clip(soft_pred, 0, None), None),
        'sharp': (np.clip(sharp_pred, 0, None), None),
        'adaptive': (np.clip(adaptive_pred, 0, None), None),
    }

    method_maes = {}
    for name, (pred, _) in methods.items():
        method_maes[name] = mean_absolute_error(y_te, pred)

    best_method = min(method_maes, key=method_maes.get)
    best_pred = methods[best_method][0]
    mae = method_maes[best_method]
    rmse = np.sqrt(mean_squared_error(y_te, best_pred))

    print(f"\n  >> PADAVINE: Two-stage model (klasifikacija + regresija)")
    print(f"    Stage 1 (cls): rain_ratio={rain_ratio:.3f}, thresh={best_thresh:.2f}, F1={best_f1:.3f}")
    print(f"    Stage 2 (reg): train_rain={rain_mask_tr.sum()}, test_rain={(y_te >= RAIN_THRESH).sum()}")
    print(f"    Methods: " + ", ".join(f"{k}={v:.3f}" for k, v in method_maes.items()) + f" → BEST={best_method}")

    return {
        'cls_model': cls_model, 'reg_model': reg_model, 'single_model': single_model,
        'best_method': best_method, 'threshold': best_thresh,
        'mae': mae, 'rmse': rmse, 'features': feature_cols,
        'use_sqrt': rain_mask_tr.sum() >= 100,
    }


def train_all_models(df):
    """Unified training: all params use full 50K dataset. No splits.
    - Precipitation: two-stage (cls+reg) + optional blend
    - Everything else: residual+blended (direct/residual/blend)"""
    print("\n[3/6] Treniranje XGBoost modela...")

    feature_cols = get_feature_columns(df)
    print(f"  Feature-a za treniranje: {len(feature_cols)}")

    trained = {}
    results = {}

    for param, info in TARGET_PARAMS.items():
        obs_col = info['obs']
        if obs_col not in df.columns:
            print(f"  {info['display']:20s} --- SKIP (nema obs)")
            continue

        y = pd.to_numeric(df[obs_col], errors='coerce')
        valid = y.notna()
        if param == 'cloud_cover':
            valid = valid & (df.get('is_daytime', pd.Series(1, index=df.index)) > 0)

        df_v = df[valid].copy()
        y_v = y[valid]

        if len(df_v) < 500:
            print(f"  {info['display']:20s} --- SKIP ({len(df_v)} redova)")
            continue

        tr = df_v['datetime'] < SPLIT_DATE
        te = df_v['datetime'] >= SPLIT_DATE

        vf = [c for c in feature_cols if c in df_v.columns
              and df_v[c].notna().sum() > len(df_v) * 0.3]

        FILL = -999
        X_tr, y_tr = df_v.loc[tr, vf].fillna(FILL), y_v[tr]
        X_te, y_te = df_v.loc[te, vf].fillna(FILL), y_v[te]

        if len(X_tr) < 300 or len(X_te) < 50:
            print(f"  {info['display']:20s} --- SKIP (train={len(X_tr)}, test={len(X_te)})")
            continue

        if param == 'precipitation':
            X_train_p, y_train_p, X_val_p, y_val_p = _make_val_split(X_tr, y_tr)
            precip_result = _train_precipitation_twostage(
                X_train_p, y_train_p, X_te, y_te, X_val_p, y_val_p, vf
            )
            mae = precip_result['mae']
            rmse = precip_result['rmse']

            ens_col = f'{param}_ens_mean'
            blend_alpha = 1.0
            if ens_col in df_v.columns:
                ens_te_vals = pd.to_numeric(df_v.loc[te, ens_col], errors='coerce').fillna(0).values
                best_method = precip_result['best_method']
                RAIN_THRESH = 0.1
                cls_proba_te = precip_result['cls_model'].predict_proba(X_te)[:, 1]
                thresh = precip_result['threshold']
                if precip_result.get('use_sqrt', False):
                    reg_pred_te = np.square(np.clip(precip_result['reg_model'].predict(X_te), 0, None))
                else:
                    reg_pred_te = np.clip(precip_result['reg_model'].predict(X_te), 0, None)
                single_pred_te = np.clip(precip_result['single_model'].predict(X_te), 0, None)
                single_pred_te[single_pred_te < RAIN_THRESH] = 0.0

                if best_method == 'hard':
                    xgb_pred = np.where(cls_proba_te >= thresh, reg_pred_te, 0.0)
                elif best_method == 'soft':
                    xgb_pred = cls_proba_te * reg_pred_te
                elif best_method == 'sharp':
                    xgb_pred = np.where(cls_proba_te >= thresh, 0.7 * reg_pred_te + 0.3 * single_pred_te, single_pred_te * cls_proba_te)
                elif best_method == 'adaptive':
                    confidence = np.abs(cls_proba_te - 0.5) * 2
                    xgb_pred = np.where(cls_proba_te >= thresh, confidence * reg_pred_te + (1 - confidence) * single_pred_te, (1 - confidence) * single_pred_te * 0.5)
                else:
                    xgb_pred = single_pred_te
                xgb_pred = np.clip(xgb_pred, 0, None)

                b_alpha, b_mae, _ = _find_optimal_blend(xgb_pred, y_te.values, ens_te_vals)
                if b_mae < mae:
                    mae = b_mae
                    blend_alpha = b_alpha
                    precip_result['blend_alpha'] = blend_alpha
                    print(f"    Blend improved: alpha={b_alpha:.3f}, MAE={b_mae:.3f}")

            best_mae, best_m = float('inf'), ""
            for m in MODELS:
                mc = f"{m}_{param}_model"
                if mc in df_v.columns:
                    mv = pd.to_numeric(df_v.loc[te, mc], errors='coerce')
                    vv = mv.notna() & y_te.notna()
                    if vv.sum() > 50:
                        mm = mean_absolute_error(y_te[vv], mv[vv])
                        if mm < best_mae:
                            best_mae, best_m = mm, m

            ens_mae = float('inf')
            if ens_col in df_v.columns:
                ev = pd.to_numeric(df_v.loc[te, ens_col], errors='coerce')
                vv = ev.notna() & y_te.notna()
                if vv.sum() > 50:
                    ens_mae = mean_absolute_error(y_te[vv], ev[vv])

            impr = (best_mae - mae) / best_mae * 100 if best_mae < float('inf') else 0
            print(f"  {info['display']:20s} MAE: {mae:.3f}{info['unit']:5s} "
                  f"| best: {best_mae:.3f} ({best_m[:8]:8s}) "
                  f"| ens: {ens_mae:.3f} "
                  f"| +{impr:.1f}%")

            precip_result['cls_model'].save_model(os.path.join(MODEL_DIR, f"xgb_{param}_cls.json"))
            precip_result['reg_model'].save_model(os.path.join(MODEL_DIR, f"xgb_{param}_reg.json"))
            precip_result['single_model'].save_model(os.path.join(MODEL_DIR, f"xgb_{param}.json"))

            trained[param] = {
                'precip_info': precip_result,
                'features': vf,
                'mae': mae, 'rmse': rmse,
                'best_model': best_m, 'best_model_mae': best_mae,
                'ensemble_mae': ens_mae, 'improvement': impr,
            }
            results[param] = {
                'mae': round(mae, 3), 'rmse': round(rmse, 3),
                'unit': info['unit'], 'display': info['display'],
                'improvement': round(impr, 1),
                'best_model': best_m, 'best_model_mae': round(best_mae, 3),
                'ensemble_mae': round(ens_mae, 3),
                'method': precip_result['best_method'],
                'is_residual': False,
                'blend_alpha': float(precip_result.get('blend_alpha', 1.0)),
                'threshold': float(precip_result['threshold']),
                'use_sqrt': bool(precip_result.get('use_sqrt', False)),
                'is_precip': True,
            }
            continue

        if param in ('temperature_2m', 'dew_point_2m', 'pressure_msl'):
            hp = dict(n_estimators=1200, max_depth=6, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                      reg_lambda=1.0, min_child_weight=5, gamma=0.05,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=40)
        elif param in ('cloud_cover', 'shortwave_radiation'):
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.65, reg_alpha=0.2,
                      reg_lambda=1.5, min_child_weight=5, gamma=0.05,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=40)
        elif param == 'relative_humidity_2m':
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.65, reg_alpha=0.15,
                      reg_lambda=1.5, min_child_weight=5, gamma=0.05,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=40)
        else:
            hp = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                      reg_lambda=1.0, min_child_weight=5, gamma=0.05,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)

        ens_col = f'{param}_ens_mean'
        rb_result = _train_residual_blended(
            X_tr, y_tr, X_te, y_te, hp, param, ens_col,
            df_v.loc[tr], df_v.loc[te]
        )

        method_str = rb_result['method']
        model_obj = rb_result['model']
        y_pred = model_obj.predict(X_te)

        if rb_result['is_residual']:
            ens_te_vals = pd.to_numeric(df_v.loc[te, ens_col], errors='coerce').fillna(0).values if ens_col in df_v.columns else np.zeros(len(X_te))
            y_pred = ens_te_vals + y_pred
        elif rb_result.get('blend_alpha') is not None:
            ens_te_vals = pd.to_numeric(df_v.loc[te, ens_col], errors='coerce').fillna(0).values if ens_col in df_v.columns else np.zeros(len(X_te))
            alpha = rb_result['blend_alpha']
            y_pred = alpha * y_pred + (1 - alpha) * ens_te_vals

        if param == 'relative_humidity_2m':
            y_pred = np.clip(y_pred, 0, 100)
        elif param == 'cloud_cover':
            y_pred = np.clip(y_pred, 0, 100)
        elif param in ['wind_speed_10m', 'wind_gusts_10m', 'shortwave_radiation']:
            y_pred = np.clip(y_pred, 0, None)

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        best_mae, best_m = float('inf'), ""
        for m in MODELS:
            mc = f"{m}_{param}_model"
            if mc in df_v.columns:
                mv = pd.to_numeric(df_v.loc[te, mc], errors='coerce')
                vv = mv.notna() & y_te.notna()
                if vv.sum() > 50:
                    mm = mean_absolute_error(y_te[vv], mv[vv])
                    if mm < best_mae:
                        best_mae, best_m = mm, m

        ens_mae = float('inf')
        if ens_col in df_v.columns:
            ev = pd.to_numeric(df_v.loc[te, ens_col], errors='coerce')
            vv = ev.notna() & y_te.notna()
            if vv.sum() > 50:
                ens_mae = mean_absolute_error(y_te[vv], ev[vv])

        impr = (best_mae - mae) / best_mae * 100 if best_mae < float('inf') else 0

        print(f"    {rb_result['info_str']}")
        print(f"  {info['display']:20s} MAE: {mae:.3f}{info['unit']:5s} "
              f"| best: {best_mae:.3f} ({best_m[:8]:8s}) "
              f"| ens: {ens_mae:.3f} "
              f"| +{impr:.1f}%")

        rb_result['direct_model'].save_model(os.path.join(MODEL_DIR, f"xgb_{param}.json"))
        if rb_result.get('resid_model'):
            rb_result['resid_model'].save_model(os.path.join(MODEL_DIR, f"xgb_{param}_resid.json"))

        trained[param] = {
            'model': rb_result['model'],
            'direct_model': rb_result['direct_model'],
            'resid_model': rb_result.get('resid_model'),
            'method': method_str,
            'is_residual': rb_result['is_residual'],
            'blend_alpha': rb_result.get('blend_alpha'),
            'features': vf,
            'mae': mae, 'rmse': rmse,
            'best_model': best_m, 'best_model_mae': best_mae,
            'ensemble_mae': ens_mae, 'improvement': impr,
        }
        results[param] = {
            'mae': round(mae, 3), 'rmse': round(rmse, 3),
            'unit': info['unit'], 'display': info['display'],
            'improvement': round(impr, 1),
            'best_model': best_m, 'best_model_mae': round(best_mae, 3),
            'ensemble_mae': round(ens_mae, 3),
            'method': method_str,
            'is_residual': bool(rb_result['is_residual']),
            'blend_alpha': float(rb_result['blend_alpha']) if rb_result.get('blend_alpha') is not None else None,
            'is_precip': False,
        }

    with open(os.path.join(MODEL_DIR, 'feature_lists.json'), 'w') as f:
        json.dump({k: v['features'] for k, v in trained.items()}, f)
    with open(os.path.join(MODEL_DIR, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return trained, results


def load_trained_models():
    """Load pre-trained XGBoost models + metadata from disk. No historical data needed."""
    print("\n[3/6] Ucitavanje SACUVANIH modela (--skip-training)...")
    results_path = os.path.join(MODEL_DIR, 'training_results.json')
    features_path = os.path.join(MODEL_DIR, 'feature_lists.json')
    bias_path = os.path.join(MODEL_DIR, 'bias_tables.json')

    if not os.path.exists(results_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Nema sacuvanih modela u {MODEL_DIR}. Pokrenite prvo bez --skip-training.")

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(features_path, 'r') as f:
        feature_lists = json.load(f)

    bias_tables = {}
    if os.path.exists(bias_path):
        with open(bias_path, 'r') as f:
            bt_raw = json.load(f)
        for k, v in bt_raw.items():
            bias_tables[k] = pd.DataFrame(v)

    trained = {}
    for param, rinfo in results.items():
        features = feature_lists.get(param, [])
        if not features:
            continue

        if rinfo.get('is_precip', False):
            cls_path = os.path.join(MODEL_DIR, f"xgb_{param}_cls.json")
            reg_path = os.path.join(MODEL_DIR, f"xgb_{param}_reg.json")
            single_path = os.path.join(MODEL_DIR, f"xgb_{param}.json")
            if not all(os.path.exists(p) for p in [cls_path, reg_path, single_path]):
                print(f"  {rinfo['display']:20s} --- SKIP (fajlovi ne postoje)")
                continue

            cls_model = xgb.XGBClassifier()
            cls_model.load_model(cls_path)
            reg_model = xgb.XGBRegressor()
            reg_model.load_model(reg_path)
            single_model = xgb.XGBRegressor()
            single_model.load_model(single_path)

            trained[param] = {
                'precip_info': {
                    'cls_model': cls_model,
                    'reg_model': reg_model,
                    'single_model': single_model,
                    'best_method': rinfo['method'],
                    'threshold': rinfo.get('threshold', 0.35),
                    'use_sqrt': rinfo.get('use_sqrt', False),
                    'blend_alpha': rinfo.get('blend_alpha', 1.0),
                },
                'features': features,
                'mae': rinfo['mae'], 'rmse': rinfo['rmse'],
                'best_model': rinfo.get('best_model', ''),
                'best_model_mae': rinfo.get('best_model_mae', 0),
                'ensemble_mae': rinfo.get('ensemble_mae', 0),
                'improvement': rinfo.get('improvement', 0),
            }
            print(f"  {rinfo['display']:20s} loaded (MAE={rinfo['mae']}) [{rinfo['method']}]")
        else:
            direct_path = os.path.join(MODEL_DIR, f"xgb_{param}.json")
            resid_path = os.path.join(MODEL_DIR, f"xgb_{param}_resid.json")
            if not os.path.exists(direct_path):
                print(f"  {rinfo['display']:20s} --- SKIP (fajl ne postoji)")
                continue

            direct_model = xgb.XGBRegressor()
            direct_model.load_model(direct_path)

            is_residual = rinfo.get('is_residual', False)
            resid_model = None
            if is_residual and os.path.exists(resid_path):
                resid_model = xgb.XGBRegressor()
                resid_model.load_model(resid_path)

            if is_residual and resid_model is not None:
                active_model = resid_model
            else:
                active_model = direct_model

            trained[param] = {
                'model': active_model,
                'direct_model': direct_model,
                'resid_model': resid_model,
                'method': rinfo.get('method', 'direct'),
                'is_residual': is_residual,
                'blend_alpha': rinfo.get('blend_alpha'),
                'features': features,
                'mae': rinfo['mae'], 'rmse': rinfo['rmse'],
                'best_model': rinfo.get('best_model', ''),
                'best_model_mae': rinfo.get('best_model_mae', 0),
                'ensemble_mae': rinfo.get('ensemble_mae', 0),
                'improvement': rinfo.get('improvement', 0),
            }
            print(f"  {rinfo['display']:20s} loaded (MAE={rinfo['mae']}) [{rinfo.get('method', 'direct')}]")

    print(f"  Ucitano {len(trained)}/{len(results)} modela.")
    return trained, results, bias_tables


def fetch_live_forecasts():
    print("\n[4/6] Preuzimanje LIVE prognoza...")
    URL = "https://api.open-meteo.com/v1/forecast"
    all_fc = {}
    for model_name, model_id in MODEL_IDS.items():
        print(f"  {model_name}...", end=" ")
        params = {
            "latitude": LAT, "longitude": LON,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "auto", "temperature_unit": "celsius",
            "wind_speed_unit": "ms", "precipitation_unit": "mm",
            "models": model_id, "forecast_days": 10,
        }
        for attempt in range(3):
            try:
                r = requests.get(URL, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(60); continue
                r.raise_for_status()
                h = r.json().get('hourly', {})
                d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
                for v in HOURLY_VARS:
                    if v in h:
                        d[f"{model_name}_{v}_model"] = h[v]
                all_fc[model_name] = d
                print(f"OK ({len(d)}h)")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FAIL: {e}")
                else:
                    time.sleep(5)
        time.sleep(1.5)

    if not all_fc:
        raise RuntimeError("Nema prognoza!")

    merged = list(all_fc.values())[0]
    for k in list(all_fc.keys())[1:]:
        merged = merged.merge(all_fc[k], on='datetime', how='outer')
    merged.sort_values('datetime', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    now = pd.Timestamp.now().floor('h')
    mask = merged['datetime'] >= now
    fc_all = merged[mask].copy().reset_index(drop=True)
    print(f"  Prognoza: {fc_all.shape[0]} sati ({fc_all['datetime'].min()} --- {fc_all['datetime'].max()})")

    print("\n  Preuzimanje Previous Runs (Day1/Day2)...")
    prev_hourly_list = []
    for v in PREV_RUNS_VARS:
        prev_hourly_list.append(v)
        prev_hourly_list.append(f"{v}_previous_day1")
        prev_hourly_list.append(f"{v}_previous_day2")
    prev_hourly_str = ",".join(prev_hourly_list)

    for model_name in PREV_RUNS_MODELS:
        if model_name not in all_fc:
            continue
        model_id = MODEL_IDS[model_name]
        pr_params = {
            "latitude": LAT, "longitude": LON,
            "hourly": prev_hourly_str,
            "timezone": "auto", "models": model_id, "forecast_days": 10,
        }
        try:
            r = requests.get(PREV_RUNS_API, params=pr_params, timeout=30)
            if r.status_code == 429:
                time.sleep(60)
                r = requests.get(PREV_RUNS_API, params=pr_params, timeout=30)
            r.raise_for_status()
            h = r.json().get('hourly', {})
            d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
            added = 0
            for v in PREV_RUNS_VARS:
                for lag in ['previous_day1', 'previous_day2']:
                    col = f"{v}_{lag}"
                    new_col = f"{model_name}_{v}_{lag}"
                    if col in h:
                        d[new_col] = h[col]
                        added += 1
            fc_all = fc_all.merge(d[['datetime'] + [c for c in d.columns if c != 'datetime']], 
                                   on='datetime', how='left')
            print(f"    {model_name}: OK ({added} columns)")
        except Exception as e:
            print(f"    {model_name}: FAIL ({e})")
        time.sleep(1.5)

    return fc_all


def correct_weather_code_row(row, raw_row=None):
    """
    Models often report rain (WC >= 51) during winter overcast conditions
    when it's actually just cloudy. XGBoost precipitation correction is more accurate,
    so we trust it over the raw weather code mode.
    
    This is how we're going to fix:
    - If XGBoost says precip < 0.1mm AND raw WC is rain/drizzle → downgrade to cloud-based code
    - If XGBoost says precip > 0 but raw WC is clear → upgrade to appropriate rain code
    - Use cloud cover to determine the correct non-rain code
    """
    wc_raw = int(row.get('weather_code_raw', row.get('weather_code', 0)))
    if pd.isna(wc_raw):
        wc_raw = 0
    wc_raw = int(wc_raw)
    
    precip_xgb = row.get('precipitation_xgb', None)
    cloud_xgb = row.get('cloud_cover_xgb', None)
    
    is_rain_code = 51 <= wc_raw <= 82  # drizzle + rain + showers
    is_snow_code = 71 <= wc_raw <= 77
    is_thunderstorm = wc_raw >= 95
    
    # We don't want to mess with thunderstorm codes, as they are often correct and XGBoost precip can be underestimated in convective events
    if is_thunderstorm:
        return wc_raw
    
    if is_rain_code and precip_xgb is not None and pd.notna(precip_xgb) and precip_xgb < 0.1:
        if cloud_xgb is not None and pd.notna(cloud_xgb):
            if cloud_xgb > 80:
                return 3   
            elif cloud_xgb > 50:
                return 2   
            elif cloud_xgb > 20:
                return 1   
            else:
                return 0   
        return 3 
    
    if 51 <= wc_raw <= 55 and precip_xgb is not None and pd.notna(precip_xgb):
        if precip_xgb < 0.2:
            if cloud_xgb is not None and pd.notna(cloud_xgb) and cloud_xgb > 85:
                return 3  
    
    if wc_raw <= 3 and precip_xgb is not None and pd.notna(precip_xgb) and precip_xgb > 0.5:
        if precip_xgb > 3.0:
            return 63  
        elif precip_xgb > 1.0:
            return 61  
        else:
            return 51  
    
    if wc_raw <= 3 and cloud_xgb is not None and pd.notna(cloud_xgb):
        if cloud_xgb > 80:
            return 3   
        elif cloud_xgb > 50:
            return 2   
        elif cloud_xgb > 20:
            return 1  
        else:
            return 0   
    
    return wc_raw


def apply_correction(fc_df, trained, bias_tables):
    print("\n[5/6] Primjena korekcije...")

    fc = apply_bias_features(fc_df.copy(), bias_tables)
    fc = engineer_features(fc)

    corrected = fc[['datetime']].copy()

    for param in TARGET_PARAMS:
        ens = f'{param}_ens_mean'
        if ens in fc.columns:
            corrected[f'{param}_ensemble'] = fc[ens]

    for param, minfo in trained.items():
        features = minfo['features']
        available = [f for f in features if f in fc.columns]

        if len(available) < len(features) * 0.4:
            print(f"  {TARGET_PARAMS[param]['display']:20s} --- nedovoljno feature-a ({len(available)}/{len(features)})")
            continue

        X = fc[available].copy()
        for c in features:
            if c not in X.columns:
                X[c] = -999
        X = X[features].fillna(-999)

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(-999)

        if param == 'precipitation' and 'precip_info' in minfo:
            pinfo = minfo['precip_info']
            method = pinfo['best_method']

            cls_proba = pinfo['cls_model'].predict_proba(X)[:, 1]
            thresh = pinfo['threshold']

            if pinfo.get('use_sqrt', False):
                reg_pred = np.square(np.clip(pinfo['reg_model'].predict(X), 0, None))
            else:
                reg_pred = np.clip(pinfo['reg_model'].predict(X), 0, None)

            single_pred = np.clip(pinfo['single_model'].predict(X), 0, None)
            single_pred[single_pred < 0.1] = 0.0

            if method == 'hard':
                pred = np.where(cls_proba >= thresh, reg_pred, 0.0)
            elif method == 'soft':
                pred = cls_proba * reg_pred
            elif method == 'sharp':
                pred = np.where(cls_proba >= thresh, 0.7 * reg_pred + 0.3 * single_pred, single_pred * cls_proba)
            elif method == 'adaptive':
                confidence = np.abs(cls_proba - 0.5) * 2
                pred = np.where(cls_proba >= thresh, confidence * reg_pred + (1 - confidence) * single_pred, (1 - confidence) * single_pred * 0.5)
            else:
                pred = single_pred

            pred = np.clip(pred, 0, 50)
            p_blend_alpha = pinfo.get('blend_alpha', 1.0)
            if p_blend_alpha < 1.0:
                ens_col_p = f'{param}_ens_mean'
                ens_vals_p = pd.to_numeric(fc[ens_col_p], errors='coerce').fillna(0).values if ens_col_p in fc.columns else np.zeros(len(X))
                pred = p_blend_alpha * pred + (1 - p_blend_alpha) * ens_vals_p
                pred = np.clip(pred, 0, 50)

            corrected[f'{param}_xgb'] = pred
            method_lbl = method + (f'+blend({p_blend_alpha:.2f})' if p_blend_alpha < 1.0 else '')
            print(f"  {TARGET_PARAMS[param]['display']:20s} (MAE={minfo['mae']:.3f}{TARGET_PARAMS[param]['unit']}) [{method_lbl}]")
            continue

        method_name = minfo.get('method', 'direct')
        model = minfo['model']
        pred = model.predict(X)

        if minfo.get('is_residual'):
            ens_col = f'{param}_ens_mean'
            ens_vals = pd.to_numeric(fc[ens_col], errors='coerce').fillna(0).values if ens_col in fc.columns else np.zeros(len(X))
            pred = ens_vals + pred
        elif minfo.get('blend_alpha') is not None:
            ens_col = f'{param}_ens_mean'
            ens_vals = pd.to_numeric(fc[ens_col], errors='coerce').fillna(0).values if ens_col in fc.columns else np.zeros(len(X))
            alpha = minfo['blend_alpha']
            pred = alpha * pred + (1 - alpha) * ens_vals

        if param == 'relative_humidity_2m':
            pred = np.clip(pred, 0, 100)
        elif param == 'cloud_cover':
            pred = np.clip(pred, 0, 100)
            # One issue that I've had with clouds is winter morning overcast scenarios, the XGBoost model regularly overpowers, so to fix the XGBoost offset: when ensemble unanimously says clear sky,
            # don't let XGBoost hallucinate clouds from climatology.
            ens_col_cc = f'{param}_ens_mean'
            if ens_col_cc in fc.columns:
                ens_cc = pd.to_numeric(fc[ens_col_cc], errors='coerce').fillna(0).values
                # If ensemble < 10%, cap XGBoost at ensemble + 30%
                low_ens = ens_cc < 10
                if low_ens.any():
                    pred[low_ens] = np.minimum(pred[low_ens], ens_cc[low_ens] + 30)
                # If ensemble > 90%, floor XGBoost at ensemble - 30%
                high_ens = ens_cc > 90
                if high_ens.any():
                    pred[high_ens] = np.maximum(pred[high_ens], ens_cc[high_ens] - 30)
                pred = np.clip(pred, 0, 100)
        elif param in ['wind_speed_10m', 'wind_gusts_10m', 'shortwave_radiation']:
            pred = np.clip(pred, 0, None)

        corrected[f'{param}_xgb'] = pred
        print(f"  {TARGET_PARAMS[param]['display']:20s} (MAE={minfo['mae']:.3f}{TARGET_PARAMS[param]['unit']}) [{method_name}]")

    wc_cols = [f"{m}_weather_code_model" for m in MODELS if f"{m}_weather_code_model" in fc.columns]
    if wc_cols:
        corrected['weather_code_raw'] = fc[wc_cols].apply(pd.to_numeric, errors='coerce').mode(axis=1)[0]
        corrected['weather_code'] = corrected.apply(
            lambda r: correct_weather_code_row(r, fc.loc[r.name] if r.name in fc.index else None), axis=1
        )

    for extra in ['apparent_temperature', 'snowfall', 'rain',
                  'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
                  'wind_direction_10m', 'surface_pressure']:
        ec = [f"{m}_{extra}_model" for m in MODELS if f"{m}_{extra}_model" in fc.columns]
        if ec:
            corrected[f'{extra}_ens'] = fc[ec].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    return corrected


WMO_CODES = {
    0: {"desc": "Vedro", "icon": "clear", "emoji": "\u2600\ufe0f"},
    1: {"desc": "Pretezno vedro", "icon": "mostly_clear", "emoji": "\U0001f324\ufe0f"},
    2: {"desc": "Djelimicno oblacno", "icon": "partly_cloudy", "emoji": "\u26c5"},
    3: {"desc": "Oblacno", "icon": "cloudy", "emoji": "\u2601\ufe0f"},
    45: {"desc": "Magla", "icon": "fog", "emoji": "\U0001f32b\ufe0f"},
    48: {"desc": "Magla (inje)", "icon": "fog", "emoji": "\U0001f32b\ufe0f"},
    51: {"desc": "Sitna kisa", "icon": "light_rain", "emoji": "\U0001f326\ufe0f"},
    53: {"desc": "Sitna kisa, umjerena", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    55: {"desc": "Sitna kisa, jaka", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    61: {"desc": "Slaba kisa", "icon": "light_rain", "emoji": "\U0001f326\ufe0f"},
    63: {"desc": "Umjerena kisa", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    65: {"desc": "Jaka kisa", "icon": "heavy_rain", "emoji": "\U0001f327\ufe0f\U0001f327\ufe0f"},
    71: {"desc": "Slab snijeg", "icon": "snow", "emoji": "\U0001f328\ufe0f"},
    73: {"desc": "Umjeren snijeg", "icon": "snow", "emoji": "\U0001f328\ufe0f"},
    75: {"desc": "Jak snijeg", "icon": "heavy_snow", "emoji": "\u2744\ufe0f\u2744\ufe0f"},
    80: {"desc": "Slabi pljuskovi", "icon": "light_rain", "emoji": "\U0001f326\ufe0f"},
    81: {"desc": "Umjereni pljuskovi", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    82: {"desc": "Jaki pljuskovi", "icon": "heavy_rain", "emoji": "\u26c8\ufe0f"},
    95: {"desc": "Grmljavina", "icon": "thunderstorm", "emoji": "\u26c8\ufe0f"},
    96: {"desc": "Grmljavina + grad", "icon": "thunderstorm", "emoji": "\u26c8\ufe0f\U0001f9ca"},
    99: {"desc": "Jaka grmljavina", "icon": "thunderstorm", "emoji": "\u26c8\ufe0f\U0001f9ca"},
}


def _daily_narrative(grp):
    """
    Expects DataFrame with columns: hour, cloud_cover, precipitation,
    wind_speed_10m, wind_gusts_10m, temperature_2m, weather_code.
    """
    def _col(name):
        s = pd.to_numeric(grp.get(name, pd.Series(dtype=float)), errors='coerce')
        return s

    hr = grp['hour']
    cc = _col('cloud_cover')
    pr = _col('precipitation')
    ws = _col('wind_speed_10m')
    wg = _col('wind_gusts_10m')
    tp = _col('temperature_2m')
    wc = _col('weather_code')

    def _period(h0, h1):
        mask = (hr >= h0) & (hr < h1)
        sub_cc = cc[mask].dropna()
        sub_pr = pr[mask].dropna()
        sub_wc = wc[mask].dropna()
        return {
            'cloud': float(sub_cc.mean()) if len(sub_cc) else None,
            'precip': float(sub_pr.sum()) if len(sub_pr) else 0,
            'has_rain': float(sub_pr.sum()) > 0.1 if len(sub_pr) else False,
            'has_thunder': bool((sub_wc >= 95).any()) if len(sub_wc) else False,
            'has_snow': bool(((sub_wc >= 71) & (sub_wc <= 75)).any()) if len(sub_wc) else False,
            'has_fog': bool(((sub_wc >= 45) & (sub_wc <= 48)).any()) if len(sub_wc) else False,
            'n': int(mask.sum()),
        }

    morn = _period(6, 12)
    aftn = _period(12, 18)
    eve = _period(18, 24)

    total_precip = float(pr.sum()) if pr.notna().any() else 0
    max_wind = float(ws.max()) if ws.notna().any() else 0
    max_gust = float(wg.max()) if wg.notna().any() else 0
    temp_max = float(tp.max()) if tp.notna().any() else 15
    temp_min = float(tp.min()) if tp.notna().any() else 10

    def sky(c):
        if c is None:
            return 'unknown'
        if c < 20:
            return 'clear'
        if c < 40:
            return 'mostly_clear'
        if c < 65:
            return 'partly_cloudy'
        if c < 85:
            return 'mostly_cloudy'
        return 'cloudy'

    ms, as_, es = sky(morn['cloud']), sky(aftn['cloud']), sky(eve['cloud'])
    rain_m, rain_a, rain_e = morn['has_rain'], aftn['has_rain'], eve['has_rain']
    has_thunder = morn['has_thunder'] or aftn['has_thunder'] or eve['has_thunder']
    has_snow = morn['has_snow'] or aftn['has_snow'] or eve['has_snow']
    has_fog_morn = morn['has_fog']

    parts = []

    if has_snow:
        if rain_m and not rain_a:
            parts.append("Snijeg ujutru, prestanak poslijepodne")
        elif not rain_m and rain_a:
            parts.append("Suvo ujtro, snijeg poslijepodne")
        elif rain_m and rain_a and rain_e:
            parts.append("Snijeg tokom cijelog dana")
        elif total_precip >= 5:
            parts.append("Obilniji snijeg")
        else:
            parts.append("Povremeni snijeg")
    elif has_thunder:
        if not rain_m and rain_a and not rain_e:
            parts.append("Sunce pa grmljavina poslijepodne")
        elif rain_m and not rain_a:
            parts.append("Grmljavina ujutru, razvedravanje poslijepodne")
        elif not rain_m and not rain_a and rain_e:
            parts.append("Suvo tokom dana, grmljavina uveče")
        elif rain_m and rain_a:
            parts.append("Oblačno uz povremenu grmljavinu")
        else:
            parts.append("Oblačno sa grmljavinom")
    elif total_precip > 0.2:
        if rain_m and rain_a and rain_e:
            if total_precip >= 10:
                parts.append("Obilna kiša tokom cijelog dana")
            elif total_precip >= 3:
                parts.append("Kiša tokom cijelog dana")
            else:
                parts.append("Povremena slaba kiša")
        elif rain_m and rain_a and not rain_e:
            parts.append("Kiša ujutru i poslijepodne, suvo uveče")
        elif rain_m and not rain_a and not rain_e:
            parts.append("Kiša ujutru, razvedravanje poslijepodne")
        elif rain_m and not rain_a and rain_e:
            parts.append("Kiša ujutru i uveče, suvo poslijepodne")
        elif not rain_m and rain_a and rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append("Sunčano ujtro, kiša od poslijepodneva")
            else:
                parts.append("Kiša od poslijepodneva")
        elif not rain_m and rain_a and not rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append("Sunčano pa kiša poslijepodne")
            else:
                parts.append("Oblačno, kiša poslijepodne")
        elif not rain_m and not rain_a and rain_e:
            parts.append("Suvo do večeri, zatim kiša")
        else:
            parts.append("Povremena kiša")
    elif has_fog_morn:
        if aftn['cloud'] is not None and aftn['cloud'] < 40:
            parts.append("Magla ujtro, sunčano poslijepodne")
        else:
            parts.append("Magla ujtro, oblačno poslijepodne")
    else:
        if morn['n'] == 0 and aftn['n'] > 0:
            sky_labels = {
                'clear': 'Vedro i sunčano', 'mostly_clear': 'Pretežno vedro',
                'partly_cloudy': 'Djelimično oblačno', 'mostly_cloudy': 'Pretežno oblačno',
                'cloudy': 'Oblačno',
            }
            parts.append(sky_labels.get(as_, 'Promjenljivo'))
        elif ms == as_:
            sky_labels = {
                'clear': 'Vedro i sunčano tokom dana',
                'mostly_clear': 'Pretežno vedro i sunčano',
                'partly_cloudy': 'Smjena sunca i oblaka',
                'mostly_cloudy': 'Pretežno oblačno',
                'cloudy': 'Oblačno tokom dana',
            }
            parts.append(sky_labels.get(ms, 'Promjenljivo'))
        elif ms in ('clear', 'mostly_clear') and as_ in ('mostly_cloudy', 'cloudy'):
            parts.append("Sunčano ujtro, naoblačenje poslijepodne")
        elif ms in ('mostly_cloudy', 'cloudy') and as_ in ('clear', 'mostly_clear'):
            parts.append("Oblačno ujtro, razvedravanje poslijepodne")
        elif ms in ('clear', 'mostly_clear') and as_ == 'partly_cloudy':
            parts.append("Sunčano uz poneki oblak poslijepodne")
        elif ms == 'partly_cloudy' and as_ in ('clear', 'mostly_clear'):
            parts.append("Oblaci ujtro, sunčano poslijepodne")
        elif ms == 'partly_cloudy' and as_ in ('mostly_cloudy', 'cloudy'):
            parts.append("Sve više oblaka tokom dana")
        elif ms in ('mostly_cloudy', 'cloudy') and as_ == 'partly_cloudy':
            parts.append("Oblačno ujtro, djelimično razvedravanje")
        else:
            parts.append("Promjenljivo oblačno")

        if es != 'unknown' and len(parts) > 0:
            curr_end = as_ if as_ != 'unknown' else ms
            if curr_end in ('clear', 'mostly_clear') and es in ('mostly_cloudy', 'cloudy'):
                parts[0] += "; naoblačenje uveče"
            elif curr_end in ('mostly_cloudy', 'cloudy') and es in ('clear', 'mostly_clear'):
                parts[0] += "; razvedravanje uveče"

    if max_wind >= 9.0:
        parts.append("jak vjetar")
    elif max_wind >= 6.0:
        parts.append("vjetrovito")
    elif max_wind >= 4.0:
        parts.append("umjeren vjetar")

    if max_gust >= 15:
        parts.append(f"udari do {max_gust:.0f} m/s")

    if temp_max >= 34:
        parts.append("izuzetno vruće")
    elif temp_max >= 30:
        parts.append("vruće")
    elif temp_min <= -5:
        parts.append("jak mraz")
    elif temp_min <= 0:
        parts.append("mraz")

    if len(parts) == 0:
        return "Promjenljivo"
    elif len(parts) == 1:
        return parts[0]
    else:
        return f"{parts[0]}; {'; '.join(parts[1:])}"


def generate_output(corrected, trained, results, fc_raw=None):
    print("\n[6/6] Generisanje izlaza...")
    now_str = pd.Timestamp.now().isoformat()
    now_ts = pd.Timestamp.now().floor('h')
    cutoff_48h = now_ts + pd.Timedelta(hours=48)

    forecast_hours = []
    for _, row in corrected.iterrows():
        if row['datetime'] >= cutoff_48h:
            continue
        wc = int(row.get('weather_code', 0)) if pd.notna(row.get('weather_code', np.nan)) else 0
        wmo = WMO_CODES.get(wc, WMO_CODES[0])

        entry = {
            "datetime": row['datetime'].isoformat(),
            "hour": int(row['datetime'].hour),
            "date": row['datetime'].strftime('%Y-%m-%d'),
            "day_name": row['datetime'].strftime('%A'),
            "weather_code": wc,
            "weather_desc": wmo['desc'],
            "weather_icon": wmo['icon'],
            "weather_emoji": wmo['emoji'],
        }

        for param, info in TARGET_PARAMS.items():
            xgb_col = f'{param}_xgb'
            ens_col = f'{param}_ensemble'
            val = row.get(xgb_col, row.get(ens_col, None))
            if val is not None and pd.notna(val):
                entry[param] = round(float(val), 2)
            ens_val = row.get(ens_col, None)
            if ens_val is not None and pd.notna(ens_val):
                entry[f'{param}_raw'] = round(float(ens_val), 2)

        for extra in ['apparent_temperature', 'wind_direction_10m', 'surface_pressure',
                      'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'rain', 'snowfall']:
            v = row.get(f'{extra}_ens', None)
            if v is not None and pd.notna(v):
                entry[extra] = round(float(v), 2)

        forecast_hours.append(entry)

    fc_df = pd.DataFrame(forecast_hours)
    daily = []
    for date, grp in fc_df.groupby('date'):
        ds = {
            "date": date,
            "day_name": grp.iloc[0]['day_name'],
            "temp_min": round(float(grp['temperature_2m'].min()), 1) if 'temperature_2m' in grp else None,
            "temp_max": round(float(grp['temperature_2m'].max()), 1) if 'temperature_2m' in grp else None,
            "wind_max": round(float(grp['wind_speed_10m'].max()), 1) if 'wind_speed_10m' in grp else None,
            "gust_max": round(float(grp['wind_gusts_10m'].max()), 1) if 'wind_gusts_10m' in grp else None,
            "precip_total": round(float(grp['precipitation'].sum()), 1) if 'precipitation' in grp else 0,
            "humidity_avg": round(float(grp['relative_humidity_2m'].mean()), 0) if 'relative_humidity_2m' in grp else None,
            "pressure_avg": round(float(grp['pressure_msl'].mean()), 0) if 'pressure_msl' in grp else None,
        }
        if 'wind_direction_10m' in grp.columns:
            wd = pd.to_numeric(grp['wind_direction_10m'], errors='coerce').dropna()
            if len(wd) > 0:
                rad = np.radians(wd)
                ds['wind_dir_avg'] = round(float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360), 0)
        cloud_avg = 0
        if 'cloud_cover' in grp:
            daytime_cc = grp[(grp['hour'] >= 7) & (grp['hour'] <= 19)]
            if len(daytime_cc) > 0:
                cloud_avg = float(daytime_cc['cloud_cover'].mean())
                ds['cloud_cover_day'] = round(cloud_avg, 0)

        if 'weather_code' in grp:
            wc_mode = int(grp['weather_code'].mode().iloc[0])
            precip = ds.get('precip_total', 0)
            rain_hours = grp[grp['weather_code'] >= 51]

            if wc_mode <= 1 and precip > 0.1:
                wc_mode = int(rain_hours['weather_code'].mode().iloc[0]) if len(rain_hours) > 0 else 61
            elif wc_mode == 0 and cloud_avg > 60:
                wc_mode = 3
            elif wc_mode == 0 and cloud_avg > 30:
                wc_mode = 2
            elif wc_mode == 1 and cloud_avg > 60:
                wc_mode = 3

            wmo = WMO_CODES.get(wc_mode, WMO_CODES[0])
            ds.update({"weather_code": wc_mode, "weather_desc": wmo['desc'],
                       "weather_icon": wmo['icon'], "weather_emoji": wmo['emoji']})

        ds['day_narrative'] = _daily_narrative(grp)
        daily.append(ds)

    long_range = []
    long_data = corrected[corrected['datetime'] >= cutoff_48h].copy()
    if len(long_data) > 0:
        long_data['_date'] = long_data['datetime'].dt.strftime('%Y-%m-%d')
        long_data['_day_name'] = long_data['datetime'].dt.strftime('%A')
        long_data['_hour'] = long_data['datetime'].dt.hour

        for ldate, lgrp in long_data.groupby('_date'):
            def _get(col_xgb, col_ens):
                c = lgrp.get(col_xgb, lgrp.get(col_ens, pd.Series(dtype=float)))
                return pd.to_numeric(c, errors='coerce') if isinstance(c, pd.Series) else pd.Series(dtype=float)

            temp = _get('temperature_2m_xgb', 'temperature_2m_ensemble')
            wind = _get('wind_speed_10m_xgb', 'wind_speed_10m_ensemble')
            gusts = _get('wind_gusts_10m_xgb', 'wind_gusts_10m_ensemble')
            precip = _get('precipitation_xgb', 'precipitation_ensemble')
            humid = _get('relative_humidity_2m_xgb', 'relative_humidity_2m_ensemble')
            pres = _get('pressure_msl_xgb', 'pressure_msl_ensemble')

            lds = {"date": ldate, "day_name": lgrp.iloc[0]['_day_name']}
            if temp.notna().any():
                lds['temp_min'] = round(float(temp.min()), 1)
                lds['temp_max'] = round(float(temp.max()), 1)
            if wind.notna().any():
                lds['wind_max'] = round(float(wind.max()), 1)
            if gusts.notna().any():
                lds['gust_max'] = round(float(gusts.max()), 1)
            lds['precip_total'] = round(float(precip.sum()), 1) if precip.notna().any() else 0
            if humid.notna().any():
                lds['humidity_avg'] = round(float(humid.mean()), 0)
            if pres.notna().any():
                lds['pressure_avg'] = round(float(pres.mean()), 0)

            wd_s = pd.to_numeric(lgrp.get('wind_direction_10m_ens', pd.Series(dtype=float)), errors='coerce').dropna()
            if len(wd_s) > 0:
                rad = np.radians(wd_s)
                lds['wind_dir_avg'] = round(float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360), 0)

            lr_cloud_avg = 0
            cloud = _get('cloud_cover_xgb', 'cloud_cover_ensemble')
            lr_daytime = (lgrp['_hour'] >= 7) & (lgrp['_hour'] <= 19)
            if cloud.notna().any() and lr_daytime.any():
                dc = cloud[lr_daytime].dropna()
                if len(dc) > 0:
                    lr_cloud_avg = float(dc.mean())
                    lds['cloud_cover_day'] = round(lr_cloud_avg, 0)

            wc_s = pd.to_numeric(lgrp.get('weather_code', pd.Series(dtype=float)), errors='coerce').dropna()
            if len(wc_s) > 0:
                wc_mode = int(wc_s.mode().iloc[0])
                lr_precip = lds.get('precip_total', 0)
                lr_rain_codes = wc_s[wc_s >= 51]

                if wc_mode <= 1 and lr_precip > 0.1:
                    wc_mode = int(lr_rain_codes.mode().iloc[0]) if len(lr_rain_codes) > 0 else 61
                elif wc_mode == 0 and lr_cloud_avg > 60:
                    wc_mode = 3
                elif wc_mode == 0 and lr_cloud_avg > 30:
                    wc_mode = 2
                elif wc_mode == 1 and lr_cloud_avg > 60:
                    wc_mode = 3

                wmo = WMO_CODES.get(wc_mode, WMO_CODES[0])
                lds.update({"weather_code": wc_mode, "weather_desc": wmo['desc'],
                           "weather_icon": wmo['icon'], "weather_emoji": wmo['emoji']})

            lr_norm = pd.DataFrame({
                'hour': lgrp['_hour'].values,
                'cloud_cover': cloud.values,
                'precipitation': precip.values,
                'wind_speed_10m': wind.values,
                'wind_gusts_10m': gusts.values,
                'temperature_2m': temp.values,
                'weather_code': pd.to_numeric(lgrp.get('weather_code', pd.Series(dtype=float)), errors='coerce').values,
            })
            lds['day_narrative'] = _daily_narrative(lr_norm)

            if fc_raw is not None:
                lr_mask = fc_raw['datetime'].isin(lgrp['datetime'])
                lr_raw = fc_raw[lr_mask]
                precip_cols = [f"{m}_precipitation_model" for m in MODELS if f"{m}_precipitation_model" in lr_raw.columns]
                if precip_cols:
                    model_has_rain = [(pd.to_numeric(lr_raw[c], errors='coerce').fillna(0) > 0.1).any() for c in precip_cols]
                    lds['precip_probability'] = round(sum(model_has_rain) / len(model_has_rain) * 100)

            long_range.append(lds)

        print(f"  Long range: {len(long_range)} dana")

    output = {
        "generated": now_str,
        "location": {"name": "Budva, Crna Gora", "lat": LAT, "lon": LON,
                      "station": "ibudva5 (Weather Underground)"},
        "method": "XGBoost Multi-Model Ensemble + Historical Bias + Forecast Revision v3",
        "description": "8 modela, 6 godina podataka (2020-2026), pametna korekcija + Day1/Day2 revizije",
        "models": MODELS,
        "training_metrics": results,
        "daily_summary": daily,
        "hourly_forecast": forecast_hours,
        "long_range": long_range,
    }

    json_path = os.path.join(OUTPUT_DIR, "forecast_48h.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    csv_path = os.path.join(OUTPUT_DIR, "forecast_48h.csv")
    corrected.to_csv(csv_path, index=False)

    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    print("\n" + "=" * 72)
    print("  KORIGOVANA PROGNOZA ZA BUDVU --- Naredna 48 sata")
    print("=" * 72)
    for d in daily:
        em = d.get('weather_emoji', '')
        print(f"\n  {em} {d['day_name']} {d['date']}")
        print(f"     Temp: {d['temp_min']}\u00b0 --- {d['temp_max']}\u00b0C  |  Vlaznost: {d['humidity_avg']}%")
        print(f"     Vjetar: do {d['wind_max']} m/s (udari {d['gust_max']} m/s)")
        print(f"     Padavine: {d['precip_total']} mm  |  Pritisak: {d['pressure_avg']} hPa")
        print(f"     {d.get('day_narrative', d.get('weather_desc', ''))}")

    print("\n  " + "-" * 68)
    print(f"  {'Sat':>12}  {'Temp':>6}  {'Vlaz':>5}  {'Vjet':>5}  {'Prit':>6}  {'Obl':>4}  {'Kisa':>6}  Opis")
    print(f"  {'':>12}  {'':>6}  {'':>5}  {'':>5}  {'':>6}  {'':>4}  {'':>6}")
    for _, r in corrected.iterrows():
        ts = r['datetime'].strftime('%d.%m %H:%M')
        t = r.get('temperature_2m_xgb', r.get('temperature_2m_ensemble', float('nan')))
        h = r.get('relative_humidity_2m_xgb', r.get('relative_humidity_2m_ensemble', float('nan')))
        w = r.get('wind_speed_10m_xgb', r.get('wind_speed_10m_ensemble', float('nan')))
        p = r.get('pressure_msl_xgb', r.get('pressure_msl_ensemble', float('nan')))
        c = r.get('cloud_cover_xgb', r.get('cloud_cover_ensemble', float('nan')))
        pr = r.get('precipitation_xgb', r.get('precipitation_ensemble', float('nan')))
        wc = int(r.get('weather_code', 0)) if pd.notna(r.get('weather_code', np.nan)) else 0
        em = WMO_CODES.get(wc, WMO_CODES[0])['emoji']

        tf = f"{t:5.1f}\u00b0" if pd.notna(t) else "  N/A"
        hf = f"{h:3.0f}%" if pd.notna(h) else " N/A"
        wf = f"{w:4.1f}" if pd.notna(w) else " N/A"
        pf = f"{p:5.0f}" if pd.notna(p) else "  N/A"
        cf = f"{c:3.0f}%" if pd.notna(c) else " N/A"
        rf = f"{pr:5.2f}" if pd.notna(pr) else "  N/A"

        print(f"  {ts:>12}  {tf}  {hf}  {wf}  {pf}  {cf}  {rf}  {em}")

    return json_path, csv_path


if __name__ == "__main__":
    skip_training = '--skip-training' in sys.argv or '--skip_training' in sys.argv

    if skip_training:
        print("\n  MODE: --skip-training (ucitavam sacuvane modele)")
        trained, results, bias_tables = load_trained_models()
    else:
        hist = load_historical_data()

        print("\n[2/6] Feature engineering + tabele biasa...")
        bias_tables = compute_bias_tables(hist)
        hist = apply_bias_features(hist, bias_tables)
        hist = engineer_features(hist)
        print(f"  Dimenzije: {hist.shape[0]} x {hist.shape[1]}")

        bias_path = os.path.join(MODEL_DIR, 'bias_tables.json')
        bt_serializable = {}
        for k, v in bias_tables.items():
            bt_serializable[k] = v.to_dict(orient='records')
        with open(bias_path, 'w') as f:
            json.dump(bt_serializable, f)
        print(f"  Bias tabele: {bias_path}")

        trained, results = train_all_models(hist)

    fc_all = fetch_live_forecasts()
    corrected = apply_correction(fc_all, trained, bias_tables)
    json_path, csv_path = generate_output(corrected, trained, results, fc_raw=fc_all)

    print("\n" + "=" * 72)
    print("  GOTOVO! Fajlovi:", OUTPUT_DIR)
    print("=" * 72)
