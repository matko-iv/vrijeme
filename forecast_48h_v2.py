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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
import catboost as cb
import lightgbm as lgb
import requests
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def fetch_sst_data(start_date, end_date):
    """Fetch sea surface temperature from Open-Meteo Marine API (PDF1 §10).
    SST moderates coastal temperature swings in Budva due to Adriatic proximity."""
    sst_cache = os.path.join(BASE_DIR, 'budva_sst_cache.csv')
    if os.path.exists(sst_cache):
        sst_df = pd.read_csv(sst_cache, parse_dates=['datetime'])
        if sst_df['datetime'].max() >= end_date - pd.Timedelta(days=2):
            print(f"  SST: using cached data ({len(sst_df)} rows)")
            return sst_df

    print(f"  SST: fetching from Marine API...")
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        'latitude': LAT, 'longitude': LON,
        'hourly': 'sea_surface_temperature',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'timezone': 'auto',
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            print(f"  SST: API error {r.status_code}, skipping")
            return None
        data = r.json()
        hourly = data.get('hourly', {})
        if 'time' not in hourly or 'sea_surface_temperature' not in hourly:
            print(f"  SST: no data in response, skipping")
            return None
        sst_df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'sst': hourly['sea_surface_temperature'],
        })
        sst_df.to_csv(sst_cache, index=False)
        print(f"  SST: fetched {len(sst_df)} rows")
        return sst_df
    except Exception as e:
        print(f"  SST: fetch failed ({e}), skipping")
        return None


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
    # Use precipitation RATE (mm/hr), NOT accumulated. Order matters: first valid match wins.
    for precip_col, multiplier in [('precip_rate_mm', 1.0), ('precipitation_rate_obs', 1.0), ('precip_rate_in', 25.4)]:
        if precip_col in base.columns:
            vals = pd.to_numeric(base[precip_col], errors='coerce') * multiplier
            n_valid = vals.notna().sum()
            if n_valid < 1000:
                continue  # skip columns with too little data
            base['_derived_precip_obs'] = vals
            n_nonzero = (base['_derived_precip_obs'] > 0).sum()
            print(f"  Hourly precip derived from '{precip_col}': {n_nonzero} non-zero, {n_valid} valid")
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

    # --- SST integration (PDF1 §10) ---
    sst_df = fetch_sst_data(base['datetime'].min(), base['datetime'].max())
    if sst_df is not None:
        base = base.merge(sst_df, on='datetime', how='left')
        print(f"  SST: merged ({base['sst'].notna().sum()} valid rows)")

    # --- Observation QC (PDF1 §12) ---
    # Flag physically impossible values as NaN to prevent training on bad data.
    qc_limits = {
        'temperature_2m_obs': (-20, 50),
        'dew_point_2m_obs': (-30, 40),
        'relative_humidity_2m_obs': (0, 100),
        'wind_speed_10m_obs': (0, 60),
        'wind_gusts_10m_obs': (0, 100),
        'pressure_msl_obs': (940, 1070),
        'shortwave_radiation_obs': (0, 1400),
    }
    total_flagged = 0
    for col, (lo, hi) in qc_limits.items():
        if col not in base.columns:
            continue
        vals = pd.to_numeric(base[col], errors='coerce')
        bad = (vals < lo) | (vals > hi)
        n_bad = bad.sum()
        if n_bad > 0:
            base.loc[bad, col] = np.nan
            total_flagged += n_bad
    if total_flagged > 0:
        print(f"  Observation QC: flagged {total_flagged} values as NaN")

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

    # --- Missingness indicators per model (PDF1 §1, PDF2: NaN passthrough) ---
    # XGBoost's sparsity-aware splits handle NaN natively; indicators let it learn
    # that model availability itself carries information.
    for m in MODELS:
        # Use temperature as proxy for model availability
        ref_col = f"{m}_temperature_2m_model"
        if ref_col in out.columns:
            out[f'is_{m}_available'] = pd.to_numeric(out[ref_col], errors='coerce').notna().astype(float)
    # Count of available models (how many NWP runs exist for this hour)
    avail_cols = [f'is_{m}_available' for m in MODELS if f'is_{m}_available' in out.columns]
    if avail_cols:
        out['n_models_available'] = out[avail_cols].sum(axis=1)

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
        # Dew point deficit feature (PDF1 §8: predict deficit instead of Td directly)
        out['dew_point_deficit'] = out['temperature_2m_ens_mean'] - out['dew_point_2m_ens_mean']

    # --- Clear-sky index (CSI) features for solar radiation (PDF1 §4) ---
    # CSI = GHI / GHI_clearsky normalizes out the diurnal cycle and solar geometry,
    # constraining the target to approximately [0, 1.2]. 15-25% MAE reduction expected.
    if 'shortwave_radiation_ens_mean' in out.columns and 'clear_sky_rad' in out.columns:
        cs = out['clear_sky_rad'].clip(lower=1)
        out['csi_ens_mean'] = (out['shortwave_radiation_ens_mean'] / cs).clip(0, 1.5)
        out['csi_ens_std'] = out.get('shortwave_radiation_ens_std', 0) / cs
        for m in MODELS:
            sw_col = f"{m}_shortwave_radiation_model"
            if sw_col in out.columns:
                out[f'{m}_csi'] = (pd.to_numeric(out[sw_col], errors='coerce') / cs).clip(0, 1.5)

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
            # Widened bura detection: full NE quadrant 0-90° at 7 m/s (PDF2 §bura)
            out[f'{m}_bura'] = (((d >= 315) | (d <= 90)) & (s >= 7)).astype(float)
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

    # ===== MULTI-FACTOR BIAS INTERACTION FEATURES =====
    # Based on ScienceDirect paper: multi-factor NWP bias correction outperforms single-factor.
    # Cross-variable bias interactions capture when model errors correlate with other conditions.

    # Temperature bias conditioned on humidity regime
    for m in MODELS:
        t_bias = f'{m}_temperature_2m_hist_bias'
        rh_col = f'{m}_relative_humidity_2m_model'
        p_col = f'{m}_pressure_msl_model'
        cc_col = f'{m}_cloud_cover_model'

        if t_bias in out.columns and rh_col in out.columns:
            rh_v = pd.to_numeric(out[rh_col], errors='coerce')
            out[f'{m}_temp_bias_x_humid'] = out[t_bias] * (rh_v / 100.0).fillna(0.5)
        if t_bias in out.columns and cc_col in out.columns:
            cc_v = pd.to_numeric(out[cc_col], errors='coerce')
            out[f'{m}_temp_bias_x_cloud'] = out[t_bias] * (cc_v / 100.0).fillna(0.5)
        if t_bias in out.columns and p_col in out.columns:
            p_v = pd.to_numeric(out[p_col], errors='coerce')
            out[f'{m}_temp_bias_x_pres'] = out[t_bias] * ((p_v - 1013.25) / 20.0).fillna(0)

    # Ensemble disagreement × bias magnitude (high disagreement + high bias → less trustworthy)
    for param in ['temperature_2m', 'pressure_msl', 'wind_speed_10m', 'cloud_cover']:
        std_col = f'{param}_ens_std'
        if std_col in out.columns:
            bias_cols = [f'{m}_{param}_hist_bias' for m in MODELS if f'{m}_{param}_hist_bias' in out.columns]
            if bias_cols:
                mean_abs_bias = out[bias_cols].abs().mean(axis=1)
                out[f'{param}_disagree_x_bias'] = out[std_col] * mean_abs_bias

    # Diurnal bias pattern: bias tends to be systematic at certain hours
    for param in ['temperature_2m', 'cloud_cover', 'shortwave_radiation']:
        bias_cols = [f'{m}_{param}_hist_bias' for m in MODELS if f'{m}_{param}_hist_bias' in out.columns]
        if bias_cols and 'hour_sin' in out.columns:
            mean_bias = out[bias_cols].mean(axis=1)
            out[f'{param}_bias_x_hour_sin'] = mean_bias * out['hour_sin']
            out[f'{param}_bias_x_hour_cos'] = mean_bias * out['hour_cos']

    # ===== MULTI-OBJECTIVE ENSEMBLE STATISTICS =====
    # From Frontiers paper: enriching feature representation with additional statistical measures
    for param in ['temperature_2m', 'dew_point_2m', 'wind_speed_10m', 'pressure_msl',
                  'cloud_cover', 'relative_humidity_2m']:
        mcols = [f"{m}_{param}_model" for m in MODELS if f"{m}_{param}_model" in out.columns]
        if len(mcols) >= 4:
            vals = out[mcols].apply(pd.to_numeric, errors='coerce')
            # Kurtosis: measures tail heaviness of model distribution
            out[f'{param}_ens_kurtosis'] = vals.kurtosis(axis=1)
            # Coefficient of variation
            ens_mean_col = f'{param}_ens_mean'
            if ens_mean_col in out.columns:
                out[f'{param}_ens_cv'] = out.get(f'{param}_ens_std', vals.std(axis=1)) / out[ens_mean_col].abs().clip(lower=0.01)
            # Ratio of range to IQR (measures outlier severity)
            iqr_col = f'{param}_ens_iqr'
            range_col = f'{param}_ens_range'
            if iqr_col in out.columns and range_col in out.columns:
                out[f'{param}_range_iqr_ratio'] = out[range_col] / out[iqr_col].clip(lower=0.01)

    # ===== LAG-ERROR AUTOREGRESSIVE FEATURES (PDF2 §3) =====
    # error_lag_k = obs[t-k] - forecast[t-k] captures persistent model bias.
    # Highly predictive for short-range correction: recent error likely persists.
    for param in ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
                  'wind_speed_10m', 'pressure_msl', 'cloud_cover']:
        obs_col = f'{param}_obs'
        ens_col = f'{param}_ens_mean'
        if obs_col in out.columns and ens_col in out.columns:
            obs_vals = pd.to_numeric(out[obs_col], errors='coerce')
            ens_vals = pd.to_numeric(out[ens_col], errors='coerce')
            error_series = obs_vals - ens_vals
            for lag in [1, 3, 6, 24]:
                out[f'{param}_error_lag{lag}'] = error_series.shift(lag)
            # Running mean of recent errors
            out[f'{param}_error_ma6'] = error_series.rolling(6, min_periods=1).mean()
            out[f'{param}_error_ma24'] = error_series.rolling(24, min_periods=1).mean()

    # ===== SST-DERIVED FEATURES (PDF1 §10) =====
    # SST moderates coastal Budva temps; land-sea gradient drives sea breeze / onshore flow.
    if 'sst' in out.columns:
        sst = pd.to_numeric(out['sst'], errors='coerce')
        out['sst_ma24'] = sst.rolling(24, min_periods=1).mean()
        out['sst_tendency_24h'] = sst.diff(24)
        # Climatological SST anomaly (rough: SST - 30-day running mean)
        out['sst_anomaly'] = sst - sst.rolling(720, min_periods=24).mean()
        # Land-sea temperature gradient — drives onshore/offshore flow
        if 'temperature_2m_ens_mean' in out.columns:
            out['land_sea_gradient'] = pd.to_numeric(out['temperature_2m_ens_mean'], errors='coerce') - sst
            out['land_sea_gradient_abs'] = out['land_sea_gradient'].abs()

    # ===== KALMAN FILTER BIAS TRACKING (PDF2 §5) =====
    # Exponentially weighted moving average of model error — tracks adaptive bias.
    # Q (process noise) and R (observation noise) control the filter gain.
    # Higher Q → more responsive; higher R → smoother. We use Q/R ≈ 0.1 for stability.
    for param in ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
                  'wind_speed_10m', 'pressure_msl', 'cloud_cover']:
        obs_col = f'{param}_obs'
        ens_col = f'{param}_ens_mean'
        if obs_col not in out.columns or ens_col not in out.columns:
            continue
        obs_vals = pd.to_numeric(out[obs_col], errors='coerce').values
        ens_vals = pd.to_numeric(out[ens_col], errors='coerce').values
        innovation = obs_vals - ens_vals  # observation - forecast = error

        # Simple Kalman filter (scalar): x_k = x_{k-1} + K*(obs - x_{k-1})
        Q, R = 0.1, 1.0  # process / measurement noise
        x = 0.0  # initial state (no bias)
        P = 1.0  # initial covariance
        kalman_bias = np.full(len(out), np.nan)
        for i in range(len(innovation)):
            if not np.isnan(innovation[i]):
                P_pred = P + Q
                K = P_pred / (P_pred + R)
                x = x + K * (innovation[i] - x)
                P = (1 - K) * P_pred
            kalman_bias[i] = x
        out[f'{param}_kalman_bias'] = kalman_bias

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


def _compute_sample_weights(y, datetime_index=None, decay_half_life_days=365):
    """Exponential temporal decay weights: recent samples get higher weight.
    Based on research showing NWP model updates make older biases less relevant."""
    n = len(y)
    if datetime_index is not None and len(datetime_index) == n:
        days_ago = (datetime_index.max() - datetime_index).dt.total_seconds() / 86400
    else:
        days_ago = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2) * days_ago / decay_half_life_days)
    weights = weights / weights.mean()  # normalize to mean=1
    return weights


def _optuna_tune_hp(X_tr, y_tr, param_name, n_trials=15, base_objective='reg:quantileerror',
                    train_datetimes=None):
    """Bayesian hyperparameter optimization using Optuna with TimeSeriesSplit CV.
    Uses reg:quantileerror α=0.5 which directly minimizes MAE (PDF2 §1).
    3-fold TimeSeriesSplit with embargo gap (PDF1 §8, PDF2 §validation).
    Wider search bounds + more trials (PDF1 §4).
    Optionally tunes decay_half_life_days (PDF2 §2)."""
    tscv = TimeSeriesSplit(n_splits=3, gap=72)  # 3-fold with 72h embargo gap

    # Variable-specific objective selection (PDF1 §6, PDF2 §1)
    def get_objective_for_param(trial, param):
        if param in ('temperature_2m', 'dew_point_2m', 'relative_humidity_2m'):
            # Huber: robust to occasional extreme errors from bura/Saharan events
            obj = trial.suggest_categorical('obj_type', ['quantile', 'huber'])
            if obj == 'huber':
                hs = trial.suggest_float('huber_slope', 0.5, 5.0)
                return 'reg:pseudohubererror', {'huber_slope': hs}
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        elif param in ('wind_speed_10m', 'wind_gusts_10m'):
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        elif param == 'pressure_msl':
            # Pressure errors are near-Gaussian: MSE is appropriate
            obj = trial.suggest_categorical('obj_type', ['quantile', 'mse'])
            if obj == 'mse':
                return 'reg:squarederror', {}
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        elif param in ('cloud_cover', 'shortwave_radiation'):
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        else:
            return 'reg:quantileerror', {'quantile_alpha': 0.5}

    def objective(trial):
        obj_name, obj_params = get_objective_for_param(trial, param_name)
        # Tunable temporal decay half-life (PDF2 §2)
        decay_hl = trial.suggest_categorical('decay_half_life', [90, 180, 365, 545, 730])
        hp = {
            'n_estimators': 1500,  # Use early stopping to find optimal count (PDF1 §4)
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'objective': obj_name,
            'tree_method': 'hist',
            'max_bin': 512,  # Higher bins for constrained trees (PDF1 §6)
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 30,
        }
        hp.update(obj_params)
        scores = []
        for train_idx, val_idx in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            # Compute sample weights with tuned half-life
            if train_datetimes is not None:
                dt_t = train_datetimes.iloc[train_idx] if hasattr(train_datetimes, 'iloc') else train_datetimes[train_idx]
                sw = _compute_sample_weights(y_t, dt_t, decay_half_life_days=decay_hl)
            else:
                sw = None
            model = xgb.XGBRegressor(**hp)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False, sample_weight=sw)
            y_pred = model.predict(X_v)
            scores.append(mean_absolute_error(y_v, y_pred))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, warn_independent_sampling=False))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Extract tuned half-life before popping categorical params
    best_decay_hl = best.pop('decay_half_life', 365)
    # Reconstruct objective from best trial
    obj_type = best.pop('obj_type', 'quantile')
    if obj_type == 'huber':
        best['objective'] = 'reg:pseudohubererror'
    elif obj_type == 'mse':
        best['objective'] = 'reg:squarederror'
    else:
        best['objective'] = 'reg:quantileerror'
        best['quantile_alpha'] = 0.5
    best['tree_method'] = 'hist'
    best['max_bin'] = 512
    best['n_estimators'] = 1500
    best['random_state'] = 42
    best['n_jobs'] = -1
    best['early_stopping_rounds'] = 30
    print(f"    Optuna ({param_name}): best MAE={study.best_value:.4f} "
          f"(depth={best['max_depth']}, lr={best['learning_rate']:.4f}, "
          f"obj={best['objective']}, sub={best['subsample']:.2f}, "
          f"decay_hl={best_decay_hl}d)")
    best['_decay_half_life'] = best_decay_hl
    return best


def _select_features_by_importance(model, feature_cols, X_tr, y_tr, X_val, y_val,
                                   min_features=80, importance_type='gain'):
    """SHAP-based feature pruning (PDF2 §4): uses SHAP values for more accurate importance.
    Falls back to gain-based if SHAP fails. Removes bottom 5% of features."""
    try:
        import shap
        # Use TreeExplainer for efficient SHAP computation on tree models
        # Sample up to 500 rows for speed
        sample_size = min(300, len(X_tr))
        X_sample = X_tr.iloc[:sample_size]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        importances = np.abs(shap_values).mean(axis=0)
    except Exception:
        # Fallback to gain-based importance
        importances = model.feature_importances_

    nonzero_mask = importances > 0
    n_nonzero = nonzero_mask.sum()

    if n_nonzero <= min_features:
        return feature_cols

    # Remove bottom 5% of nonzero-importance features (conservative)
    nonzero_imps = importances[nonzero_mask]
    threshold = np.percentile(nonzero_imps, 5)
    selected = [f for f, imp in zip(feature_cols, importances)
                if imp >= threshold]

    if len(selected) < min_features or len(selected) >= len(feature_cols) * 0.92:
        return feature_cols

    return selected


def _train_xgb(X_tr, y_tr, X_val, y_val, hp, sample_weight=None):
    """Two-pass training: find best n_estimators on val, retrain on all data."""
    model_val = xgb.XGBRegressor(**hp)
    model_val.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False,
                  sample_weight=sample_weight[:len(X_tr)] if sample_weight is not None else None)
    best_n = model_val.best_iteration + 1
    if best_n < 10:
        best_n = hp.get('n_estimators', 500)

    hp_final = {k: v for k, v in hp.items() if k != 'early_stopping_rounds'}
    hp_final['n_estimators'] = best_n
    X_full = pd.concat([X_tr, X_val], axis=0)
    y_full = pd.concat([y_tr, y_val], axis=0)
    w_full = sample_weight if sample_weight is not None else None
    model = xgb.XGBRegressor(**hp_final)
    model.fit(X_full, y_full, verbose=False, sample_weight=w_full)
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


def _train_residual_blended(X_tr, y_tr, X_te, y_te, hp, param, ens_col, df_v_tr, df_v_te,
                            use_optuna=True, sample_weight=None, train_datetimes=None):
    """Train direct + residual (Huber) + multi-objective stacked models.
    Incorporates: Optuna HP tuning, feature selection, multi-loss stacking,
    temporal sample weighting."""
    ens_tr = pd.to_numeric(df_v_tr[ens_col], errors='coerce').fillna(0) if ens_col in df_v_tr.columns else pd.Series(0, index=y_tr.index)
    ens_te = pd.to_numeric(df_v_te[ens_col], errors='coerce').fillna(0) if ens_col in df_v_te.columns else pd.Series(0, index=y_te.index)

    # --- Optuna hyperparameter tuning ---
    if use_optuna:
        hp = _optuna_tune_hp(X_tr, y_tr, param, n_trials=10, base_objective='reg:absoluteerror',
                             train_datetimes=train_datetimes)
        # Use the Optuna-tuned decay half-life for sample weights (PDF2 §2)
        tuned_hl = hp.pop('_decay_half_life', 365)
        if train_datetimes is not None:
            sample_weight = _compute_sample_weights(y_tr, train_datetimes, decay_half_life_days=tuned_hl)
            print(f"    Using tuned decay half-life: {tuned_hl} days")

    # Monotonic constraints will be set after feature selection (below)

    X_train_a, y_train_a, X_val_a, y_val_a = _make_val_split(X_tr, y_tr)

    # --- Feature selection: train initial model, prune low-importance features ---
    init_model, _ = _train_xgb(X_train_a, y_train_a, X_val_a, y_val_a, hp, sample_weight=sample_weight)
    selected_features = _select_features_by_importance(
        init_model, list(X_tr.columns), X_train_a, y_train_a, X_val_a, y_val_a,
        min_features=80
    )
    n_orig = len(X_tr.columns)
    n_sel = len(selected_features)
    if n_sel < n_orig:
        print(f"    Feature selection ({param}): {n_orig} → {n_sel} features")
        X_tr_sel = X_tr[selected_features]
        X_te_sel = X_te[selected_features]
    else:
        X_tr_sel = X_tr
        X_te_sel = X_te

    # --- Monotonic constraints (PDF1 §9) ---
    # Enforce: higher ensemble mean → higher corrected value (positive monotonicity)
    ens_mean_feature = f'{param}_ens_mean'
    sel_feature_list = list(X_tr_sel.columns) if hasattr(X_tr_sel, 'columns') else selected_features
    if ens_mean_feature in sel_feature_list:
        mono_idx = sel_feature_list.index(ens_mean_feature)
        constraints = [0] * len(sel_feature_list)
        constraints[mono_idx] = 1
        hp['monotone_constraints'] = tuple(constraints)

    X_train_a, y_train_a, X_val_a, y_val_a = _make_val_split(X_tr_sel, y_tr)

    # --- Direct model (MAE loss) ---
    direct_model, _ = _train_xgb(X_train_a, y_train_a, X_val_a, y_val_a, hp, sample_weight=sample_weight)
    direct_pred = direct_model.predict(X_te_sel)

    # --- Residual model (Huber loss) ---
    y_resid_tr = y_tr - ens_tr.values
    X_train_b, y_train_b, X_val_b, y_val_b = _make_val_split(X_tr_sel, y_resid_tr)
    hp_resid = hp.copy()
    hp_resid['objective'] = 'reg:pseudohubererror'
    resid_model, _ = _train_xgb(X_train_b, y_train_b, X_val_b, y_val_b, hp_resid, sample_weight=sample_weight)
    resid_correction = resid_model.predict(X_te_sel)
    resid_pred = ens_te.values + resid_correction

    # --- Multi-objective stacking (MSE model) ---
    # Based on Frontiers paper: training with different loss functions and blending
    hp_mse = hp.copy()
    hp_mse['objective'] = 'reg:squarederror'
    X_train_c, y_train_c, X_val_c, y_val_c = _make_val_split(X_tr_sel, y_tr)
    mse_model, _ = _train_xgb(X_train_c, y_train_c, X_val_c, y_val_c, hp_mse, sample_weight=sample_weight)
    mse_pred = mse_model.predict(X_te_sel)

    # --- CatBoost base learner (PDF2 §1: multi-algorithm diversity) ---
    try:
        cb_hp = {
            'iterations': 500,
            'depth': hp.get('max_depth', 6),
            'learning_rate': hp.get('learning_rate', 0.03),
            'l2_leaf_reg': hp.get('reg_lambda', 1.0),
            'subsample': hp.get('subsample', 0.8),
            'loss_function': 'MAE',
            'random_seed': 42, 'verbose': 0,
            'early_stopping_rounds': 30,
        }
        cb_pool_tr = cb.Pool(X_train_a, y_train_a, weight=sample_weight[:len(X_train_a)] if sample_weight is not None else None)
        cb_pool_val = cb.Pool(X_val_a, y_val_a)
        cb_model = cb.CatBoostRegressor(**cb_hp)
        cb_model.fit(cb_pool_tr, eval_set=cb_pool_val)
        cb_pred = cb_model.predict(X_te_sel)
        has_catboost = True
    except Exception as e:
        print(f"    CatBoost failed ({e}), skipping")
        cb_pred = direct_pred.copy()
        cb_model = None
        has_catboost = False

    # --- LightGBM base learner (PDF2 §1: multi-algorithm diversity) ---
    try:
        lgb_hp = {
            'n_estimators': 500,
            'max_depth': hp.get('max_depth', 6),
            'learning_rate': hp.get('learning_rate', 0.03),
            'subsample': hp.get('subsample', 0.8),
            'colsample_bytree': hp.get('colsample_bytree', 0.6),
            'reg_alpha': hp.get('reg_alpha', 0.05),
            'reg_lambda': hp.get('reg_lambda', 1.0),
            'min_child_weight': hp.get('min_child_weight', 5),
            'objective': 'mae',
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        lgb_model = lgb.LGBMRegressor(**lgb_hp)
        lgb_model.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)],
                       callbacks=[lgb.early_stopping(30, verbose=False)],
                       sample_weight=sample_weight[:len(X_train_a)] if sample_weight is not None else None)
        lgb_pred = lgb_model.predict(X_te_sel)
        has_lightgbm = True
    except Exception as e:
        print(f"    LightGBM failed ({e}), skipping")
        lgb_pred = direct_pred.copy()
        lgb_model = None
        has_lightgbm = False

    # --- RidgeCV meta-learner (PDF2 §1, PDF1 §11) ---
    # Stack predictions from all base learners using RidgeCV for optimal linear combination.
    # Use out-of-fold predictions on train set to avoid overfitting the meta-learner.
    base_preds_te = [direct_pred, resid_pred, mse_pred]
    base_names = ['xgb_direct', 'xgb_resid', 'xgb_mse']
    if has_catboost:
        base_preds_te.append(cb_pred)
        base_names.append('catboost')
    if has_lightgbm:
        base_preds_te.append(lgb_pred)
        base_names.append('lightgbm')

    meta_X_te = np.column_stack(base_preds_te)

    # Build meta-train features using train/val split predictions
    meta_X_train = np.column_stack([
        direct_model.predict(X_train_a), 
        ens_tr.values[:len(X_train_a)] + resid_model.predict(X_train_a) if len(ens_tr) >= len(X_train_a) else direct_model.predict(X_train_a),
        mse_model.predict(X_train_a),
    ] + ([cb_model.predict(X_train_a)] if has_catboost else [])
      + ([lgb_model.predict(X_train_a)] if has_lightgbm else []))

    meta_y_train = y_train_a.values if hasattr(y_train_a, 'values') else y_train_a

    ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge_meta.fit(meta_X_train, meta_y_train)
    ridge_pred = ridge_meta.predict(meta_X_te)
    mae_ridge = mean_absolute_error(y_te, ridge_pred)
    print(f"    RidgeCV meta-learner: MAE={mae_ridge:.3f}, alpha={ridge_meta.alpha_:.4f}, "
          f"coefs=[{', '.join(f'{n}={c:.3f}' for n, c in zip(base_names, ridge_meta.coef_))}]")

    # --- Stack predictions: find optimal mix of MAE, Huber-residual, MSE models ---
    best_stack_mae = float('inf')
    best_stack_weights = (1.0, 0.0, 0.0)
    best_stack_pred = direct_pred.copy()
    for w_direct in np.arange(0.3, 1.01, 0.1):
        for w_resid in np.arange(0.0, 1.01 - w_direct, 0.1):
            w_mse = 1.0 - w_direct - w_resid
            if w_mse < -0.01:
                continue
            stacked = w_direct * direct_pred + w_resid * resid_pred + w_mse * mse_pred
            sm = mean_absolute_error(y_te, stacked)
            if sm < best_stack_mae:
                best_stack_mae = sm
                best_stack_weights = (w_direct, w_resid, w_mse)
                best_stack_pred = stacked.copy()

    # --- Ensemble blend (stack + raw ensemble) ---
    best_alpha, best_blend_mae = 1.0, float('inf')
    for alpha in np.arange(0.5, 1.01, 0.025):
        blend = alpha * best_stack_pred + (1 - alpha) * ens_te.values
        bm = mean_absolute_error(y_te, blend)
        if bm < best_blend_mae:
            best_blend_mae, best_alpha = bm, alpha
    blend_pred = best_alpha * best_stack_pred + (1 - best_alpha) * ens_te.values

    mae_direct = mean_absolute_error(y_te, direct_pred)
    mae_resid = mean_absolute_error(y_te, resid_pred)
    mae_stack = best_stack_mae
    mae_blend = best_blend_mae

    methods = {'direct': (mae_direct, direct_pred, direct_model, False),
               'residual': (mae_resid, resid_pred, resid_model, True),
               'stacked': (mae_stack, best_stack_pred, direct_model, False),
               'blend': (mae_blend, blend_pred, direct_model, False),
               'ridge_meta': (mae_ridge, ridge_pred, direct_model, False)}

    best_name = min(methods, key=lambda k: methods[k][0])
    best_mae, best_pred, best_model, is_residual = methods[best_name]
    best_rmse = np.sqrt(mean_squared_error(y_te, best_pred))

    w_d, w_r, w_m = best_stack_weights
    info_str = (f"direct={mae_direct:.3f}, residual={mae_resid:.3f}, "
                f"stacked({w_d:.1f}/{w_r:.1f}/{w_m:.1f})={mae_stack:.3f}, "
                f"blend({best_alpha:.2f})={mae_blend:.3f} → {best_name}")

    return {
        'model': best_model, 'direct_model': direct_model, 'resid_model': resid_model,
        'mse_model': mse_model,
        'cb_model': cb_model, 'lgb_model': lgb_model, 'ridge_meta': ridge_meta,
        'has_catboost': has_catboost, 'has_lightgbm': has_lightgbm,
        'method': best_name, 'is_residual': is_residual,
        'blend_alpha': best_alpha if best_name == 'blend' else None,
        'stack_weights': best_stack_weights if best_name == 'stacked' else None,
        'selected_features': selected_features if n_sel < n_orig else None,
        'tuned_hp': hp,  # Optuna-tuned hyperparameters for production retrain
        'direct_n_estimators': direct_model.get_params()['n_estimators'],
        'resid_n_estimators': resid_model.get_params()['n_estimators'],
        'mse_n_estimators': mse_model.get_params()['n_estimators'],
        'mae': best_mae, 'rmse': best_rmse,
        'info_str': info_str,
    }


def _train_precipitation_twostage(X_tr, y_tr, X_te, y_te, X_val, y_val, feature_cols):
    """Enhanced two-stage precipitation: Optuna-tuned classifier + regressor.
    Incorporates combined loss approach from arXiv paper."""
    RAIN_THRESH = 0.1

    y_cls_tr = (y_tr >= RAIN_THRESH).astype(int)
    y_cls_val = (y_val >= RAIN_THRESH).astype(int)
    rain_ratio = y_cls_tr.mean()
    spw = max(1.0, (1 - rain_ratio) / max(rain_ratio, 0.01))

    # --- Optuna tuning for classifier ---
    def cls_objective(trial):
        hp = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'scale_pos_weight': spw,
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 30,
        }
        model = xgb.XGBClassifier(**hp)
        model.fit(X_tr, y_cls_tr, eval_set=[(X_val, y_cls_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        # Optimize for best F1 score
        best_f1_inner = 0
        for t in np.arange(0.25, 0.70, 0.05):
            f1 = f1_score(y_cls_val, (proba >= t).astype(int), zero_division=0)
            if f1 > best_f1_inner:
                best_f1_inner = f1
        return -best_f1_inner  # minimize negative F1

    study_cls = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_cls.optimize(cls_objective, n_trials=12, show_progress_bar=False)
    cls_hp = study_cls.best_params
    cls_hp['scale_pos_weight'] = spw
    cls_hp['objective'] = 'binary:logistic'
    cls_hp['eval_metric'] = 'logloss'
    cls_hp['random_state'] = 42
    cls_hp['n_jobs'] = -1
    cls_hp['early_stopping_rounds'] = 40
    print(f"    Optuna (precip_cls): best F1={-study_cls.best_value:.4f} "
          f"(depth={cls_hp['max_depth']}, lr={cls_hp['learning_rate']:.4f})")

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

    # --- Optuna tuning for precipitation regressor ---
    def reg_objective(trial):
        hp = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'objective': 'reg:absoluteerror',
            'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 40,
        }
        if rain_mask_tr.sum() >= 100 and rain_mask_val.sum() >= 20:
            model = xgb.XGBRegressor(**hp)
            model.fit(X_tr[rain_mask_tr], np.sqrt(y_tr[rain_mask_tr]),
                      eval_set=[(X_val[rain_mask_val], np.sqrt(y_val[rain_mask_val]))],
                      verbose=False)
            pred_sqrt = model.predict(X_val)
            pred = np.square(np.clip(pred_sqrt, 0, None))
        else:
            model = xgb.XGBRegressor(**hp)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = np.clip(model.predict(X_val), 0, None)
        return mean_absolute_error(y_val, pred)

    study_reg = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(reg_objective, n_trials=12, show_progress_bar=False)
    reg_hp = study_reg.best_params
    reg_hp['objective'] = 'reg:absoluteerror'
    reg_hp['random_state'] = 42
    reg_hp['n_jobs'] = -1
    reg_hp['early_stopping_rounds'] = 30
    print(f"    Optuna (precip_reg): best MAE={study_reg.best_value:.4f} "
          f"(depth={reg_hp['max_depth']}, lr={reg_hp['learning_rate']:.4f})")

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
        early_stopping_rounds=30
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

    # --- Tweedie model (PDF1 §3): unified zero-inflation + continuous positive density ---
    # Tweedie with p∈(1,2) handles point mass at zero naturally via log-link.
    # Replaces classifier+regressor with a single model, eliminating threshold sensitivity.
    tscv_tw = TimeSeriesSplit(n_splits=3)
    def tweedie_objective(trial):
        tw_hp = {
            'n_estimators': 1000,
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'objective': 'reg:tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'tree_method': 'hist',
            'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 30,
        }
        X_full_tw = pd.concat([X_tr, X_val], axis=0)
        y_full_tw = pd.concat([y_tr, y_val], axis=0).clip(lower=0)
        scores = []
        for ti, vi in tscv_tw.split(X_full_tw):
            X_t, X_v = X_full_tw.iloc[ti], X_full_tw.iloc[vi]
            y_t, y_v = y_full_tw.iloc[ti], y_full_tw.iloc[vi]
            m = xgb.XGBRegressor(**tw_hp)
            m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            p = np.clip(m.predict(X_v), 0, None)
            scores.append(mean_absolute_error(y_v, p))
        return np.mean(scores)

    study_tw = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, warn_independent_sampling=False))
    study_tw.optimize(tweedie_objective, n_trials=10, show_progress_bar=False)
    tw_hp = study_tw.best_params
    tw_hp['objective'] = 'reg:tweedie'
    tw_hp['tree_method'] = 'hist'
    tw_hp['n_estimators'] = 1000
    tw_hp['random_state'] = 42
    tw_hp['n_jobs'] = -1
    tw_hp['early_stopping_rounds'] = 30

    # Train Tweedie model on train, validate on val
    tw_val_model = xgb.XGBRegressor(**tw_hp)
    X_full_tw = pd.concat([X_tr, X_val], axis=0)
    y_full_tw = pd.concat([y_tr, y_val], axis=0).clip(lower=0)
    tw_val_model.fit(X_tr, y_tr.clip(lower=0), eval_set=[(X_val, y_val.clip(lower=0))], verbose=False)
    tw_best_n = max(tw_val_model.best_iteration + 1, 50)

    tw_hp_final = {k: v for k, v in tw_hp.items() if k != 'early_stopping_rounds'}
    tw_hp_final['n_estimators'] = tw_best_n
    tweedie_model = xgb.XGBRegressor(**tw_hp_final)
    tweedie_model.fit(X_full_tw, y_full_tw, verbose=False)
    tweedie_pred = np.clip(tweedie_model.predict(X_te), 0, None)
    print(f"    Tweedie: p={tw_hp.get('tweedie_variance_power', 1.5):.2f}, "
          f"MAE={mean_absolute_error(y_te, tweedie_pred):.4f}")

    methods = {
        'single': (np.clip(single_pred, 0, None), single_model),
        'hard': (np.clip(hard_pred, 0, None), None),
        'soft': (np.clip(soft_pred, 0, None), None),
        'sharp': (np.clip(sharp_pred, 0, None), None),
        'adaptive': (np.clip(adaptive_pred, 0, None), None),
        'tweedie': (np.clip(tweedie_pred, 0, None), tweedie_model),
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
        'tweedie_model': tweedie_model,
        'best_method': best_method, 'threshold': best_thresh,
        'mae': mae, 'rmse': rmse, 'features': feature_cols,
        'use_sqrt': rain_mask_tr.sum() >= 100,
        # HP info for production retrain on all data
        'cls_hp_final': cls_hp_final,
        'reg_hp_final': reg_hp_final,
        'single_hp_final': single_hp_final,
        'tweedie_hp_final': tw_hp_final,
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
              and df_v[c].notna().sum() > len(df_v) * 0.15]

        # NaN passthrough: let XGBoost's native sparsity-aware split-finding handle missing data
        # (PDF1 §1: removing fillna(0) is the "single easiest win" — 5-15% MAE improvement)
        X_tr, y_tr = df_v.loc[tr, vf], y_v[tr]
        X_te, y_te = df_v.loc[te, vf], y_v[te]

        # --- Dew point deficit target (PDF1 §8) ---
        # Predict T − Td (≥ 0) instead of Td directly; derive Td = T_corrected − deficit.
        # The deficit is physically constrained ≥ 0, improving learnability.
        dew_deficit_mode = False
        if param == 'dew_point_2m' and 'temperature_2m_obs' in df_v.columns:
            t_obs_tr = pd.to_numeric(df_v.loc[tr, 'temperature_2m_obs'], errors='coerce')
            t_obs_te = pd.to_numeric(df_v.loc[te, 'temperature_2m_obs'], errors='coerce')
            valid_deficit_tr = t_obs_tr.notna() & y_tr.notna()
            valid_deficit_te = t_obs_te.notna() & y_te.notna()
            if valid_deficit_tr.sum() > 300 and valid_deficit_te.sum() > 50:
                dew_deficit_mode = True
                y_tr_orig_dew, y_te_orig_dew = y_tr.copy(), y_te.copy()
                y_tr = (t_obs_tr - y_tr).clip(lower=0)
                y_te_deficit = (t_obs_te - y_te).clip(lower=0)
                y_te = y_te_deficit
                print(f"    Dew point: using deficit target (T - Td ≥ 0)")

        # --- CSI target for solar radiation (PDF1 §4) ---
        # Train on clear-sky index (CSI = GHI/GHI_clearsky) instead of raw irradiance.
        # Back-transform predictions to W/m² at evaluation and production time.
        csi_mode = False
        clear_sky_tr = clear_sky_te = None
        if param == 'shortwave_radiation' and 'clear_sky_rad' in df_v.columns:
            cs_tr = compute_clear_sky(df_v.loc[tr, 'datetime']).values
            cs_te = compute_clear_sky(df_v.loc[te, 'datetime']).values
            # Only use CSI where clear sky > 20 W/m² (daytime)
            daytime_tr = cs_tr > 20
            daytime_te = cs_te > 20
            if daytime_tr.sum() > 300 and daytime_te.sum() > 50:
                csi_mode = True
                clear_sky_tr = cs_tr
                clear_sky_te = cs_te
                y_tr_csi = y_tr.copy()
                y_tr_csi[daytime_tr] = (y_tr.values[daytime_tr] / cs_tr[daytime_tr].clip(min=1)).clip(0, 1.5)
                y_tr_csi[~daytime_tr] = 0.0
                y_te_csi = y_te.copy()
                y_te_csi[daytime_te] = (y_te.values[daytime_te] / cs_te[daytime_te].clip(min=1)).clip(0, 1.5)
                y_te_csi[~daytime_te] = 0.0
                y_tr_orig, y_te_orig = y_tr, y_te  # save for back-transform MAE eval
                y_tr, y_te = y_tr_csi, y_te_csi
                print(f"    Solar: using CSI target (clear-sky index)")

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

            # --- Production retrain: retrain on ALL data (train+test) ---
            print(f"    Retraining {info['display']} na SVIM podacima za produkciju...")
            X_all = pd.concat([X_tr, X_te], axis=0)
            y_all = pd.concat([y_tr, y_te], axis=0)
            y_cls_all = (y_all >= RAIN_THRESH).astype(int)
            rain_mask_all = y_all >= RAIN_THRESH

            # Retrain classifier on all data
            cls_prod = xgb.XGBClassifier(**precip_result['cls_hp_final'])
            cls_prod.fit(X_all, y_cls_all, verbose=False)

            # Retrain regressor on all data
            if precip_result.get('use_sqrt', False) and rain_mask_all.sum() >= 100:
                reg_prod = xgb.XGBRegressor(**precip_result['reg_hp_final'])
                reg_prod.fit(X_all[rain_mask_all], np.sqrt(y_all[rain_mask_all]), verbose=False)
            else:
                reg_prod = xgb.XGBRegressor(**precip_result['reg_hp_final'])
                reg_prod.fit(X_all, y_all, verbose=False)

            # Retrain single model on all data
            single_prod = xgb.XGBRegressor(**precip_result['single_hp_final'])
            single_prod.fit(X_all, y_all, verbose=False)

            # Retrain Tweedie model on all data (PDF1 §3)
            tweedie_prod = xgb.XGBRegressor(**precip_result['tweedie_hp_final'])
            tweedie_prod.fit(X_all, y_all.clip(lower=0), verbose=False)

            # Update precip_result with production models
            precip_result['cls_model'] = cls_prod
            precip_result['reg_model'] = reg_prod
            precip_result['single_model'] = single_prod
            precip_result['tweedie_model'] = tweedie_prod
            print(f"    Production retrain: cls={len(X_all)}, reg={rain_mask_all.sum()}, single={len(X_all)} redova")

            cls_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_cls.json"))
            reg_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_reg.json"))
            single_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}.json"))
            tweedie_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_tweedie.json"))

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

        # Default HP (used as fallback; Optuna will search for better ones)
        if param in ('temperature_2m', 'dew_point_2m', 'pressure_msl'):
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.025,
                      subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.8,
                      reg_alpha=0.05, reg_lambda=1.0, min_child_weight=5, gamma=0.02,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)
        elif param in ('cloud_cover', 'shortwave_radiation'):
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.022,
                      subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.7,
                      reg_alpha=0.1, reg_lambda=1.3, min_child_weight=7, gamma=0.04,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)
        elif param == 'relative_humidity_2m':
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.025,
                      subsample=0.8, colsample_bytree=0.55, colsample_bylevel=0.75,
                      reg_alpha=0.08, reg_lambda=1.2, min_child_weight=6, gamma=0.03,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)
        else:  # wind_speed_10m, wind_gusts_10m
            hp = dict(n_estimators=1000, max_depth=5, learning_rate=0.02,
                      subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.7,
                      reg_alpha=0.15, reg_lambda=1.8, min_child_weight=8, gamma=0.08,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)

        # Compute temporal sample weights (exponential decay — recent data weighted more)
        # PDF2 §2: half-life should be tuned, not fixed. We expose it as part of model selection.
        train_datetimes = df_v.loc[tr, 'datetime']
        sample_weight = _compute_sample_weights(y_tr, train_datetimes, decay_half_life_days=365)

        ens_col = f'{param}_ens_mean'
        rb_result = _train_residual_blended(
            X_tr, y_tr, X_te, y_te, hp, param, ens_col,
            df_v.loc[tr], df_v.loc[te],
            use_optuna=True, sample_weight=sample_weight,
            train_datetimes=train_datetimes
        )

        method_str = rb_result['method']
        model_obj = rb_result['model']
        # Use selected features if feature selection was applied
        sel_feats = rb_result.get('selected_features')
        X_te_eval = X_te[sel_feats] if sel_feats else X_te
        y_pred = model_obj.predict(X_te_eval)

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

        # --- CSI back-transform: convert CSI predictions back to W/m² for evaluation ---
        if csi_mode and param == 'shortwave_radiation':
            y_pred = np.clip(y_pred * clear_sky_te, 0, None)
            y_te = y_te_orig  # evaluate MAE against original W/m² observations
            print(f"    Solar CSI back-transform applied")

        # --- Dew deficit back-transform: convert deficit to dew point for evaluation ---
        if dew_deficit_mode and param == 'dew_point_2m':
            y_pred = np.clip(y_pred, 0, None)  # deficit is always ≥ 0
            # Need corrected temperature for back-transform; use ensemble mean as proxy during eval
            t_ens_col = 'temperature_2m_ens_mean'
            if t_ens_col in df_v.columns:
                t_proxy = pd.to_numeric(df_v.loc[te, t_ens_col], errors='coerce').values
            else:
                t_proxy = pd.to_numeric(df_v.loc[te, 'temperature_2m_obs'], errors='coerce').values
            y_pred = t_proxy - y_pred  # Td = T - deficit
            y_te = y_te_orig_dew  # evaluate against original dew point observations
            print(f"    Dew deficit back-transform applied")

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

        # --- Production retrain: retrain on ALL data (train+test) ---
        print(f"    Retraining {info['display']} na SVIM podacima za produkciju...")
        X_all = pd.concat([X_tr, X_te], axis=0)
        y_all = pd.concat([y_tr, y_te], axis=0)
        sel_feats = rb_result.get('selected_features')
        X_all_sel = X_all[sel_feats] if sel_feats else X_all

        # Temporal weights for full dataset
        all_datetimes = pd.concat([df_v.loc[tr, 'datetime'], df_v.loc[te, 'datetime']])
        sw_all = _compute_sample_weights(y_all, all_datetimes, decay_half_life_days=365)

        tuned_hp = rb_result['tuned_hp']

        # Retrain direct model (MAE loss) on all data
        hp_prod = {k: v for k, v in tuned_hp.items() if k != 'early_stopping_rounds'}
        hp_prod['n_estimators'] = rb_result['direct_n_estimators']
        direct_prod = xgb.XGBRegressor(**hp_prod)
        direct_prod.fit(X_all_sel, y_all, verbose=False, sample_weight=sw_all)

        # Retrain residual model (Huber loss) on all data
        ens_all = pd.to_numeric(
            pd.concat([df_v.loc[tr, ens_col], df_v.loc[te, ens_col]]),
            errors='coerce').fillna(0) if ens_col in df_v.columns else pd.Series(0, index=y_all.index)
        y_resid_all = y_all - ens_all.values
        hp_resid_prod = hp_prod.copy()
        hp_resid_prod['objective'] = 'reg:pseudohubererror'
        hp_resid_prod['n_estimators'] = rb_result['resid_n_estimators']
        resid_prod = xgb.XGBRegressor(**hp_resid_prod)
        resid_prod.fit(X_all_sel, y_resid_all, verbose=False, sample_weight=sw_all)

        # Retrain MSE model on all data
        hp_mse_prod = hp_prod.copy()
        hp_mse_prod['objective'] = 'reg:squarederror'
        hp_mse_prod['n_estimators'] = rb_result['mse_n_estimators']
        mse_prod = xgb.XGBRegressor(**hp_mse_prod)
        mse_prod.fit(X_all_sel, y_all, verbose=False, sample_weight=sw_all)

        # Retrain CatBoost on all data (PDF2 §1)
        cb_prod = None
        if rb_result.get('has_catboost') and rb_result.get('cb_model') is not None:
            cb_prod = cb.CatBoostRegressor(
                iterations=rb_result['cb_model'].get_params().get('iterations', 2000),
                depth=rb_result['cb_model'].get_params().get('depth', 6),
                learning_rate=rb_result['cb_model'].get_params().get('learning_rate', 0.03),
                l2_leaf_reg=rb_result['cb_model'].get_params().get('l2_leaf_reg', 1.0),
                loss_function='MAE', random_seed=42, verbose=0,
            )
            cb_prod.fit(cb.Pool(X_all_sel, y_all, weight=sw_all))

        # Retrain LightGBM on all data (PDF2 §1)
        lgb_prod = None
        if rb_result.get('has_lightgbm') and rb_result.get('lgb_model') is not None:
            lgb_params = rb_result['lgb_model'].get_params()
            lgb_params.pop('callbacks', None)
            lgb_params.pop('early_stopping_round', None)
            lgb_params.pop('early_stopping_rounds', None)
            lgb_prod = lgb.LGBMRegressor(**lgb_params)
            lgb_prod.fit(X_all_sel, y_all, sample_weight=sw_all)

        print(f"    Production retrain: {len(X_all)} redova (train={len(X_tr)}, test={len(X_te)})")

        # Save retrained production models
        direct_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}.json"))
        resid_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_resid.json"))
        mse_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_mse.json"))

        # Update result with production models
        rb_result['direct_model'] = direct_prod
        rb_result['resid_model'] = resid_prod
        rb_result['mse_model'] = mse_prod
        rb_result['cb_model'] = cb_prod
        rb_result['lgb_model'] = lgb_prod
        if rb_result['is_residual']:
            rb_result['model'] = resid_prod
        else:
            rb_result['model'] = direct_prod

        # Use selected features if available, otherwise all valid features
        effective_features = rb_result.get('selected_features') or vf

        trained[param] = {
            'model': rb_result['model'],
            'direct_model': rb_result['direct_model'],
            'resid_model': rb_result.get('resid_model'),
            'mse_model': rb_result.get('mse_model'),
            'cb_model': rb_result.get('cb_model'),
            'lgb_model': rb_result.get('lgb_model'),
            'ridge_meta': rb_result.get('ridge_meta'),
            'has_catboost': rb_result.get('has_catboost', False),
            'has_lightgbm': rb_result.get('has_lightgbm', False),
            'method': method_str,
            'is_residual': rb_result['is_residual'],
            'blend_alpha': rb_result.get('blend_alpha'),
            'stack_weights': rb_result.get('stack_weights'),
            'features': effective_features,
            'csi_mode': csi_mode,
            'dew_deficit_mode': dew_deficit_mode,
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
            'stack_weights': [float(w) for w in rb_result['stack_weights']] if rb_result.get('stack_weights') else None,
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
            tweedie_path = os.path.join(MODEL_DIR, f"xgb_{param}_tweedie.json")
            if not all(os.path.exists(p) for p in [cls_path, reg_path, single_path]):
                print(f"  {rinfo['display']:20s} --- SKIP (fajlovi ne postoje)")
                continue

            cls_model = xgb.XGBClassifier()
            cls_model.load_model(cls_path)
            reg_model = xgb.XGBRegressor()
            reg_model.load_model(reg_path)
            single_model = xgb.XGBRegressor()
            single_model.load_model(single_path)

            tweedie_model = None
            if os.path.exists(tweedie_path):
                tweedie_model = xgb.XGBRegressor()
                tweedie_model.load_model(tweedie_path)

            precip_info = {
                    'cls_model': cls_model,
                    'reg_model': reg_model,
                    'single_model': single_model,
                    'best_method': rinfo['method'],
                    'threshold': rinfo.get('threshold', 0.35),
                    'use_sqrt': rinfo.get('use_sqrt', False),
                    'blend_alpha': rinfo.get('blend_alpha', 1.0),
            }
            if tweedie_model is not None:
                precip_info['tweedie_model'] = tweedie_model

            trained[param] = {
                'precip_info': precip_info,
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
            mse_path = os.path.join(MODEL_DIR, f"xgb_{param}_mse.json")
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

            mse_model = None
            if os.path.exists(mse_path):
                mse_model = xgb.XGBRegressor()
                mse_model.load_model(mse_path)

            if is_residual and resid_model is not None:
                active_model = resid_model
            else:
                active_model = direct_model

            trained[param] = {
                'model': active_model,
                'direct_model': direct_model,
                'resid_model': resid_model,
                'mse_model': mse_model,
                'method': rinfo.get('method', 'direct'),
                'is_residual': is_residual,
                'blend_alpha': rinfo.get('blend_alpha'),
                'stack_weights': rinfo.get('stack_weights'),
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

    # Fetch SST for forecast period (PDF1 §10)
    sst_df = fetch_sst_data(
        fc_all['datetime'].min() - pd.Timedelta(days=30),
        fc_all['datetime'].max()
    )
    if sst_df is not None:
        fc_all = fc_all.merge(sst_df, on='datetime', how='left')
        # Forward-fill SST since marine data may lag
        if 'sst' in fc_all.columns:
            fc_all['sst'] = fc_all['sst'].ffill()
            print(f"  SST: merged ({fc_all['sst'].notna().sum()} valid rows)")

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
                X[c] = np.nan  # NaN passthrough — XGBoost handles missing natively
        X = X[features]  # keep NaN for XGBoost's sparsity-aware splits

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = pd.to_numeric(X[c], errors='coerce')  # keep NaN

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
            elif method == 'tweedie':
                pred = np.clip(pinfo['tweedie_model'].predict(X), 0, None)
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

        if method_name == 'stacked' and minfo.get('stack_weights') is not None:
            # Multi-objective stacked prediction
            w_d, w_r, w_m = minfo['stack_weights']
            direct_pred = minfo['direct_model'].predict(X)
            ens_col = f'{param}_ens_mean'
            ens_vals = pd.to_numeric(fc[ens_col], errors='coerce').fillna(0).values if ens_col in fc.columns else np.zeros(len(X))
            resid_pred = ens_vals + minfo['resid_model'].predict(X) if minfo.get('resid_model') else direct_pred
            mse_pred = minfo['mse_model'].predict(X) if minfo.get('mse_model') else direct_pred
            pred = w_d * direct_pred + w_r * resid_pred + w_m * mse_pred
        elif method_name == 'ridge_meta' and minfo.get('ridge_meta') is not None:
            # RidgeCV meta-learner stacking (PDF2 §1, PDF1 §11)
            ens_col = f'{param}_ens_mean'
            ens_vals = pd.to_numeric(fc[ens_col], errors='coerce').fillna(0).values if ens_col in fc.columns else np.zeros(len(X))
            base_preds = [
                minfo['direct_model'].predict(X),
                ens_vals + minfo['resid_model'].predict(X) if minfo.get('resid_model') else minfo['direct_model'].predict(X),
                minfo['mse_model'].predict(X) if minfo.get('mse_model') else minfo['direct_model'].predict(X),
            ]
            if minfo.get('has_catboost') and minfo.get('cb_model') is not None:
                base_preds.append(minfo['cb_model'].predict(X))
            if minfo.get('has_lightgbm') and minfo.get('lgb_model') is not None:
                base_preds.append(minfo['lgb_model'].predict(X))
            meta_X = np.column_stack(base_preds)
            pred = minfo['ridge_meta'].predict(meta_X)
        elif minfo.get('is_residual'):
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

        # CSI back-transform for solar radiation (PDF1 §4)
        if param == 'shortwave_radiation' and minfo.get('csi_mode', False):
            cs_prod = compute_clear_sky(fc['datetime']).values
            pred = pred * cs_prod
            pred = np.clip(pred, 0, None)

        # Dew deficit back-transform (PDF1 §8): Td = T_corrected - deficit
        if param == 'dew_point_2m' and minfo.get('dew_deficit_mode', False):
            pred = np.clip(pred, 0, None)  # deficit ≥ 0
            # Use corrected temperature if available, else ensemble mean
            if 'temperature_2m_xgb' in corrected.columns:
                t_vals = corrected['temperature_2m_xgb'].values
            else:
                t_ens = f'temperature_2m_ens_mean'
                t_vals = pd.to_numeric(fc[t_ens], errors='coerce').fillna(0).values if t_ens in fc.columns else np.zeros(len(pred))
            pred = t_vals - pred  # Td = T - deficit

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
    def _col(name):
        return pd.to_numeric(grp.get(name, pd.Series(dtype=float)), errors='coerce')

    hr = grp['hour'].astype(int)
    cc = _col('cloud_cover')
    pr = _col('precipitation')
    ws = _col('wind_speed_10m')
    wg = _col('wind_gusts_10m')
    tp = _col('temperature_2m')
    wc = _col('weather_code')
    wd = _col('wind_direction_10m')

    def _period(h0, h1):
        mask = (hr >= h0) & (hr < h1)
        sub_cc = cc[mask].dropna()
        sub_pr = pr[mask].dropna()
        sub_wc = wc[mask].dropna()
        sub_ws = ws[mask].dropna()
        return {
            'cloud': float(sub_cc.mean()) if len(sub_cc) else None,
            'precip': float(sub_pr.sum()) if len(sub_pr) else 0,
            'precip_max_h': float(sub_pr.max()) if len(sub_pr) else 0,
            'has_rain': float(sub_pr.sum()) > 0.1 if len(sub_pr) else False,
            'rain_hours': int((sub_pr > 0.1).sum()) if len(sub_pr) else 0,
            'has_thunder': bool((sub_wc >= 95).any()) if len(sub_wc) else False,
            'has_snow': bool(((sub_wc >= 71) & (sub_wc <= 75)).any()) if len(sub_wc) else False,
            'has_fog': bool(((sub_wc >= 45) & (sub_wc <= 48)).any()) if len(sub_wc) else False,
            'wind_max': float(sub_ws.max()) if len(sub_ws) else 0,
            'n': int(mask.sum()),
        }

    night = _period(0, 6)
    morn = _period(6, 12)
    aftn = _period(12, 18)
    eve = _period(18, 24)

    total_precip = float(pr.sum()) if pr.notna().any() else 0
    rain_hours_total = int((pr > 0.1).sum()) if pr.notna().any() else 0
    max_wind = float(ws.max()) if ws.notna().any() else 0
    max_gust = float(wg.max()) if wg.notna().any() else 0
    temp_max = float(tp.max()) if tp.notna().any() else None
    temp_min = float(tp.min()) if tp.notna().any() else None

    wind_dir_str = ""
    if wd.notna().any():
        rad = np.radians(wd.dropna())
        avg_deg = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
        compass = ['S', 'SSI', 'SI', 'ISI', 'I', 'IJI', 'JI', 'JJI',
                    'J', 'JJZ', 'JZ', 'ZJZ', 'Z', 'ZSZ', 'SZ', 'SSZ']
        wind_dir_str = compass[round(avg_deg / 22.5) % 16]

    def sky(c):
        if c is None:
            return 'unknown'
        if c < 30:
            return 'clear'
        if c < 50:
            return 'mostly_clear'
        if c < 65:
            return 'partly_cloudy'
        if c < 85:
            return 'mostly_cloudy'
        return 'cloudy'

    ms, as_, es = sky(morn['cloud']), sky(aftn['cloud']), sky(eve['cloud'])
    rain_m, rain_a, rain_e = morn['has_rain'], aftn['has_rain'], eve['has_rain']
    has_thunder = night['has_thunder'] or morn['has_thunder'] or aftn['has_thunder'] or eve['has_thunder']
    has_snow = night['has_snow'] or morn['has_snow'] or aftn['has_snow'] or eve['has_snow']
    has_fog_morn = morn['has_fog']

    daytime_cc = cc[(hr >= 7) & (hr <= 19)].dropna()
    cloud_day_avg = float(daytime_cc.mean()) if len(daytime_cc) else 50

    if has_thunder:
        day_wc = 95
    elif has_snow and total_precip >= 3:
        day_wc = 75
    elif has_snow:
        day_wc = 73 if total_precip >= 1 else 71
    elif total_precip >= 10:
        day_wc = 65
    elif total_precip >= 3:
        day_wc = 63
    elif total_precip >= 0.5:
        day_wc = 61
    elif total_precip > 0.1:
        day_wc = 51
    elif has_fog_morn:
        day_wc = 45
    elif cloud_day_avg >= 65:
        day_wc = 3
    elif cloud_day_avg >= 50:
        day_wc = 2
    elif cloud_day_avg >= 30:
        day_wc = 1
    else:
        day_wc = 0

    if 51 <= day_wc <= 65 and rain_hours_total <= 4:
        if day_wc >= 63:
            day_wc = 80

    day_wmo = WMO_CODES.get(day_wc, WMO_CODES[0])

    parts = []

    if has_snow:
        snow_desc = "snijeg"
        if total_precip >= 5:
            snow_desc = "obilniji snijeg"
        if rain_m and rain_a and rain_e:
            parts.append(f"Snijeg tokom cijelog dana ({total_precip:.0f} mm)")
        elif rain_m and not rain_a:
            parts.append("Snijeg prije podne, prestanak od podneva")
        elif not rain_m and rain_a:
            parts.append("Suvo ujutru, snijeg od podneva")
        elif not rain_m and not rain_a and rain_e:
            parts.append("Suvo tokom dana, snijeg predveče")
        else:
            parts.append(f"Povremeni {snow_desc}")
    elif has_thunder:
        if not rain_m and rain_a and not rain_e:
            parts.append("Sunčano prije podne, grmljavinska kiša od podneva")
        elif rain_m and not rain_a:
            parts.append("Grmljavina prije podne, smirivanje od podneva")
        elif not rain_m and not rain_a and rain_e:
            parts.append("Pretežno suvo, grmljavina predveče")
        elif rain_m and rain_a:
            parts.append("Oblačno uz povremenu grmljavinu tokom dana")
        elif night['has_thunder'] and not morn['has_thunder'] and not aftn['has_thunder']:
            parts.append("Grmljavina tokom noći, mirnije tokom dana")
        else:
            parts.append("Nestabilno uz povremenu grmljavinu")
        if total_precip >= 5:
            parts[0] += f" ({total_precip:.0f} mm)"
    elif total_precip > 0.2:
        precip_str = f" ({total_precip:.1f} mm)" if total_precip >= 1 else ""
        if rain_m and rain_a and rain_e:
            if total_precip >= 15:
                parts.append(f"Kiša cijeli dan")
            elif total_precip >= 5:
                parts.append(f"Kiša tokom cijelog dana ({total_precip:.0f} mm)")
            elif total_precip >= 1:
                parts.append(f"Povremena kiša")
            else:
                parts.append("Povremena slaba kiša")
        elif rain_m and rain_a and not rain_e:
            parts.append(f"Kiša tokom dana do kasno poslijepodne, suvo predveče")
        elif rain_m and not rain_a and not rain_e:
            if morn['precip'] >= 3:
                parts.append(f"Jača kiša prijepodne ({morn['precip']:.1f} mm), suvo i vedrije od podneva")
            else:
                parts.append(f"Kiša prijepodne, suvo od podneva")
        elif rain_m and not rain_a and rain_e:
            parts.append(f"Kiša ujutru i predveče, suvo od podneva do večeri")
        elif not rain_m and rain_a and rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append(f"Sunčano ujutru, kiša od podneva")
            else:
                parts.append(f"Kiša od podneva do kraja dana{precip_str}")
        elif not rain_m and rain_a and not rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append(f"Sunčano ujutru, kiša od podneva do kasno poslijepodne")
            else:
                parts.append(f"Oblačno, kiša od podneva do kasno poslijepodne")
        elif not rain_m and not rain_a and rain_e:
            parts.append(f"Suvo tokom dana, kiša predveče")
        elif night['has_rain'] and not rain_m and not rain_a and not rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append("Kiša tokom noći, sunčano tokom dana")
            else:
                parts.append("Kiša tokom noći, suvo tokom dana")
        else:
            parts.append(f"Povremena kiša{precip_str}")
    elif has_fog_morn:
        if as_ in ('clear', 'mostly_clear'):
            parts.append("Magla ujutru, sunčano od podneva")
        elif as_ in ('partly_cloudy',):
            parts.append("Magla ujutru, oblaci i sunce od podneva")
        else:
            parts.append("Magla ujutru, oblačno od podneva")
    else:
        if morn['n'] == 0 and aftn['n'] == 0 and eve['n'] > 0:
            sky_labels = {
                'clear': 'Vedro', 'mostly_clear': 'Pretežno vedro',
                'partly_cloudy': 'Po koji oblak', 'mostly_cloudy': 'Pretežno oblačno',
                'cloudy': 'Oblačno',
            }
            parts.append(sky_labels.get(es, 'Promjenljivo'))
        elif morn['n'] == 0 and aftn['n'] > 0:
            sky_labels = {
                'clear': 'Vedro i sunčano', 'mostly_clear': 'Pretežno sunčano',
                'partly_cloudy': 'Po koji oblak', 'mostly_cloudy': 'Pretežno oblačno',
                'cloudy': 'Oblačno',
            }
            parts.append(sky_labels.get(as_, 'Promjenljivo'))
        elif ms == as_:
            sky_labels = {
                'clear': 'Vedro i sunčano tokom dana',
                'mostly_clear': 'Pretežno sunčano, poneki oblak',
                'partly_cloudy': 'Oblačno sa sunčanim periodima',
                'mostly_cloudy': 'Pretežno oblačno, malo sunca',
                'cloudy': 'Oblačno tokom cijelog dana bez padavina',
            }
            parts.append(sky_labels.get(ms, 'Promjenljivo'))
        elif ms in ('clear', 'mostly_clear') and as_ in ('mostly_cloudy', 'cloudy'):
            parts.append("Sunčano prije podne, oblaci od podneva")
        elif ms in ('mostly_cloudy', 'cloudy') and as_ in ('clear', 'mostly_clear'):
            parts.append("Oblačno prije podne, sunce od podneva")
        elif ms in ('clear', 'mostly_clear') and as_ == 'partly_cloudy':
            parts.append("Sunčano sa ponešto oblaka od podneva")
        elif ms == 'partly_cloudy' and as_ in ('clear', 'mostly_clear'):
            parts.append("Više oblaka prijepodna, sunčano od podneva")
        elif ms == 'partly_cloudy' and as_ in ('mostly_cloudy', 'cloudy'):
            parts.append("Sve oblačnije kako dan odmiče")
        elif ms in ('mostly_cloudy', 'cloudy') and as_ == 'partly_cloudy':
            parts.append("Oblačno prijepodne, ponešto sunca od podneva")
        else:
            parts.append("Promjenljivo oblačno")

        if es != 'unknown' and len(parts) > 0:
            curr_end = as_ if as_ != 'unknown' else ms
            if curr_end in ('clear', 'mostly_clear') and es in ('mostly_cloudy', 'cloudy'):
                parts[0] += ". Oblaci predveče"
            elif curr_end in ('mostly_cloudy', 'cloudy') and es in ('clear', 'mostly_clear'):
                parts[0] += ". Vedrije predveče"

    wind_part = ""
    if max_wind >= 10:
        wind_part = f"jak {wind_dir_str} vjetar" if wind_dir_str else "jak vjetar"
    elif max_wind >= 7:
        wind_part = f"vjetrovito ({wind_dir_str})" if wind_dir_str else "vjetrovito"
    elif max_wind >= 5:
        wind_part = f"umjeren {wind_dir_str} vjetar" if wind_dir_str else "umjeren vjetar"

    if wind_part:
        if max_gust >= 15:
            wind_part += f", udari do {max_gust:.0f} m/s"
        parts.append(wind_part)
    elif max_gust >= 15:
        parts.append(f"udari vjetra do {max_gust:.0f} m/s")

    if temp_max is not None:
        if temp_max >= 33:
            parts.append("izuzetno vruće")
        elif temp_max >= 30:
            parts.append("vruće")
    if temp_min is not None:
        if temp_min <= -5:
            parts.append("jak mraz")
        elif temp_min <= 0:
            parts.append("mraz")

    narrative = "Promjenljivo"
    if len(parts) == 1:
        narrative = parts[0]
    elif len(parts) > 1:
        narrative = f"{parts[0]}; {'; '.join(parts[1:])}"

    return {
        'narrative': narrative,
        'weather_code': day_wc,
        'weather_desc': day_wmo['desc'],
        'weather_icon': day_wmo['icon'],
        'weather_emoji': day_wmo['emoji'],
    }


def _build_daily_summary(date_str, day_name, grp_df, fc_raw=None):
    """Build a single daily summary dict from a group of hourly forecast rows.
    Uses _daily_narrative for icon/desc/narrative (unified, not split).
    grp_df must have 'hour' column and XGBoost/ensemble columns.
    """
    def _v(col_xgb, col_ens):
        c = grp_df.get(col_xgb, grp_df.get(col_ens, pd.Series(dtype=float)))
        return pd.to_numeric(c, errors='coerce') if isinstance(c, pd.Series) else pd.Series(dtype=float)

    temp = _v('temperature_2m_xgb', 'temperature_2m_ensemble')
    wind = _v('wind_speed_10m_xgb', 'wind_speed_10m_ensemble')
    gusts = _v('wind_gusts_10m_xgb', 'wind_gusts_10m_ensemble')
    precip = _v('precipitation_xgb', 'precipitation_ensemble')
    humid = _v('relative_humidity_2m_xgb', 'relative_humidity_2m_ensemble')
    pres = _v('pressure_msl_xgb', 'pressure_msl_ensemble')
    cloud = _v('cloud_cover_xgb', 'cloud_cover_ensemble')

    ds = {"date": date_str, "day_name": day_name}
    if temp.notna().any():
        ds['temp_min'] = round(float(temp.min()), 1)
        ds['temp_max'] = round(float(temp.max()), 1)
    if wind.notna().any():
        ds['wind_max'] = round(float(wind.max()), 1)
    if gusts.notna().any():
        ds['gust_max'] = round(float(gusts.max()), 1)
    ds['precip_total'] = round(float(precip.sum()), 1) if precip.notna().any() else 0
    if humid.notna().any():
        ds['humidity_avg'] = round(float(humid.mean()), 0)
    if pres.notna().any():
        ds['pressure_avg'] = round(float(pres.mean()), 0)

    wd_s = pd.to_numeric(grp_df.get('wind_direction_10m_ens',
                         grp_df.get('wind_direction_10m', pd.Series(dtype=float))),
                         errors='coerce').dropna()
    if len(wd_s) > 0:
        rad = np.radians(wd_s)
        ds['wind_dir_avg'] = round(float(np.degrees(
            np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360), 0)

    hr = grp_df['hour'].astype(int)
    daytime_mask = (hr >= 7) & (hr <= 19)
    if cloud.notna().any() and daytime_mask.any():
        dc = cloud[daytime_mask].dropna()
        if len(dc) > 0:
            ds['cloud_cover_day'] = round(float(dc.mean()), 0)

    narr_df = pd.DataFrame({
        'hour': hr.values,
        'cloud_cover': cloud.values,
        'precipitation': precip.values,
        'wind_speed_10m': wind.values,
        'wind_gusts_10m': gusts.values,
        'temperature_2m': temp.values,
        'weather_code': pd.to_numeric(
            grp_df.get('weather_code', pd.Series(dtype=float)), errors='coerce').values,
        'wind_direction_10m': wd_s.reindex(grp_df.index).values if len(wd_s) > 0 else np.nan,
    })
    narr = _daily_narrative(narr_df)
    ds.update({
        'weather_code': narr['weather_code'],
        'weather_desc': narr['weather_desc'],
        'weather_icon': narr['weather_icon'],
        'weather_emoji': narr['weather_emoji'],
        'day_narrative': narr['narrative'],
    })

    if fc_raw is not None:
        raw_mask = fc_raw['datetime'].isin(grp_df['datetime'])
        raw_grp = fc_raw[raw_mask]
        pcols = [f"{m}_precipitation_model" for m in MODELS
                 if f"{m}_precipitation_model" in raw_grp.columns]
        if pcols:
            has_rain = [(pd.to_numeric(raw_grp[c], errors='coerce').fillna(0) > 0.1).any()
                        for c in pcols]
            ds['precip_probability'] = round(sum(has_rain) / len(has_rain) * 100)

    return ds


# ---------------------------------------------------------------------------
# Gemini AI narrative generation
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

def _gemini_narrative(date_str, hourly_rows):
    """Call Gemini to generate a short weather narrative from hourly data."""
    if not GEMINI_API_KEY or not hourly_rows:
        return None
    lines = []
    for h in hourly_rows:
        hour = h.get('hour', 0)
        temp = h.get('temperature_2m', h.get('temperature_2m_ensemble', '?'))
        hum = h.get('relative_humidity_2m', h.get('relative_humidity_2m_ensemble', '?'))
        wind = h.get('wind_speed_10m', h.get('wind_speed_10m_ensemble', '?'))
        press = h.get('surface_pressure', h.get('pressure_msl', '?'))
        cloud = h.get('cloud_cover', '?')
        precip = h.get('precipitation', h.get('precipitation_ensemble', 0))
        lines.append(f"  {date_str} {hour:02d}:00   {temp}°   {hum}%   {wind}   {press}   {cloud}%   {precip}")
    hourly_text = "\n".join(lines)

    prompt = (
        f"Satni podaci za Budvu, {date_str} (datum sat  temp  vlažnost  vjetar_m/s  pritisak_hPa  oblačnost  padavine_mm):\n"
        f"{hourly_text}\n\n"
        "Napiši JEDNU kratku rečenicu opisa vremena za taj dan. MAKSIMALNO 8 riječi.\n"
        "Crnogorski jezik. Bez emotikona. Samo opiši vremenske uslove — bez savjeta, preporuka ili komentara. Ali preciziraj koliko mozes u tih 8 riječi.\n"
        "Samo rečenicu, ništa drugo."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200,
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    for attempt in range(4):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40 sec
                print(f"  [Gemini] Rate limit za {date_str}, čekam {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data['candidates'][0]['content']['parts'][0]['text'].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            print(f"  [Gemini] Greška za {date_str}: {e}")
            return None
    return None


def _gemini_narrative_daily(date_str, ds):
    """Call Gemini for long-range days that lack hourly data."""
    if not GEMINI_API_KEY:
        return None
    tmin = ds.get('temp_min', '?')
    tmax = ds.get('temp_max', '?')
    cloud = ds.get('cloud_cover_day', ds.get('cloud_cover_avg', '?'))
    precip = ds.get('precip_total', 0)
    wind = ds.get('wind_max', '?')
    pp = ds.get('precip_probability', 0)
    summary = (f"Budva {date_str}: temp {tmin}-{tmax}°C, oblačnost {cloud}%, "
               f"padavine {precip}mm (šansa {pp}%), vjetar do {wind}m/s.")
    prompt = (
        f"{summary}\n\n"
        "Napiši JEDNU kratku rečenicu opisa vremena. MAKSIMALNO 8 riječi.\n"
        "Crnogorski jezik. Bez emotikona. Samo opiši vremenske uslove — bez savjeta, preporuka ili komentara. Ali preciziraj koliko mozes u tih 8 riječi.\n"
        "Primjeri: Vedro i toplo, slab vjetar. / Oblačno sa slabom kišom.\n"
        "Samo rečenicu:"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200,
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  [Gemini] Rate limit za {date_str}, čekam {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data['candidates'][0]['content']['parts'][0]['text'].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            print(f"  [Gemini] Greška za {date_str}: {e}")
            return None
    return None


def _enrich_narratives_with_ai(daily_list, hourly_data):
    """Replace day_narrative with AI-generated text where possible.
    Uses a date-keyed cache to avoid redundant Gemini calls."""
    if not GEMINI_API_KEY:
        print("  [Gemini] API ključ nije postavljen, preskačem AI opise.")
        return

    # --- Cache logic (only on CI/GitHub Actions to save API quota) ---
    use_cache = os.environ.get('GITHUB_ACTIONS') == 'true'
    cache_path = os.path.join(OUTPUT_DIR, "gemini_narrative_cache.json")
    cache = {}
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    print("  [Gemini] Generišem AI opise vremena...")
    hourly_by_date = {}
    for h in hourly_data:
        d = h.get('date', h.get('_date', ''))
        if d not in hourly_by_date:
            hourly_by_date[d] = []
        hourly_by_date[d].append(h)

    count = 0
    api_calls = 0
    for ds in daily_list:
        date_str = ds.get('date', '')

        # Use cache if we already have a narrative for this date
        if date_str in cache:
            ds['day_narrative'] = cache[date_str]
            count += 1
            continue

        rows = hourly_by_date.get(date_str, [])
        # Sort by hour to ensure chronological order
        rows.sort(key=lambda r: r.get('hour', 0))
        if len(rows) >= 8:
            narrative = _gemini_narrative(date_str, rows)
        else:
            narrative = _gemini_narrative_daily(date_str, ds)
        api_calls += 1
        if narrative:
            ds['day_narrative'] = narrative
            cache[date_str] = narrative
            count += 1
        if api_calls < len(daily_list):
            time.sleep(12)

    # Save cache only on CI
    if use_cache:
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        cache = {k: v for k, v in cache.items() if k >= today_str}
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    print(f"  [Gemini] Generisano {count}/{len(daily_list)} AI opisa ({api_calls} API poziva, {count - api_calls} iz keša).")


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

    all_data = corrected.copy()
    all_data['_date'] = all_data['datetime'].dt.strftime('%Y-%m-%d')
    all_data['_day_name'] = all_data['datetime'].dt.strftime('%A')
    all_data['hour'] = all_data['datetime'].dt.hour

    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    today_cache_path = os.path.join(OUTPUT_DIR, "today_daily_cache.json")

    all_daily = {}  # date_str -> summary dict
    for date_str, grp in all_data.groupby('_date'):
        day_name = grp.iloc[0]['_day_name']

        if date_str == today_str:
            first_hour = int(grp['hour'].min())
            if first_hour < 10:
                ds = _build_daily_summary(date_str, day_name, grp, fc_raw=fc_raw)
                all_daily[date_str] = ds
                try:
                    with open(today_cache_path, 'w', encoding='utf-8') as _cf:
                        json.dump(ds, _cf, ensure_ascii=False)
                except Exception:
                    pass
            else:
                cached = None
                if os.path.exists(today_cache_path):
                    try:
                        with open(today_cache_path, 'r', encoding='utf-8') as _cf:
                            cached = json.load(_cf)
                        if cached.get('date') != today_str:
                            cached = None
                    except Exception:
                        cached = None
                if cached:
                    all_daily[date_str] = cached
                else:
                    all_daily[date_str] = _build_daily_summary(
                        date_str, day_name, grp, fc_raw=fc_raw
                    )
        else:
            all_daily[date_str] = _build_daily_summary(
                date_str, day_name, grp, fc_raw=fc_raw
            )

    dates_48h = set()
    if len(forecast_hours) > 0:
        fc_df = pd.DataFrame(forecast_hours)
        dates_48h = set(fc_df['date'].unique())

    daily = []
    long_range = []
    for date_str in sorted(all_daily.keys()):
        ds = all_daily[date_str]
        if date_str in dates_48h:
            daily.append(ds)
        else:
            long_range.append(ds)

    if long_range:
        print(f"  Long range: {len(long_range)} dana")

    # Enrich narratives with Gemini AI (daily + long_range, uses ALL hourly data)
    all_days_for_ai = daily + long_range
    all_hourly = all_data.to_dict('records')
    _enrich_narratives_with_ai(all_days_for_ai, all_hourly)

    output = {
        "generated": now_str,
        "location": {"name": "Budva, Crna Gora", "lat": LAT, "lon": LON,
                      "station": "ibudva5 (Weather Underground)"},
        "method": "XGBoost Multi-Model Ensemble + Historical Bias + Forecast Revision v3",
        "description": "11 modela, 6 godina podataka (2020-2026), pametna korekcija + Day1/Day2 revizije",
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
        print(f"     Temp: {d.get('temp_min', '?')}° — {d.get('temp_max', '?')}°C  |  Vlaznost: {d.get('humidity_avg', '?')}%")
        print(f"     Vjetar: do {d.get('wind_max', '?')} m/s (udari {d.get('gust_max', '?')} m/s)")
        print(f"     Padavine: {d.get('precip_total', 0)} mm  |  Pritisak: {d.get('pressure_avg', '?')} hPa")
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

        tf = f"{t:5.1f}°" if pd.notna(t) else "  N/A"
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
