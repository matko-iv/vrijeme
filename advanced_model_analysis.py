"""
Analiza gresaka vremenskih modela za Budvu.
Preuzima podatke sa Open-Meteo, poredi sa WU opservacijama.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json

LAT, LON = 42.29, 18.84
OBS_CSV = "wu_data\\ibudva5_hourly_3years_first_hour.csv"

START_DATE = "2023-02-10"
END_DATE = datetime.now().strftime("%Y-%m-%d")

MODELS = {
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ICON_SEAMLESS": "icon_seamless",
    "GFS_SEAMLESS": "gfs_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ARPEGE_EUROPE": "arpege_europe",
    "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",

}

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

CLOUD_THRESHOLD_SOLAR = 150
CLOUD_THRESHOLD_PERCENT = 60


def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj


def categorize_weather_conditions(df):
    df['month'] = df['datetime'].dt.month
    df['season'] = df['month'].apply(lambda m: 
        'Zima' if m in [12, 1, 2] else
        'Proleće' if m in [3, 4, 5] else
        'Ljeto' if m in [6, 7, 8] else 'Jesen'
    )

    df['hour'] = df['datetime'].dt.hour
    df['time_of_day'] = df['hour'].apply(lambda h:
        'Noć' if h < 6 or h >= 22 else
        'Jutro' if h < 12 else
        'Popodne' if h < 18 else 'Veče'
    )

    df['day_night'] = df['hour'].apply(lambda h: 'Dan' if 6 <= h < 20 else 'Noć')

    if 'precip_rate_mm' in df.columns:
        df['has_rain'] = (df['precip_rate_mm'].fillna(0) > 0.1) | (df['precip_accum_mm'].fillna(0) > 0.1)
        df['light_rain'] = (df['precip_accum_mm'].fillna(0) > 0.1) & (df['precip_accum_mm'].fillna(0) <= 2.0)
        df['heavy_rain'] = df['precip_accum_mm'].fillna(0) > 5.0
    else:
        df['has_rain'] = False
        df['light_rain'] = False
        df['heavy_rain'] = False

    if 'wind_ms' in df.columns:
        df['strong_wind'] = df['wind_ms'] > 8.0
        df['very_strong_wind'] = df['wind_ms'] > 12.0
    else:
        df['strong_wind'] = False
        df['very_strong_wind'] = False

    if 'wind_dir_obs' in df.columns and 'wind_ms' in df.columns:
        if df['wind_dir_obs'].dtype == 'object':
            direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
                'North': 0, 'NNorth': 0, 'South': 180, 'East': 90, 'West': 270
            }
            df['wind_dir_degrees'] = df['wind_dir_obs'].map(direction_map).fillna(0)
        else:
            df['wind_dir_degrees'] = df['wind_dir_obs']

        df['is_bura'] = (
            (((df['wind_dir_degrees'] >= 315) | (df['wind_dir_degrees'] <= 45)) & 
             (df['wind_ms'] > 8.0))
        )
        df['winter_bura'] = (df['season'] == 'Zima') & df['is_bura']
    else:
        df['is_bura'] = False
        df['winter_bura'] = False

    if 'solar_wm2' in df.columns:
        df['cloudy'] = ((df['solar_wm2'] < CLOUD_THRESHOLD_SOLAR) & (df['day_night'] == 'Dan'))
    else:
        df['cloudy'] = False

    if 'temp_obs' in df.columns:
        temp_p10 = df['temp_obs'].quantile(0.10)
        temp_p90 = df['temp_obs'].quantile(0.90)
        df['extreme_cold'] = df['temp_obs'] < temp_p10
        df['extreme_hot'] = df['temp_obs'] > temp_p90

    return df


def calculate_metrics(obs, model, name=""):
    mask = ~(obs.isna() | model.isna())
    if mask.sum() == 0:
        return None, None, None, 0

    o = obs[mask]
    m = model[mask]

    mae = float(np.mean(np.abs(o - m)))
    rmse = float(np.sqrt(np.mean((o - m)**2)))
    bias = float(np.mean(m - o))
    n = int(mask.sum())

    return mae, rmse, bias, n


def cloud_skill_metrics(df, obs_flag_col='cloudy', model_cloud_col='cloud_cover_model', threshold=60):
    mask = df[model_cloud_col].notna() & df[obs_flag_col].notna()
    if mask.sum() == 0:
        return None

    obs_cloudy = df.loc[mask, obs_flag_col].astype(bool)
    model_cloudy = df.loc[mask, model_cloud_col] >= threshold

    hits = int(((obs_cloudy == True) & (model_cloudy == True)).sum())
    misses = int(((obs_cloudy == True) & (model_cloudy == False)).sum())
    false_alarms = int(((obs_cloudy == False) & (model_cloudy == True)).sum())
    correct_neg = int(((obs_cloudy == False) & (model_cloudy == False)).sum())

    total = hits + misses + false_alarms + correct_neg
    if total == 0:
        return None

    accuracy = float((hits + correct_neg) / total)
    hit_rate = float(hits / (hits + misses)) if (hits + misses) > 0 else None
    false_alarm_rate = float(false_alarms / (false_alarms + correct_neg)) if (false_alarms + correct_neg) > 0 else None

    return {
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'correct_neg': correct_neg,
        'accuracy': accuracy,
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
        'threshold': threshold,
    }


def fetch_model_series(model_name, model_id):
    print(f"\nPreuzimam: {model_name} ({model_id})...")

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
    ]

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": hourly_vars,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "models": model_id,
    }

    max_retries = 3
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)

            if r.status_code == 429:
                if attempt < max_retries - 1:
                    print(f"Rate limit (429 (anti evropski)) - čekam {retry_delay} sekundi...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"   ❌ Rate limit - maksimalan broj pokušaja dostignut")
                    return None

            r.raise_for_status()
            data = r.json()
            break

        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  HTTP greška: {e} - pokušavam ponovo...")
                time.sleep(5)
                continue
            else:
                print(f"   ❌ Greška: {e}")
                return None
        except Exception as e:
            print(f"   ❌ Greška: {e}")
            return None

    if "hourly" not in data:
        print(f"   ⚠️  Nema podataka")
        return None

    hourly = data["hourly"]

    df_data = {"datetime": pd.to_datetime(hourly["time"])}

    for var in hourly_vars:
        if var in hourly:
            df_data[f"{var}_model"] = hourly[var]

    df = pd.DataFrame(df_data)
    print(f"   ✅ Preuzeto {len(df)} sati, {len(df_data)-1} varijabli")

    return df


def load_observed(csv_path):
    print(f"\nUčitavam opservacije...")

    obs = pd.read_csv(csv_path)
    obs['datetime'] = pd.to_datetime(obs['datetime'])

    obs = obs.rename(columns={
        'temp_c': 'temperature_2m_obs',
        'humidity_pct': 'relative_humidity_2m_obs',
        'dewpoint_c': 'dew_point_2m_obs',
        'wind_ms': 'wind_speed_10m_obs',
        'gust_ms': 'wind_gusts_10m_obs',
        'wind_dir': 'wind_direction_10m_obs',
        'pressure_hpa': 'pressure_msl_obs',
        'precip_rate_mm': 'precipitation_rate_obs',
        'precip_accum_mm': 'precipitation_obs',
        'solar_wm2': 'shortwave_radiation_obs',
        'uv': 'uv_index_obs'
    })

    print(f"Učitano {len(obs)} sati")

    return obs


def analyze_model_errors(model_name, merged_df):
    print(f"\n{'='*80}")
    print(f"DETALJNI IZVJEŠTAJ: {model_name}")
    print(f"{'='*80}")

    report = {
        'model_name': model_name,
        'overall': {},
        'overall_bias_all_vars': {},
        'by_season': {},
        'by_time_of_day': {},
        'by_weather': {},
        'by_extremes': {},
        'by_wind_conditions': {},
        'cloudiness_skill': {},
        'precipitation_by_intensity': {}
    }

    print("\nUKUPNE METRIKE:")

    var_pairs = [
        ('temperature_2m', 'Temperatura', '°C'),
        ('relative_humidity_2m', 'Vlažnost', '%'),
        ('dew_point_2m', 'Dewpoint', '°C'),
        ('wind_speed_10m', 'Brzina vjetra', 'm/s'),
        ('wind_gusts_10m', 'Udari vjetra', 'm/s'),
        ('pressure_msl', 'Pritisak', 'hPa'),
        ('precipitation', 'Padavine', 'mm'),
        ('cloud_cover', 'Oblačnost', '%'),
        ('shortwave_radiation', 'Solarna radijacija', 'W/m²'),
    ]

    for var_base, var_name, unit in var_pairs:
        obs_col = f'{var_base}_obs'
        model_col = f'{var_base}_model'

        if obs_col in merged_df.columns and model_col in merged_df.columns:
            mae, rmse, bias, n = calculate_metrics(
                merged_df[obs_col], 
                merged_df[model_col]
            )

            if mae is not None:
                print(f"   {var_name:25s}: MAE={mae:7.2f} {unit:5s}  RMSE={rmse:7.2f} {unit:5s}  Bias={bias:+7.2f} {unit:5s}  (n={n:,})")
                report['overall'][var_base] = {
                    'mae': mae, 'rmse': rmse, 'bias': bias, 'n': n, 'unit': unit
                }
                report['overall_bias_all_vars'][var_base] = {
                    'bias': bias,
                    'unit': unit,
                    'interpretation': 'Pregrijava' if bias > 0 and var_name == 'Temperatura' else 
                                      'Pothlađuje' if bias < 0 and var_name == 'Temperatura' else
                                      'Precjenjuje' if bias > 0 else 'Potcjenjuje'
                }

    if 'cloud_cover_model' in merged_df.columns:
        cloud_skill = cloud_skill_metrics(
            merged_df, 
            obs_flag_col='cloudy', 
            model_cloud_col='cloud_cover_model', 
            threshold=CLOUD_THRESHOLD_PERCENT
        )
        if cloud_skill:
            print(f"\n☁️  SKILL OBLAČNOSTI (prag: solar<{CLOUD_THRESHOLD_SOLAR}W/m², cloud>{CLOUD_THRESHOLD_PERCENT}%):")
            print(f"   Accuracy       = {cloud_skill['accuracy']*100:5.1f}%")
            if cloud_skill['hit_rate'] is not None:
                print(f"   Hit rate       = {cloud_skill['hit_rate']*100:5.1f}%  (kad je oblačno, koliko model pogodi)")
            if cloud_skill['false_alarm_rate'] is not None:
                print(f"   False alarm    = {cloud_skill['false_alarm_rate']*100:5.1f}%  (kad je sunčano, koliko kaže oblačno)")
            print(f"   Hits/Misses/FA/CorNeg = {cloud_skill['hits']}/{cloud_skill['misses']}/"
                  f"{cloud_skill['false_alarms']}/{cloud_skill['correct_neg']}")
            report['cloudiness_skill']['global'] = cloud_skill

    print("\nGREŠKE PO SEZONAMA:")

    for season in ['Zima', 'Proleće', 'Leto', 'Jesen']:
        mask = merged_df['season'] == season

        # Temperatura
        mae_T, rmse_T, bias_T, n_T = calculate_metrics(
            merged_df.loc[mask, 'temperature_2m_obs'],
            merged_df.loc[mask, 'temperature_2m_model']
        )

        # Padavine
        mae_P, rmse_P, bias_P, n_P = None, None, None, 0
        if 'precipitation_obs' in merged_df.columns and 'precipitation_model' in merged_df.columns:
            mae_P, rmse_P, bias_P, n_P = calculate_metrics(
                merged_df.loc[mask, 'precipitation_obs'],
                merged_df.loc[mask, 'precipitation_model']
            )

        if mae_T is not None:
            print(f"   {season:10s} Temp: MAE={mae_T:5.2f}°C  RMSE={rmse_T:5.2f}°C  Bias={bias_T:+5.2f}°C  (n={n_T:,})")
        if mae_P is not None:
            print(f"              Pad:  MAE={mae_P:5.2f}mm  RMSE={rmse_P:5.2f}mm  Bias={bias_P:+5.2f}mm  (n={n_P:,})")

        report['by_season'][season] = {
            'temperature_2m': {'mae': mae_T, 'rmse': rmse_T, 'bias': bias_T, 'n': n_T},
            'precipitation': {'mae': mae_P, 'rmse': rmse_P, 'bias': bias_P, 'n': n_P}
        }

    print("\nGREŠKE PO DOBU DANA:")

    for time_period in ['Noć', 'Jutro', 'Popodne', 'Veče']:
        mask = merged_df['time_of_day'] == time_period

        mae_T, rmse_T, bias_T, n_T = calculate_metrics(
            merged_df.loc[mask, 'temperature_2m_obs'],
            merged_df.loc[mask, 'temperature_2m_model']
        )

        mae_P, rmse_P, bias_P, n_P = None, None, None, 0
        if 'precipitation_obs' in merged_df.columns and 'precipitation_model' in merged_df.columns:
            mae_P, rmse_P, bias_P, n_P = calculate_metrics(
                merged_df.loc[mask, 'precipitation_obs'],
                merged_df.loc[mask, 'precipitation_model']
            )

        if mae_T is not None:
            print(f"   {time_period:10s} Temp: MAE={mae_T:5.2f}°C  RMSE={rmse_T:5.2f}°C  Bias={bias_T:+5.2f}°C  (n={n_T:,})")
        if mae_P is not None:
            print(f"              Pad:  MAE={mae_P:5.2f}mm  RMSE={rmse_P:5.2f}mm  Bias={bias_P:+5.2f}mm  (n={n_P:,})")

        report['by_time_of_day'][time_period] = {
            'temperature_2m': {'mae': mae_T, 'rmse': rmse_T, 'bias': bias_T, 'n': n_T},
            'precipitation': {'mae': mae_P, 'rmse': rmse_P, 'bias': bias_P, 'n': n_P}
        }

    print("\nGREŠKE PO VREMENSKIM USLOVIMA:")

    weather_conditions = {
        'Kiša': merged_df['has_rain'],
        'Bez kiše': ~merged_df['has_rain'],
        'Slab vjetar': ~merged_df['strong_wind'],
        'Oblačno': merged_df['cloudy'],
        'Sunčano': ~merged_df['cloudy'] & (merged_df['day_night'] == 'Dan'),
    }

    for cond_name, mask in weather_conditions.items():
        if mask.sum() < 100:
            continue

        mae_T, rmse_T, bias_T, n_T = calculate_metrics(
            merged_df.loc[mask, 'temperature_2m_obs'],
            merged_df.loc[mask, 'temperature_2m_model']
        )

        mae_P, rmse_P, bias_P, n_P = None, None, None, 0
        if 'precipitation_obs' in merged_df.columns and 'precipitation_model' in merged_df.columns:
            mae_P, rmse_P, bias_P, n_P = calculate_metrics(
                merged_df.loc[mask, 'precipitation_obs'],
                merged_df.loc[mask, 'precipitation_model']
            )

        if mae_T is not None:
            print(f"   {cond_name:15s} Temp: MAE={mae_T:5.2f}°C  RMSE={rmse_T:5.2f}°C  Bias={bias_T:+5.2f}°C  (n={n_T:,})")
        if mae_P is not None:
            print(f"                   Pad:  MAE={mae_P:5.2f}mm  RMSE={rmse_P:5.2f}mm  Bias={bias_P:+5.2f}mm  (n={n_P:,})")

        report['by_weather'][cond_name] = {
            'temperature_2m': {'mae': mae_T, 'rmse': rmse_T, 'bias': bias_T, 'n': n_T},
            'precipitation': {'mae': mae_P, 'rmse': rmse_P, 'bias': bias_P, 'n': n_P}
        }

    print("\nGREŠKE PO JAKOM VJETRU I BURI:")

    wind_conditions = {
        'Jak vjetar (>8 m/s)': merged_df['strong_wind'],
        'Vrlo jak vjetar (>12 m/s)': merged_df['very_strong_wind'],
        'BURA (SJ/SJI >8 m/s)': merged_df['is_bura'],
        'Zima + BURA': merged_df['winter_bura'],
    }

    for cond_name, mask in wind_conditions.items():
        if mask.sum() < 50:
            continue

        mae_T, rmse_T, bias_T, n_T = calculate_metrics(
            merged_df.loc[mask, 'temperature_2m_obs'],
            merged_df.loc[mask, 'temperature_2m_model']
        )

        mae_P, rmse_P, bias_P, n_P = None, None, None, 0
        if 'precipitation_obs' in merged_df.columns and 'precipitation_model' in merged_df.columns:
            mae_P, rmse_P, bias_P, n_P = calculate_metrics(
                merged_df.loc[mask, 'precipitation_obs'],
                merged_df.loc[mask, 'precipitation_model']
            )

        if mae_T is not None:
            print(f"   {cond_name:30s} Temp: MAE={mae_T:5.2f}°C  Bias={bias_T:+5.2f}°C  (n={n_T:,})")
        if mae_P is not None:
            print(f"                                  Pad:  MAE={mae_P:5.2f}mm  Bias={bias_P:+5.2f}mm  (n={n_P:,})")

        report['by_wind_conditions'][cond_name] = {
            'temperature_2m': {'mae': mae_T, 'rmse': rmse_T, 'bias': bias_T, 'n': n_T},
            'precipitation': {'mae': mae_P, 'rmse': rmse_P, 'bias': bias_P, 'n': n_P}
        }

    if 'precipitation_obs' in merged_df.columns and 'precipitation_model' in merged_df.columns:
        print("\nPADAVINE PO INTENZITETU:")

        rain_intensity = {
            'Slaba kiša (0.1-2mm)': merged_df['light_rain'],
            'Jaka kiša (>5mm)': merged_df['heavy_rain'],
        }

        for intensity_name, mask in rain_intensity.items():
            if mask.sum() < 50:
                continue

            mae_P, rmse_P, bias_P, n_P = calculate_metrics(
                merged_df.loc[mask, 'precipitation_obs'],
                merged_df.loc[mask, 'precipitation_model']
            )

            if mae_P is not None:
                print(f"   {intensity_name:25s}: MAE={mae_P:5.2f}mm  RMSE={rmse_P:5.2f}mm  Bias={bias_P:+5.2f}mm  (n={n_P:,})")
                report['precipitation_by_intensity'][intensity_name] = {
                    'mae': mae_P, 'rmse': rmse_P, 'bias': bias_P, 'n': n_P
                }

    print("\nGREŠKE U EKSTREMNIM SITUACIJAMA (temperatura):")

    extreme_conditions = {
        'Ekstremna hladnoća': merged_df['extreme_cold'],
        'Ekstremna vrućina': merged_df['extreme_hot'],
    }

    for cond_name, mask in extreme_conditions.items():
        mae, rmse, bias, n = calculate_metrics(
            merged_df.loc[mask, 'temperature_2m_obs'],
            merged_df.loc[mask, 'temperature_2m_model']
        )

        if mae is not None:
            print(f"   {cond_name:20s}: MAE={mae:5.2f}°C  RMSE={rmse:5.2f}°C  Bias={bias:+5.2f}°C  (n={n:,})")
            report['by_extremes'][cond_name] = {'mae': mae, 'rmse': rmse, 'bias': bias, 'n': n}

    return report


def main():
    print("=" * 80)
    print("NAPREDNA ANALIZA VREMENSKIH MODELA ZA BUDVU")
    print(f"Prag oblačnosti: solar < {CLOUD_THRESHOLD_SOLAR} W/m² = oblačno")
    print("Retry mehanizam: 429 → čeka 60 sekundi, 3 pokušaja")
    print("=" * 80)

    obs = load_observed(OBS_CSV)

    all_reports = []

    for model_name, model_id in MODELS.items():

        model_df = fetch_model_series(model_name, model_id)

        if model_df is None:
            continue

        merged = pd.merge(obs, model_df, on='datetime', how='inner')

        if len(merged) == 0:
            print(f"   ⚠️  Nema preklapanja")
            continue

        merged = categorize_weather_conditions(merged)

        merged['temp_obs'] = merged['temperature_2m_obs']
        merged['wind_ms'] = merged['wind_speed_10m_obs']
        merged['solar_wm2'] = merged.get('shortwave_radiation_obs', 0)
        merged['precip_rate_mm'] = merged.get('precipitation_rate_obs', 0)
        merged['precip_accum_mm'] = merged.get('precipitation_obs', 0)

        merged = categorize_weather_conditions(merged)

        report = analyze_model_errors(model_name, merged)
        all_reports.append(report)

        merged.to_csv(f"budva_{model_name}_detailed.csv", index=False)
        print(f"\n💾 Sačuvano: budva_{model_name}_detailed.csv")

        time.sleep(5)

    all_reports_clean = numpy_to_python(all_reports)

    with open('budva_detailed_error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(all_reports_clean, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ANAL ZAVRŠENA!")
    print("=" * 80)
    print("\nKreirane datoteke:")
    print("  • budva_detailed_error_analysis.json - svi izvještaji")
    print("  • budva_{MODEL}_detailed.csv - detaljni podaci po modelu")


if __name__ == "__main__":
    main()
