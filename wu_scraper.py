"""
Weather Underground Scraper za IBUDVA5 stanicu
Izvlači satne podatke za 2 godine unazad
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import re
import json

from wu_aggregation import resample_to_hourly

STATION_ID = "IBUDVA5"
OUTPUT_DIR = "wu_data"
YEARS_BACK = 2
START_DATE_OVERRIDE = "2020-04-01" 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def f_to_c(f):
    """Fahrenheit to Celsius"""
    if f is None:
        return None
    return round((f - 32) * 5/9, 2)

def in_to_mm(inches):
    """Inches to millimeters"""
    if inches is None:
        return None
    return round(inches * 25.4, 2)

def mph_to_ms(mph):
    """Miles per hour to meters per second"""
    if mph is None:
        return None
    return round(mph * 0.44704, 2)

def inhg_to_hpa(inhg):
    """Inches of mercury to hectopascals"""
    if inhg is None:
        return None
    return round(inhg * 33.8639, 2)

# The dashboard table is now rendered client-side, so the HTML holds no rows.
# The same 5-min records come from the PWS history API that the page calls.
# Fields match the old table exactly (units=e): tempAvg, dewptAvg, humidityAvg,
# windspeedAvg, windgustAvg, pressureMax, precipRate, precipTotal, uvHigh,
# solarRadiationHigh.
API_URL = "https://api.weather.com/v2/pws/history/all"
# Key used by the wunderground.com front end; set WU_API_KEY to use your own.
WEB_API_KEY = "6532d6454b8aa370768e63d6ba5a832e"
COMPASS = ['North', 'NNE', 'NE', 'ENE', 'East', 'ESE', 'SE', 'SSE',
           'South', 'SSW', 'SW', 'WSW', 'West', 'WNW', 'NW', 'NNW']


def api_key():
    return os.environ.get("WU_API_KEY") or WEB_API_KEY


def deg_to_compass(deg):
    if deg is None:
        return None
    return COMPASS[int((float(deg) % 360) / 22.5 + 0.5) % 16]


def _clock_text(ts):
    h12 = ts.hour % 12 or 12
    return f"{h12}:{ts.minute:02d} {'AM' if ts.hour < 12 else 'PM'}"


def observations_to_frame(observations, date_str):
    """API observations (units=e) -> rows with the raw_5min.csv columns."""
    rows = []
    for o in observations:
        imp = o.get('imperial') or {}
        local = o.get('obsTimeLocal')
        if not local:
            continue
        ts = pd.Timestamp(local).floor('min')
        if ts.strftime("%Y-%m-%d") != date_str:
            continue
        row = {
            'datetime': ts.strftime("%Y-%m-%d %H:%M:%S"),
            'date': date_str,
            'time': _clock_text(ts),
            'temp_f': imp.get('tempAvg'),
            'dewpoint_f': imp.get('dewptAvg'),
            'humidity_pct': o.get('humidityAvg'),
            'wind_dir': deg_to_compass(o.get('winddirAvg')),
            'wind_mph': imp.get('windspeedAvg'),
            'gust_mph': imp.get('windgustAvg'),
            'pressure_in': imp.get('pressureMax'),
            'precip_rate_in': imp.get('precipRate'),
            'precip_accum_in': imp.get('precipTotal'),
            'uv': o.get('uvHigh'),
            'solar_wm2': o.get('solarRadiationHigh'),
        }
        row['temp_c'] = f_to_c(row['temp_f'])
        row['dewpoint_c'] = f_to_c(row['dewpoint_f'])
        row['wind_ms'] = mph_to_ms(row['wind_mph'])
        row['gust_ms'] = mph_to_ms(row['gust_mph'])
        row['pressure_hpa'] = inhg_to_hpa(row['pressure_in'])
        row['precip_rate_mm'] = in_to_mm(row['precip_rate_in'])
        row['precip_accum_mm'] = in_to_mm(row['precip_accum_in'])
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates(subset=['datetime'], keep='last')


def scrape_day(date_str, session, retries=3):
    """
    Skini jedan dan 5-min podataka (date_str format: "2026-02-07").
    Vraca DataFrame sa kolonama raw_5min.csv ili None.
    """
    params = {
        'stationId': STATION_ID, 'format': 'json', 'units': 'e',
        'numericPrecision': 'decimal', 'date': date_str.replace('-', ''),
        'apiKey': api_key(),
    }
    headers = {'User-Agent': HEADERS['User-Agent'],
               'Origin': 'https://www.wunderground.com',
               'Referer': 'https://www.wunderground.com/'}
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=params, headers=headers, timeout=30)
            if response.status_code == 204:
                print("   Nema podataka (204)")
                return None
            if response.status_code in (401, 403):
                print(f"   HTTP {response.status_code}: {response.text[:80]} "
                      "(postavi WU_API_KEY)")
                return None
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            if response.status_code != 200:
                print(f"   HTTP {response.status_code}")
                return None
            df = observations_to_frame(response.json().get('observations', []), date_str)
            if df is None:
                print("   Nema podataka")
            return df
        except requests.exceptions.Timeout:
            print("   Timeout")
            time.sleep(5)
        except Exception as e:
            print(f"   Error: {str(e)[:80]}")
            return None
    return None

# resample_to_hourly is now imported from wu_aggregation.py
# Aggregation rules:
#   gust_ms -> MAX (was bug: was mean)
#   wind_ms -> mean (target) + adds wind_ms_max, wind_ms_p95 features
#   gust_ms -> MAX + adds gust_ms_p95
#   precip_rate_mm -> mean + precip_rate_max
#   solar_wm2 -> mean + solar_wm2_max
#   wind_dir -> vector mean by wind speed -> wind_dir_deg (0-360)

def main():
    """Glavna funkcija"""
    print("=" * 70)
    print("🌤 WEATHER UNDERGROUND SCRAPER")
    print(f"📍 Stanica: {STATION_ID}")
    print(f"📅 Period: {YEARS_BACK} godine unazad")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    end_date = datetime.now()
    if START_DATE_OVERRIDE:
        start_date = datetime.strptime(START_DATE_OVERRIDE, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=YEARS_BACK * 365)
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    total_days = len(dates)
    print(f"Ukupno dana: {total_days}")
    print("-" * 70)
    
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")
    all_data_file = os.path.join(OUTPUT_DIR, "all_data.csv")
    raw_5min_file = os.path.join(OUTPUT_DIR, "raw_5min.csv")

    completed_dates = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            completed_dates = set(progress.get('completed', []))
        print(f"Pronađen progress: {len(completed_dates)} dana već završeno")

    all_data = []
    if os.path.exists(all_data_file):
        existing_df = pd.read_csv(all_data_file)
        all_data = [existing_df]
        print(f"Postojeći hourly podaci: {len(existing_df)} redova")

    raw_chunks = []
    if os.path.exists(raw_5min_file):
        existing_raw = pd.read_csv(raw_5min_file)
        raw_chunks = [existing_raw]
        print(f"Postojeći raw 5-min podaci: {len(existing_raw)} redova")

    session = requests.Session()

    failed_dates = []
    new_completed = 0

    for i, date_str in enumerate(dates):
        if date_str in completed_dates:
            continue

        progress_pct = ((i + 1) / total_days) * 100
        print(f"[{i+1}/{total_days}] ({progress_pct:.1f}%) {date_str}...", end=" ")

        df = scrape_day(date_str, session)

        if df is not None and not df.empty:
            # Save raw 5-min data first
            raw_chunks.append(df.copy())
            hourly_df = resample_to_hourly(df)
            if hourly_df is not None:
                all_data.append(hourly_df)
                print(f"{len(df)} raw → {len(hourly_df)} hourly")
                completed_dates.add(date_str)
                new_completed += 1
            else:
                print(f"Resample failed")
                failed_dates.append(date_str)
        else:
            print(f"No data")
            failed_dates.append(date_str)

        if new_completed > 0 and new_completed % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump({'completed': list(completed_dates)}, f)
            # Flush raw to disk periodically (it's the irreplaceable data)
            if raw_chunks:
                pd.concat(raw_chunks, ignore_index=True).drop_duplicates(subset=['datetime']) \
                    .sort_values('datetime').to_csv(raw_5min_file, index=False)
            print(f"   💾 Progress saved ({len(completed_dates)} dana)")

        time.sleep(2)  # 2 sekunde pauza
    
    print("\n" + "=" * 70)
    print("💾 Čuvam podatke...")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['datetime'])
        final_df = final_df.sort_values('datetime')

        final_df.to_csv(all_data_file, index=False)
        print(f"Sačuvano hourly: {all_data_file}")
        print(f"   Redova: {len(final_df)}")
        print(f"   Period: {final_df['datetime'].min()} - {final_df['datetime'].max()}")

    if raw_chunks:
        raw_df = pd.concat(raw_chunks, ignore_index=True)
        raw_df = raw_df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        raw_df.to_csv(raw_5min_file, index=False)
        print(f"Sačuvano raw 5-min: {raw_5min_file}")
        print(f"   Redova: {len(raw_df)}")
    
    with open(progress_file, 'w') as f:
        json.dump({
            'completed': list(completed_dates),
            'failed': failed_dates
        }, f, indent=2)
    
    print(f"\nSTATISTIKA:")
    print(f"   Uspješno: {len(completed_dates)} dana")
    print(f"   Neuspješno: {len(failed_dates)} dana")
    
    if failed_dates:
        print(f"\nNeuspješni datumi sačuvani u: {progress_file}")
    
    print("\n" + "=" * 70)
    print("GOTOVO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
