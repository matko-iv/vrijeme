"""
Weather Underground Scraper za IBUDVA5 stanicu
Izvlači satne podatke za 2 godine unazad
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import re
import json

# Konfiguracija
STATION_ID = "IBUDVA5"
BASE_URL = f"https://www.wunderground.com/dashboard/pws/{STATION_ID}/table"
OUTPUT_DIR = "wu_data"
YEARS_BACK = 2

# Headers da izgleda kao browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def parse_value(text):
    """Parsira vrijednost iz teksta (npr. '52.4 °F' -> 52.4)"""
    if not text or text.strip() == '--':
        return None
    # Izvuci broj
    match = re.search(r'([\d.]+)', text.replace(',', ''))
    if match:
        return float(match.group(1))
    return None

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

def scrape_day(date_str, session):
    """
    Scrape jedan dan podataka
    date_str format: "2026-02-07"
    """
    url = f"{BASE_URL}/{date_str}/{date_str}/daily"
    
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            print(f"   ❌ HTTP {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Pronađi tabelu sa podacima
        # WU koristi Angular, pa podaci mogu biti u JSON-u ili tabeli
        rows = []
        
        # Traži tabelu
        tables = soup.find_all('table')
        
        for table in tables:
            tbody = table.find('tbody')
            if not tbody:
                continue
                
            for tr in tbody.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 11:  # Očekujemo 11 kolona
                    try:
                        time_text = tds[0].get_text(strip=True)
                        
                        # Parsiraj vrijeme
                        if 'AM' in time_text or 'PM' in time_text:
                            try:
                                time_obj = datetime.strptime(f"{date_str} {time_text}", "%Y-%m-%d %I:%M %p")
                            except:
                                continue
                        else:
                            continue
                        
                        row = {
                            'datetime': time_obj.strftime("%Y-%m-%d %H:%M:%S"),
                            'date': date_str,
                            'time': time_text,
                            'temp_f': parse_value(tds[1].get_text()),
                            'dewpoint_f': parse_value(tds[2].get_text()),
                            'humidity_pct': parse_value(tds[3].get_text()),
                            'wind_dir': tds[4].get_text(strip=True) if len(tds) > 4 else None,
                            'wind_mph': parse_value(tds[5].get_text()) if len(tds) > 5 else None,
                            'gust_mph': parse_value(tds[6].get_text()) if len(tds) > 6 else None,
                            'pressure_in': parse_value(tds[7].get_text()) if len(tds) > 7 else None,
                            'precip_rate_in': parse_value(tds[8].get_text()) if len(tds) > 8 else None,
                            'precip_accum_in': parse_value(tds[9].get_text()) if len(tds) > 9 else None,
                            'uv': parse_value(tds[10].get_text()) if len(tds) > 10 else None,
                            'solar_wm2': parse_value(tds[11].get_text()) if len(tds) > 11 else None,
                        }
                        
                        # Konvertuj u metrički sistem
                        row['temp_c'] = f_to_c(row['temp_f'])
                        row['dewpoint_c'] = f_to_c(row['dewpoint_f'])
                        row['wind_ms'] = mph_to_ms(row['wind_mph'])
                        row['gust_ms'] = mph_to_ms(row['gust_mph'])
                        row['pressure_hpa'] = inhg_to_hpa(row['pressure_in'])
                        row['precip_rate_mm'] = in_to_mm(row['precip_rate_in'])
                        row['precip_accum_mm'] = in_to_mm(row['precip_accum_in'])
                        
                        rows.append(row)
                    except Exception as e:
                        continue
        
        if rows:
            return pd.DataFrame(rows)
        else:
            # Možda stranica koristi JavaScript - probaj izvući iz script tagova
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'observations' in script.string:
                    # Probaj parsirati JSON
                    try:
                        match = re.search(r'observations["\s:]+(\[.*?\])', script.string, re.DOTALL)
                        if match:
                            data = json.loads(match.group(1))
                            # Parsiraj...
                            pass
                    except:
                        pass
            
            print(f"   ⚠️ Nema podataka u tabeli")
            return None
            
    except requests.exceptions.Timeout:
        print(f"   ⏱️ Timeout")
        return None
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}")
        return None

def resample_to_hourly(df):
    """Resample podatke na satnu rezoluciju (prosjek)"""
    if df is None or df.empty:
        return None
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Numeričke kolone za prosjek
    numeric_cols = ['temp_c', 'dewpoint_c', 'humidity_pct', 'wind_ms', 'gust_ms', 
                    'pressure_hpa', 'precip_rate_mm', 'uv', 'solar_wm2']
    
    # Resample na sat
    hourly = df[numeric_cols].resample('1H').mean()
    
    # Padavine - suma umjesto prosjeka za akumulaciju
    hourly['precip_accum_mm'] = df['precip_accum_mm'].resample('1H').max()
    
    # Smjer vjetra - uzmi modus ili prvi
    hourly['wind_dir'] = df['wind_dir'].resample('1H').first()
    
    return hourly.reset_index()

def main():
    """Glavna funkcija"""
    print("=" * 70)
    print("🌤️  WEATHER UNDERGROUND SCRAPER")
    print(f"📍 Stanica: {STATION_ID}")
    print(f"📅 Period: {YEARS_BACK} godine unazad")
    print("=" * 70)
    
    # Kreiraj output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generiši listu datuma
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS_BACK * 365)
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    total_days = len(dates)
    print(f"📊 Ukupno dana: {total_days}")
    print("-" * 70)
    
    # Provjeri postojeći progress
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")
    all_data_file = os.path.join(OUTPUT_DIR, "all_data.csv")
    
    completed_dates = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            completed_dates = set(progress.get('completed', []))
        print(f"✅ Pronađen progress: {len(completed_dates)} dana već završeno")
    
    # Učitaj postojeće podatke
    all_data = []
    if os.path.exists(all_data_file):
        existing_df = pd.read_csv(all_data_file)
        all_data = [existing_df]
        print(f"✅ Postojeći podaci: {len(existing_df)} redova")
    
    # Session za requests
    session = requests.Session()
    
    # Scrape
    failed_dates = []
    new_completed = 0
    
    for i, date_str in enumerate(dates):
        # Preskoči već završene
        if date_str in completed_dates:
            continue
        
        progress_pct = ((i + 1) / total_days) * 100
        print(f"[{i+1}/{total_days}] ({progress_pct:.1f}%) {date_str}...", end=" ")
        
        df = scrape_day(date_str, session)
        
        if df is not None and not df.empty:
            # Resample na satnu rezoluciju
            hourly_df = resample_to_hourly(df)
            if hourly_df is not None:
                all_data.append(hourly_df)
                print(f"✅ {len(df)} → {len(hourly_df)} satnih")
                completed_dates.add(date_str)
                new_completed += 1
            else:
                print(f"⚠️ Resample failed")
                failed_dates.append(date_str)
        else:
            print(f"❌ No data")
            failed_dates.append(date_str)
        
        # Sačuvaj progress svakih 10 dana
        if new_completed > 0 and new_completed % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump({'completed': list(completed_dates)}, f)
            print(f"   💾 Progress saved ({len(completed_dates)} dana)")
        
        # Rate limiting - pauza između zahtjeva
        time.sleep(2)  # 2 sekunde pauza
    
    # Finalno čuvanje
    print("\n" + "=" * 70)
    print("💾 Čuvam podatke...")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['datetime'])
        final_df = final_df.sort_values('datetime')
        
        final_df.to_csv(all_data_file, index=False)
        print(f"✅ Sačuvano: {all_data_file}")
        print(f"   Redova: {len(final_df)}")
        print(f"   Period: {final_df['datetime'].min()} - {final_df['datetime'].max()}")
    
    # Sačuvaj finalni progress
    with open(progress_file, 'w') as f:
        json.dump({
            'completed': list(completed_dates),
            'failed': failed_dates
        }, f, indent=2)
    
    print(f"\n📊 STATISTIKA:")
    print(f"   ✅ Uspješno: {len(completed_dates)} dana")
    print(f"   ❌ Neuspješno: {len(failed_dates)} dana")
    
    if failed_dates:
        print(f"\n⚠️ Neuspješni datumi sačuvani u: {progress_file}")
    
    print("\n" + "=" * 70)
    print("✅ GOTOVO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
