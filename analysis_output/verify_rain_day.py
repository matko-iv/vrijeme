"""Freeze the forecasts for one day, then score them against station IBUDVA5.

    python analysis_output/verify_rain_day.py snapshot 2026-10-08
        Writes analysis_output/verification_<date>/:
          published_forecasts.csv  every published forecast_48h.json (git history)
                                   with hours on <date>: lead, rain, PoP, raw blend,
                                   the ICON-2I value the pipeline saw, temperature, wind
          models_raw.csv           the latest raw Open-Meteo forecast of each model in
                                   trained_models_v2/api_cache (ICON-2I, ECMWF, ...)
          icon2i_raw.json          the cached ICON-2I response itself (all variables)
          snapshot.json            what was frozen and when

    python analysis_output/verify_rain_day.py verify 2026-10-08
        Fetches the station's 5-minute data from Weather Underground (the same page
        JSON the forecast pipeline used to scrape), turns it into hourly rain and
        temperature, re-reads the published forecasts (so runs published after the
        snapshot are included) and scores every forecast: rain hours (>= 0.2 mm), CSI,
        totals, first rain hour, PoP Brier score and temperature MAE. Writes
        verification.json and observations_hourly.csv next to the snapshot.

All hours are local time (Europe/Podgorica), labelled by their start: 13:00 is
rain from 13:00 to 14:00. The published JSON already uses that label; raw
Open-Meteo rows (rain over [t-1, t]) are moved back one hour to match.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = 'docs/forecast_data/forecast_48h.json'
CACHE = os.path.join(ROOT, 'trained_models_v2', 'api_cache')
WET_MM = 0.2
WU_URL = 'https://www.wunderground.com/dashboard/pws/IBUDVA5/table/{d}/{d}/daily'
WU_HEADERS = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                             '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
              'Accept-Language': 'en-US,en;q=0.5'}
MODEL_NAMES = {'italia_meteo_arpae_icon_2i': 'ICON2I', 'ecmwf_ifs025': 'ECMWF_IFS025',
               'ecmwf_ifs': 'ECMWF_IFS', 'gfs_seamless': 'GFS', 'icon_seamless': 'ICON_EU',
               'arpege_europe': 'ARPEGE', 'meteofrance_seamless': 'METEOFRANCE',
               'ukmo_seamless': 'UKMO', 'knmi_seamless': 'KNMI', 'dmi_seamless': 'DMI'}


def outdir(day):
    path = os.path.join(ROOT, 'analysis_output', f'verification_{day}')
    os.makedirs(path, exist_ok=True)
    return path


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout


def published_forecasts(day):
    """One row per (published run, hour on `day`)."""
    since = (pd.Timestamp(day) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    log = git('log', f'--since={since}', '--format=%H|%an|%s', '--', PUBLISHED).decode().splitlines()
    rows = []
    for line in log:
        sha, author, subject = line.split('|', 2)
        if author == 'github-actions[bot]':
            source = 'GitHub Actions'
        elif 'local run' in subject:
            source = 'Claude session run'
        else:
            source = f'manual ({author})'
        try:
            data = json.loads(git('show', f'{sha}:{PUBLISHED}'))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        issued = pd.Timestamp(data['generated'][:19])
        for h in data.get('hourly_forecast', []):
            if h.get('date') != day:
                continue
            valid = pd.Timestamp(h['datetime'][:19])
            rows.append({
                'issued': issued, 'commit': sha[:7],
                'source': source,
                'valid': valid, 'lead_h': round((valid - issued).total_seconds() / 3600, 1),
                'precipitation': h.get('precipitation'), 'pop': h.get('precipitation_pop'),
                'precipitation_raw': h.get('precipitation_raw'),
                'icon2i_seen': h.get('italiameteo_precipitation'),
                'temperature': h.get('temperature_2m'), 'wind': h.get('wind_speed_10m'),
                'gusts': h.get('wind_gusts_10m'), 'weather': h.get('weather_desc'),
            })
    return pd.DataFrame(rows).sort_values(['issued', 'valid']).reset_index(drop=True)


def cached_models(day):
    """Latest cached raw forecast per model, start-labelled hours on `day`."""
    frames, meta = [], {}
    for path in sorted(glob.glob(os.path.join(CACHE, 'forecast__*__42.2864_18.84.json'))):
        model_id = os.path.basename(path).split('__')[1]
        name = MODEL_NAMES.get(model_id, model_id)
        hourly = json.load(open(path))['hourly']
        df = pd.DataFrame(hourly).rename(columns={'time': 'valid'})
        df['valid'] = pd.to_datetime(df['valid'])
        rain = df[['valid', 'precipitation']].copy()
        rain['valid'] -= pd.Timedelta(hours=1)          # [t-1, t] -> start label t-1
        temp = df[['valid', 'temperature_2m', 'wind_speed_10m', 'wind_gusts_10m']]
        merged = temp.merge(rain, on='valid', how='left')
        merged = merged[merged['valid'].dt.strftime('%Y-%m-%d') == day]
        merged.insert(0, 'model', name)
        frames.append(merged)
        stamps = {}
        for kind in ('fetched', 'upstream'):
            side = f'{path}.{kind}'
            if os.path.exists(side):
                stamps[kind] = datetime.fromtimestamp(float(open(side).read().strip()), timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        meta[name] = stamps
    return pd.concat(frames, ignore_index=True), meta


def snapshot(day):
    out = outdir(day)
    pub = published_forecasts(day)
    pub.to_csv(os.path.join(out, 'published_forecasts.csv'), index=False)
    models, meta = cached_models(day)
    models.to_csv(os.path.join(out, 'models_raw.csv'), index=False)
    icon_path = os.path.join(CACHE, 'forecast__italia_meteo_arpae_icon_2i__42.2864_18.84.json')
    if os.path.exists(icon_path):
        with open(icon_path) as src, open(os.path.join(out, 'icon2i_raw.json'), 'w') as dst:
            dst.write(src.read())
    runs = pub.groupby('issued').agg(hours=('valid', 'size'), rain_mm=('precipitation', 'sum'),
                                     icon2i_seen_mm=('icon2i_seen', 'sum'))
    info = {'day': day, 'frozen_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'published_runs': {str(k): {kk: (round(float(vv), 1) if kk != 'hours' else int(vv))
                                        for kk, vv in v.items()} for k, v in runs.iterrows()},
            'cached_models': meta,
            'cached_model_rain_mm': models.groupby('model')['precipitation'].sum().round(1).to_dict()}
    with open(os.path.join(out, 'snapshot.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2, ensure_ascii=False))


def station_hourly(day):
    """Hourly station rain (mm over [T, T+1)) and mean temperature from WU 5-minute data."""
    for attempt in range(5):           # WU sometimes serves a light page without the data
        r = requests.get(WU_URL.format(d=day), headers=WU_HEADERS, timeout=60)
        r.raise_for_status()
        m = re.search(r'<script[^>]*id="app-root-state"[^>]*>(.*?)</script>', r.text, re.S)
        if m:
            break
        time.sleep(15)
    else:
        sys.exit('WU did not return the observation data; try again later')
    raw = (m.group(1).replace('&q;', '"').replace('&a;', '&').replace('&s;', "'")
           .replace('&l;', '<').replace('&g;', '>'))
    best = []
    for v in json.loads(raw).values():
        body = v.get('b') if isinstance(v, dict) else None
        if isinstance(body, dict) and isinstance(body.get('observations'), list):
            # the list can span several days (e.g. 7-9 Oct when asking for the 8th)
            obs = [o for o in body['observations'] if str(o.get('obsTimeLocal', '')).startswith(day)]
            if len(obs) > len(best):
                best = obs
    if not best:
        sys.exit(f'No WU observations found for {day}')
    five = pd.DataFrame([{'time': o['obsTimeLocal'],
                          'accum_mm': (o.get('imperial') or {}).get('precipTotal'),
                          'temp_f': (o.get('imperial') or {}).get('tempAvg')} for o in best])
    five['time'] = pd.to_datetime(five['time'])
    five['accum_mm'] = pd.to_numeric(five['accum_mm'], errors='coerce') * 25.4
    five['temp_c'] = (pd.to_numeric(five['temp_f'], errors='coerce') - 32) * 5 / 9
    five['hour'] = five['time'].dt.floor('h')
    hours = pd.date_range(day, periods=24, freq='h')
    acc = five.groupby('hour')['accum_mm'].max().reindex(hours)
    rain = acc.diff()
    rain.iloc[0] = acc.iloc[0]                     # running total resets at midnight
    rain[acc.isna() | acc.shift(1).isna()] = np.nan
    rain.iloc[0] = acc.iloc[0]
    temp = (five.assign(slot=(five['time'] + pd.Timedelta(minutes=30)).dt.floor('h'))
            .groupby('slot')['temp_c'].mean().reindex(hours))
    count = five.groupby('hour').size().reindex(hours).fillna(0).astype(int)
    return pd.DataFrame({'valid': hours, 'rain_mm': rain.clip(lower=0).round(2).values,
                         'temp_c': temp.round(2).values, 'readings': count.values})


def score(obs_rain, fc_rain, valid):
    ok = np.isfinite(obs_rain) & np.isfinite(fc_rain)
    o, f = obs_rain[ok] >= WET_MM, fc_rain[ok] >= WET_MM
    hits, misses, fas = int((o & f).sum()), int((o & ~f).sum()), int((~o & f).sum())
    first = lambda wet, idx: (idx[wet][0].strftime('%H:%M') if wet.any() else None)
    idx = pd.DatetimeIndex(valid)[ok]
    return {'hours': int(ok.sum()), 'observed_wet_h': int(o.sum()), 'forecast_wet_h': int(f.sum()),
            'hits': hits, 'misses': misses, 'false_alarms': fas,
            'csi': round(hits / max(hits + misses + fas, 1), 3),
            'observed_mm': round(float(obs_rain[ok].sum()), 1),
            'forecast_mm': round(float(fc_rain[ok].sum()), 1),
            'first_rain_observed': first(o, idx),
            'first_rain_forecast': first(f, idx)}


def verify(day):
    out = outdir(day)
    obs = station_hourly(day)
    obs.to_csv(os.path.join(out, 'observations_hourly.csv'), index=False)
    covered = obs['rain_mm'].notna()
    print(f"station: {int(covered.sum())}/24 hours with rain data, "
          f"{obs['rain_mm'].sum():.1f} mm, {int((obs['rain_mm'] >= WET_MM).sum())} wet hours")

    results = {'day': day, 'checked_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               'station_hours_with_rain_data': int(covered.sum()),
               'station_rain_mm': round(float(obs['rain_mm'].sum()), 1), 'forecasts': {}}
    o_rain = obs['rain_mm'].values.astype(float)

    pub = published_forecasts(day)
    for issued, run in pub.groupby('issued'):
        run = run.set_index('valid').reindex(obs['valid'])
        label = f"mine, issued {issued:%d.%m %H:%M} ({run['source'].dropna().iloc[0]})"
        res = score(o_rain, run['precipitation'].values.astype(float), obs['valid'])
        pop = run['pop'].values.astype(float)
        ok = np.isfinite(pop) & np.isfinite(o_rain)
        res['pop_brier'] = round(float(np.mean((pop[ok] - (o_rain[ok] >= WET_MM)) ** 2)), 3) if ok.any() else None
        t = run['temperature'].values.astype(float)
        okt = np.isfinite(t) & np.isfinite(obs['temp_c'].values)
        res['temp_mae'] = round(float(np.mean(np.abs(t[okt] - obs['temp_c'].values[okt]))), 2) if okt.any() else None
        res['mean_lead_h'] = round(float(run['lead_h'].mean()), 1)
        results['forecasts'][label] = res
        icon = run['icon2i_seen'].values.astype(float)
        if np.isfinite(icon).any():
            results['forecasts'][f"ICON-2I as seen {issued:%d.%m %H:%M}"] = score(o_rain, icon, obs['valid'])

    snap = os.path.join(out, 'models_raw.csv')
    if os.path.exists(snap):
        models = pd.read_csv(snap, parse_dates=['valid'])
        wide = models.pivot_table(index='valid', columns='model', values='precipitation').reindex(obs['valid'])
        for name in wide.columns:
            res = score(o_rain, wide[name].values.astype(float), obs['valid'])
            if name == 'ICON2I':
                t = (models[models.model == name].set_index('valid')['temperature_2m']
                     .reindex(obs['valid']).values.astype(float))
                okt = np.isfinite(t) & np.isfinite(obs['temp_c'].values)
                res['temp_mae'] = round(float(np.mean(np.abs(t[okt] - obs['temp_c'].values[okt]))), 2)
            results['forecasts'][f'raw {name} (snapshot)'] = res
        results['forecasts']['raw multi-model mean (snapshot)'] = score(o_rain, wide.mean(axis=1).values, obs['valid'])

    with open(os.path.join(out, 'verification.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'forecast':52s} {'mm fc/obs':>11s} {'wet h fc/obs':>12s} {'hits':>4s} {'miss':>4s} {'FA':>3s} {'CSI':>5s} {'first fc/obs':>13s} {'T MAE':>6s}")
    for name, r in results['forecasts'].items():
        print(f"{name:52s} {r['forecast_mm']:5.1f}/{r['observed_mm']:<5.1f} {r['forecast_wet_h']:5d}/{r['observed_wet_h']:<6d} "
              f"{r['hits']:4d} {r['misses']:4d} {r['false_alarms']:3d} {r['csi']:5.2f} "
              f"{str(r['first_rain_forecast']):>6s}/{str(r['first_rain_observed']):<6s} "
              f"{(r.get('temp_mae') if r.get('temp_mae') is not None else ''):>6}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['snapshot', 'verify'])
    p.add_argument('day', help='YYYY-MM-DD')
    a = p.parse_args()
    os.chdir(ROOT)
    snapshot(a.day) if a.action == 'snapshot' else verify(a.day)
