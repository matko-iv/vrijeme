"""Day-ahead skill of ICON-2I vs the multi-model rain classifier.

Same protocol as analysis_output/rain_pop_day_ahead.py (production features
and parameters, walk-forward on the short-lead archive, 3-day embargo, applied
to runs made one day earlier), with ITALIAMETEO_ICON2I added as a member. ICON-2I
has no file in previous_runs_data/, so its previous runs (and the 8 other
members after previous_runs_data/ ends) are fetched from Open-Meteo once and
cached; an interrupted fetch (quota) resumes where it stopped.

Scores, on the same day-ahead hours and per season:
  - ICON-2I alone (>= 0.1 mm, and with its +/-1 h window),
  - the classifier with and without ICON-2I (ramped threshold, as in production),
  - "trust ICON-2I when dry" rules,
  - daily totals (classifier amount model, ICON-2I, ensemble mean) vs the station,
and refits the day-ahead PoP calibration with ICON-2I present.

    python analysis_output/rain_icon2i_day_ahead.py [--cache DIR]   (~10 min)
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('FC_DEVICE', 'cpu')
import forecast_48h_v3 as fc  # noqa: E402

API = 'https://previous-runs-api.open-meteo.com/v1/forecast'
ICON = 'ITALIAMETEO_ICON2I'
OTHERS = ['ARPEGE_EUROPE', 'GFS_SEAMLESS', 'ICON_SEAMLESS', 'METEOFRANCE', 'ECMWF_IFS025',
          'UKMO_SEAMLESS', 'KNMI_SEAMLESS', 'DMI_SEAMLESS']
VARS = ['precipitation', 'weather_code', 'cloud_cover', 'relative_humidity_2m', 'pressure_msl',
        'wind_speed_10m', 'wind_gusts_10m', 'temperature_2m', 'dew_point_2m']
ICON_START = date(2025, 4, 13)            # first day of ICON-2I in the archive
EXTEND_FROM = date(2026, 2, 20)           # previous_runs_data/ ends 2026-02-28
BLOCKS = [('2025-06-01', '2025-10-01'), ('2025-10-01', '2026-03-01'),
          ('2026-03-01', '2026-06-01'), ('2026-06-01', '2026-08-19')]
WARM = (6, 7, 8, 9)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', default=os.path.join(ROOT, 'analysis_output', 'cache_previous_runs'))
    return p.parse_args()


def fetch(model, start, end, cache):
    """Previous-runs rows (same-day and day-ahead) for one model, 30-day chunks, cached."""
    os.makedirs(cache, exist_ok=True)
    hourly = ','.join(VARS + [f'{v}_previous_day1' for v in VARS])
    frames, s = [], start
    while s <= end:
        e = min(s + timedelta(days=29), end)
        path = os.path.join(cache, f'{model}_{s}_{e}.csv')
        if not os.path.exists(path):
            params = dict(latitude=fc.LAT, longitude=fc.LON, hourly=hourly, models=fc.MODEL_IDS[model],
                          timezone=fc.FORECAST_TIMEZONE, start_date=s.isoformat(), end_date=e.isoformat())
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(API + '?' + urllib.parse.urlencode(params), timeout=180) as r:
                        payload = json.load(r)
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read()[:300].decode('utf-8', 'replace')
                    if 'limit exceeded' in body:
                        sys.exit(f'Open-Meteo quota exhausted at {model} {s}; re-run later (cache kept).')
                    print(f'  {model} {s}: HTTP {exc.code}, retry {attempt + 1}', flush=True)
                except Exception as exc:          # connection resets through the proxy
                    print(f'  {model} {s}: {exc}, retry {attempt + 1}', flush=True)
                time.sleep(10 * (attempt + 1))
            else:
                sys.exit(f'Fetch failed for {model} {s}; re-run later (cache kept).')
            pd.DataFrame(payload['hourly']).rename(columns={'time': 'datetime'}).to_csv(path, index=False)
            time.sleep(1.0)
        frames.append(pd.read_csv(path, parse_dates=['datetime']))
        s = e + timedelta(days=1)
    return pd.concat(frames, ignore_index=True).drop_duplicates('datetime').set_index('datetime')


def previous_runs(model, cache, last_day):
    """previous_runs_data/ where it exists, Open-Meteo after it (and for ICON-2I)."""
    if model == ICON:
        return fetch(ICON, ICON_START, last_day, cache)
    old = (pd.read_csv(f'previous_runs_data/{model}_previous_runs.csv', parse_dates=['datetime'],
                       low_memory=False).drop_duplicates('datetime').set_index('datetime'))
    new = fetch(model, EXTEND_FROM, last_day, cache)
    keep = [c for c in new.columns if c in old.columns or c.startswith('weather_code')]
    return pd.concat([old[old.index < pd.Timestamp(EXTEND_FROM)], new[keep]]).sort_index()


def scores(truth, decision):
    truth, decision = np.asarray(truth, bool), np.asarray(decision, bool)
    hits = int((truth & decision).sum())
    misses = int((truth & ~decision).sum())
    false_alarms = int((~truth & decision).sum())
    return {'csi': round(hits / max(hits + misses + false_alarms, 1), 4),
            'pod': round(hits / max(hits + misses, 1), 4),
            'far': round(false_alarms / max(hits + false_alarms, 1), 4),
            'bias': round((hits + false_alarms) / max(hits + misses, 1), 3)}


def main():
    args = parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        hist = fc.load_historical_data()
    hist = hist.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    times = pd.to_datetime(hist['datetime'])
    obs = pd.to_numeric(hist['_derived_precip_obs'], errors='coerce')
    y = (obs >= fc.CORRECTED_RAIN_THRESHOLD_MM).astype(int).values
    labelled = obs.notna().values
    last_day = times[labelled].max().date()

    runs = {m: previous_runs(m, args.cache, last_day) for m in [ICON] + OTHERS}

    def member_frame(models, lag):
        out = pd.DataFrame({'datetime': times})
        for m in models:
            for v in VARS:
                col = f'{m}_{v}_model'
                if lag is None:
                    out[col] = pd.to_numeric(hist[col], errors='coerce').values if col in hist else np.nan
                else:
                    src = f'{v}_{lag}'
                    out[col] = (pd.to_numeric(runs[m][src], errors='coerce').reindex(times).values
                                if src in runs[m].columns else np.nan)
        return out

    with_icon = [ICON] + OTHERS
    F = {('icon', 'd0'): fc.rain_occurrence_features(member_frame(with_icon, None), models=with_icon),
         ('icon', 'd1'): fc.rain_occurrence_features(member_frame(with_icon, 'previous_day1'), models=with_icon),
         ('base', 'd0'): fc.rain_occurrence_features(member_frame(OTHERS, None), models=OTHERS),
         ('base', 'd1'): fc.rain_occurrence_features(member_frame(OTHERS, 'previous_day1'), models=OTHERS)}
    cols = {k: [c for c in F[(k, 'd0')].columns
                if F[(k, 'd0')][c].notna().mean() > 0.1 and F[(k, 'd1')][c].notna().mean() > 0.1]
            for k in ('icon', 'base')}
    icon_d1 = pd.to_numeric(runs[ICON]['precipitation_previous_day1'], errors='coerce').reindex(times).values
    icon_d1_w3 = F[('icon', 'd1')][f'{ICON}_w3max'].values

    parts = []
    for start, end in BLOCKS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        fit = labelled & (times < start - pd.Timedelta(days=fc.RAIN_OCC_EMBARGO_DAYS)).values
        test = (labelled & (times >= start).values & (times < end).values & np.isfinite(icon_d1)
                & (F[('base', 'd1')]['n_avail'] >= 6).values & (F[('base', 'd0')]['n_avail'] >= 6).values)
        part = pd.DataFrame({'datetime': times[test].values, 'obs': obs[test].values, 'y': y[test],
                             'icon_d1': icon_d1[test], 'icon_d1_w3': icon_d1_w3[test],
                             'ens_d1': F[('icon', 'd1')]['ens_mean'].values[test]})
        for k in ('icon', 'base'):
            Xfit = F[(k, 'd0')].loc[fit & (F[(k, 'd0')]['n_avail'] >= 5).values, cols[k]]
            yfit = y[fit & (F[(k, 'd0')]['n_avail'] >= 5).values]
            clf = fc._new_xgb_classifier(**fc.RAIN_OCC_CLF_PARAMS)
            clf.fit(Xfit, yfit, verbose=False)
            part[f'p0_{k}'] = clf.predict_proba(F[(k, 'd0')].loc[test, cols[k]])[:, 1]
            part[f'p1_{k}'] = clf.predict_proba(F[(k, 'd1')].loc[test, cols[k]])[:, 1]
            if k == 'icon':
                wet = fit & (y == 1) & (F[(k, 'd0')]['n_avail'] >= 5).values
                amt = fc._new_xgb_regressor(**fc.RAIN_OCC_AMOUNT_PARAMS)
                amt.fit(F[(k, 'd0')].loc[wet, cols[k]], np.sqrt(obs[wet].values), verbose=False)
                part['amt1'] = np.clip(np.square(np.clip(amt.predict(F[(k, 'd1')].loc[test, cols[k]]), 0, None)),
                                       fc.CORRECTED_RAIN_THRESHOLD_MM, 50.0)
        parts.append(part)
        print(f'{start.date()}..{end.date()}: {int(test.sum())} h, {int(y[test].sum())} wet', flush=True)
    res = pd.concat(parts, ignore_index=True)
    res['season'] = np.where(pd.to_datetime(res['datetime']).dt.month.isin(WARM), 'warm_jun_sep', 'cool_oct_may')

    tau0 = fc._rain_occ_best_threshold(res['y'].values, res['p0_icon'].values)[0]
    ramp = fc.RAIN_OCC_LONG_LEAD_FACTOR * tau0
    rules = {
        'icon2i_ge_0.1mm': res['icon_d1'] >= 0.1,
        'icon2i_pm1h_ge_0.1mm': res['icon_d1_w3'] >= 0.1,
        'classifier_without_icon2i': res['p1_base'] >= ramp,
        'classifier_with_icon2i': res['p1_icon'] >= ramp,
        'ensemble_mean_ge_0.2mm': res['ens_d1'] >= 0.2,
    }
    for dry_tau in (0.30, 0.40, 0.50):
        rules[f'with_icon2i_dry_needs_{dry_tau:.2f}'] = np.where(
            res['icon_d1_w3'] >= 0.1, res['p1_icon'] >= ramp, res['p1_icon'] >= dry_tau)
    report = {'hours': int(len(res)), 'wet_hours': int(res['y'].sum()),
              'period': [str(res['datetime'].min()), str(res['datetime'].max())],
              'same_day_csi_tau': tau0, 'day_ahead_threshold': round(ramp, 4), 'decisions': {}}
    for name, decision in rules.items():
        row = {'all': scores(res['y'], decision)}
        for s in ('cool_oct_may', 'warm_jun_sep'):
            m = (res['season'] == s).values
            row[s] = scores(res['y'].values[m], np.asarray(decision)[m])
        report['decisions'][name] = row

    # reliability of the day-ahead probability with ICON-2I, and its Platt refit
    bins = [0, .1, .2, .3, .4, .5, .7, 1.01]
    report['reliability_day_ahead_with_icon2i'] = [
        {'bin': [a, min(b, 1.0)], 'n': int(((res.p1_icon >= a) & (res.p1_icon < b)).sum()),
         'observed': round(float(res.y[(res.p1_icon >= a) & (res.p1_icon < b)].mean()), 3)}
        for a, b in zip(bins[:-1], bins[1:]) if ((res.p1_icon >= a) & (res.p1_icon < b)).any()]
    p = np.clip(res['p1_icon'].values, 1e-4, 1 - 1e-4)
    lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), res['y'].values)
    a, b = fc.RAIN_OCC_DAY_AHEAD_PLATT
    current = 1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))
    refit = lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    ll = lambda q: float(-np.mean(res.y * np.log(np.clip(q, 1e-4, 1)) + (1 - res.y) * np.log(np.clip(1 - q, 1e-4, 1))))
    report['platt'] = {'refit_a': round(float(lr.intercept_[0]), 3), 'refit_b': round(float(lr.coef_[0][0]), 3),
                       'log_loss': {'raw': round(ll(p), 4), 'production_constants': round(ll(current), 4),
                                    'refit_in_sample': round(ll(refit), 4)}}

    # daily totals (row t = rain over [t-1, t])
    res['date'] = (pd.to_datetime(res['datetime']) - pd.Timedelta(hours=1)).dt.date
    res['clf_mm'] = np.where(res['p1_icon'] >= ramp, res['amt1'], 0.0)
    day = res.groupby('date').agg(obs=('obs', 'sum'), n=('obs', 'size'), icon=('icon_d1', 'sum'),
                                  clf=('clf_mm', 'sum'), ens=('ens_d1', 'sum'), season=('season', 'first'))
    day = day[day.n >= 20]
    report['daily_totals'] = {}
    for s in ('all', 'cool_oct_may', 'warm_jun_sep'):
        d = day if s == 'all' else day[day.season == s]
        wet = d[(d.obs >= 1) | (d.icon >= 1) | (d.clf >= 1) | (d.ens >= 1)]
        report['daily_totals'][s] = {
            'days': int(len(d)), 'observed_mm': round(float(d.obs.sum()), 1),
            **{k: {'total_ratio': round(float(d[k].sum() / max(d.obs.sum(), 1e-9)), 2),
                   'wet_day_mae_mm': round(float((wet[k] - wet.obs).abs().mean()), 2)}
               for k in ('icon', 'clf', 'ens')}}

    print(json.dumps(report, indent=1))
    with open('analysis_output/rain_icon2i_day_ahead.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('-> analysis_output/rain_icon2i_day_ahead.json')


if __name__ == '__main__':
    main()
