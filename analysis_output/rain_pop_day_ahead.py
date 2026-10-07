"""Day-ahead calibration of the rain classifier's PoP (RAIN_OCC_DAY_AHEAD_PLATT).

The production classifier learns from short-lead archive forecasts. To see how it
behaves on older runs, the same recipe (production features, parameters and
3-day embargo) is trained walk-forward on the archive, restricted to the 8
members and 9 variables that previous_runs_data/ also holds, and applied to the
same hours using the runs from one day earlier (Open-Meteo previous_day1).

Prints the reliability of same-day and day-ahead probabilities, scores the
current decision rule, and fits logit(p') = a + b*logit(p) on the day-ahead
hours (cross-fitted on two time halves for the held-out scores).

    python analysis_output/rain_pop_day_ahead.py   (~5 min on 4 CPUs)
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('FC_DEVICE', 'cpu')
import forecast_48h_v3 as fc  # noqa: E402

MODELS = ['ARPEGE_EUROPE', 'GFS_SEAMLESS', 'ICON_SEAMLESS', 'METEOFRANCE', 'ECMWF_IFS025',
          'UKMO_SEAMLESS', 'KNMI_SEAMLESS', 'DMI_SEAMLESS']
VARS = ['precipitation', 'weather_code', 'cloud_cover', 'relative_humidity_2m', 'pressure_msl',
        'wind_speed_10m', 'wind_gusts_10m', 'temperature_2m', 'dew_point_2m']
BLOCKS = [('2024-06-01', '2024-12-01'), ('2024-12-01', '2025-06-01'),
          ('2025-06-01', '2025-12-01'), ('2025-12-01', '2026-03-01')]
POP_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]

with contextlib.redirect_stdout(io.StringIO()):
    hist = fc.load_historical_data()
hist = hist.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
times = pd.to_datetime(hist['datetime'])
obs = pd.to_numeric(hist['_derived_precip_obs'], errors='coerce')
y = (obs >= fc.CORRECTED_RAIN_THRESHOLD_MM).astype(int).values
labelled = obs.notna().values


def member_frame(lag):
    """Archive values (lag None) or previous-runs values for one lag."""
    out = pd.DataFrame({'datetime': times})
    for m in MODELS:
        runs = None
        if lag is not None:
            runs = (pd.read_csv(f'previous_runs_data/{m}_previous_runs.csv', parse_dates=['datetime'],
                                low_memory=False)
                    .drop_duplicates('datetime').set_index('datetime'))
        for v in VARS:
            col = f'{m}_{v}_model'
            if runs is None:
                vals = pd.to_numeric(hist[col], errors='coerce').values if col in hist else np.nan
            else:
                src = f'{v}_{lag}'
                vals = (pd.to_numeric(runs[src], errors='coerce').reindex(times).values
                        if src in runs.columns else np.nan)
            out[col] = vals
    return out


same_day = fc.rain_occurrence_features(member_frame(None), models=MODELS)
day_ahead = fc.rain_occurrence_features(member_frame('previous_day1'), models=MODELS)
cols = [c for c in same_day.columns
        if same_day[c].notna().mean() > 0.3 and day_ahead[c].notna().mean() > 0.3]

parts = []
for start, end in BLOCKS:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    fit = (labelled & (times < start - pd.Timedelta(days=fc.RAIN_OCC_EMBARGO_DAYS)).values
           & (same_day['n_avail'] >= 5).values)
    clf = fc._new_xgb_classifier(**fc.RAIN_OCC_CLF_PARAMS)
    clf.fit(same_day.loc[fit, cols], y[fit], verbose=False)
    test = (labelled & (times >= start).values & (times < end).values
            & (same_day['n_avail'] >= 6).values & (day_ahead['n_avail'] >= 6).values)
    parts.append(pd.DataFrame({
        'datetime': times[test].values, 'y': y[test],
        'p_same_day': clf.predict_proba(same_day.loc[test, cols])[:, 1],
        'p_day_ahead': clf.predict_proba(day_ahead.loc[test, cols])[:, 1],
    }))
res = pd.concat(parts, ignore_index=True)
res['cool'] = ~pd.to_datetime(res['datetime']).dt.month.isin([6, 7, 8, 9]).values


def reliability(mask, col):
    rows = []
    for lo, hi in zip(POP_BINS[:-1], POP_BINS[1:]):
        m = mask & (res[col] >= lo).values & (res[col] < hi).values
        if m.any():
            rows.append({'bin': [lo, min(hi, 1.0)], 'n': int(m.sum()),
                         'observed': round(float(res['y'].values[m].mean()), 3)})
    return rows


def scores(truth, decision):
    truth, decision = np.asarray(truth, bool), np.asarray(decision, bool)
    hits = int((truth & decision).sum())
    misses = int((truth & ~decision).sum())
    false_alarms = int((~truth & decision).sum())
    return {'csi': round(hits / max(hits + misses + false_alarms, 1), 4),
            'pod': round(hits / max(hits + misses, 1), 4),
            'far': round(false_alarms / max(hits + false_alarms, 1), 4),
            'bias': round((hits + false_alarms) / max(hits + misses, 1), 3)}


everything = np.ones(len(res), bool)
report = {'hours': int(len(res)), 'wet_hours': int(res['y'].sum()),
          'period': [str(res['datetime'].min()), str(res['datetime'].max())],
          'reliability': {}}
for name, mask in (('all', everything), ('cool_oct_may', res['cool'].values),
                   ('warm_jun_sep', ~res['cool'].values)):
    report['reliability'][name] = {col: reliability(mask, col)
                                   for col in ('p_same_day', 'p_day_ahead')}

tau = fc._rain_occ_best_threshold(res['y'].values, res['p_same_day'].values)[0]
report['same_day_csi_tau'] = tau
report['decisions'] = {
    'same_day_tau': scores(res['y'], res['p_same_day'] >= tau),
    'day_ahead_ramped': scores(res['y'], res['p_day_ahead'] >= fc.RAIN_OCC_LONG_LEAD_FACTOR * tau),
    'day_ahead_unramped': scores(res['y'], res['p_day_ahead'] >= tau),
}

p = np.clip(res['p_day_ahead'].values, 1e-4, 1 - 1e-4)
logit = np.log(p / (1 - p)).reshape(-1, 1)
first_half = (pd.to_datetime(res['datetime']) < pd.Timestamp('2025-06-01')).values
held_out = np.zeros(len(res))
for fit_mask, apply_mask in ((first_half, ~first_half), (~first_half, first_half)):
    lr = LogisticRegression(C=1e6).fit(logit[fit_mask], res['y'].values[fit_mask])
    held_out[apply_mask] = lr.predict_proba(logit[apply_mask])[:, 1]
lr = LogisticRegression(C=1e6).fit(logit, res['y'].values)


def log_loss(truth, prob):
    prob = np.clip(prob, 1e-4, 1 - 1e-4)
    return float(-np.mean(truth * np.log(prob) + (1 - truth) * np.log(1 - prob)))


report['platt'] = {
    'a': round(float(lr.intercept_[0]), 3), 'b': round(float(lr.coef_[0][0]), 3),
    'held_out_log_loss': {'raw': round(log_loss(res['y'].values, p), 4),
                          'calibrated': round(log_loss(res['y'].values, held_out), 4)},
    'held_out_brier': {'raw': round(float(np.mean((p - res['y'].values) ** 2)), 4),
                       'calibrated': round(float(np.mean((held_out - res['y'].values) ** 2)), 4)},
}

print(json.dumps({k: v for k, v in report.items() if k != 'reliability'}, indent=2))
for col in ('p_same_day', 'p_day_ahead'):
    print(col, ' '.join(f"[{r['bin'][0]:.1f},{r['bin'][1]:.1f}) {r['observed']:.2f} (n={r['n']})"
                        for r in report['reliability']['all'][col]))
with open('analysis_output/rain_pop_day_ahead.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print('-> analysis_output/rain_pop_day_ahead.json')
