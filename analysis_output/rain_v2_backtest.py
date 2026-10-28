"""Rain classifier feature version 2 vs version 1, and the onset read off it.

Forward-only, the way production retrains: every 3-month block is predicted by
a model fit only on older rows (3-day embargo) and its threshold is the CSI
optimum of the out-of-fold hours before the block. Same-day rows are the
archive (short lead, all 10 members). Day-ahead rows feed the same classifier
the runs from one and two days earlier (previous_runs_data/, 8 members, 9
variables), with the production lead ramp (x0.6 threshold) and day-ahead Platt
PoP, so they mirror the live path at ~24-30 h and ~48-54 h.

The onset is scored from every hour after 3 dry station hours: the first
decided rain hour in the next 24 h against the first observed one.

    python analysis_output/rain_v2_backtest.py   (~8 min on 4 CPUs)
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('FC_DEVICE', 'cpu')
import forecast_48h_v3 as fc  # noqa: E402

PR_MODELS = ['ARPEGE_EUROPE', 'GFS_SEAMLESS', 'ICON_SEAMLESS', 'METEOFRANCE', 'ECMWF_IFS025',
             'UKMO_SEAMLESS', 'KNMI_SEAMLESS', 'DMI_SEAMLESS']
PR_VARS = ['precipitation', 'weather_code', 'cloud_cover', 'relative_humidity_2m', 'pressure_msl',
           'wind_speed_10m', 'wind_gusts_10m', 'temperature_2m', 'dew_point_2m']
EVAL_START = pd.Timestamp('2024-09-01')   # first scored block; tau needs 3 prior months
FIRST_BLOCK = pd.Timestamp('2024-06-01')
PREV_RUNS_END = pd.Timestamp('2026-03-01')
EMBARGO = pd.Timedelta(days=fc.RAIN_OCC_EMBARGO_DAYS)
# Retired discrete-time hazard onset model (onset_hazard.json, declare threshold
# 0.03), replayed on the archive 2025-07-01..2026-08-18 from every 3h-dry hour,
# next 24 h, with dry age from the replayed hourly decisions. Measured once on
# 2026-10-28 before its artifacts were removed; it never saw this period.
RETIRED_HAZARD_ONSET = {
    'cases': 6815, 'observed_onsets': 1482, 'declared_onsets': 2230,
    'event_pod': 0.829, 'event_far': 0.449, 'event_csi': 0.495,
    'median_abs_error_h': 4.0, 'hit_within_1h': 0.184, 'hit_within_2h': 0.347,
    'hit_within_3h': 0.478, 'timed_csi_3h': 0.188,
}


def rain_features_v1(frame, models=None):
    """Timing-tolerant multi-model rain features, one row per input row.

    ``frame`` needs ``datetime`` plus ``{M}_{var}_model`` columns. Features are
    computed on a gap-free hourly grid so the +/-1h windows never span a data
    gap, then mapped back to the input rows.
    """
    models = list(models or fc.MODELS)
    dt = pd.to_datetime(frame['datetime'])
    grid = pd.date_range(dt.min().floor('h'), dt.max().ceil('h'), freq='h')
    variables = (['precipitation', 'weather_code', 'wind_direction_10m']
                 + [var for var, _ in fc.RAIN_OCC_CONTEXT_VARS])
    needed = [f'{m}_{var}_model' for m in models for var in variables
              if f'{m}_{var}_model' in frame.columns]
    src = frame[needed].copy()
    src.index = dt.values
    src = src[~src.index.duplicated(keep='first')].reindex(grid)

    def _block(var):
        return pd.DataFrame({
            m: (pd.to_numeric(src[f'{m}_{var}_model'], errors='coerce')
                if f'{m}_{var}_model' in src.columns
                else pd.Series(np.nan, index=grid))
            for m in models
        }, index=grid)

    P = _block('precipitation')
    Pm1, Pp1 = P.shift(1), P.shift(-1)
    # Max over the +/-1h window; NaN only where the whole window is missing.
    W3 = pd.concat([Pm1, P, Pp1]).groupby(level=0).max().reindex(grid)
    W3 = W3.where(P.notna() | Pm1.notna() | Pp1.notna())
    S3 = (Pm1.fillna(0) + P.fillna(0) + Pp1.fillna(0)).where(P.notna())
    WC = _block('weather_code')

    out = {}
    avail = P.notna().sum(axis=1).astype(float)
    den = avail.where(avail > 0)
    avail_w3 = W3.notna().sum(axis=1).astype(float)
    den_w3 = avail_w3.where(avail_w3 > 0)
    out['n_avail'] = avail
    for thr, tag in ((0.1, '01'), (0.2, '02'), (0.5, '05'), (1.0, '1'), (3.0, '3')):
        out[f'frac_wet{tag}'] = (P >= thr).sum(axis=1) / den
        out[f'frac_w3_wet{tag}'] = (W3 >= thr).sum(axis=1) / den_w3
    out['ens_mean'] = P.mean(axis=1)
    out['ens_median'] = P.median(axis=1)
    out['ens_max'] = P.max(axis=1)
    out['ens_p75'] = P.quantile(0.75, axis=1)
    out['ens_w3max_mean'] = W3.mean(axis=1)
    out['ens_w3max_median'] = W3.median(axis=1)
    out['ens_s3_mean'] = S3.mean(axis=1)
    out['ens_mean_wet'] = P.where(P >= 0.1).mean(axis=1).fillna(0).where(avail > 0)
    ens_mean = out['ens_mean']
    ens_rev = ens_mean[::-1]
    out['ens_prev3'] = ens_mean.shift(1).rolling(3, min_periods=1).sum()
    out['ens_next3'] = ens_rev.shift(1).rolling(3, min_periods=1).sum()[::-1]
    out['ens_prev6'] = ens_mean.shift(1).rolling(6, min_periods=1).sum()
    out['ens_next6'] = ens_rev.shift(1).rolling(6, min_periods=1).sum()[::-1]

    lams = [m for m in fc.RAIN_OCC_LAMS if m in models]
    lam_avail = P[lams].notna().sum(axis=1).astype(float)
    lam_w3_avail = W3[lams].notna().sum(axis=1).astype(float)
    out['lam_n_avail'] = lam_avail
    out['lam_frac_wet01'] = (P[lams] >= 0.1).sum(axis=1) / lam_avail.where(lam_avail > 0)
    out['lam_frac_w3_wet01'] = (
        (W3[lams] >= 0.1).sum(axis=1) / lam_w3_avail.where(lam_w3_avail > 0)
    )
    out['lam_max'] = P[lams].max(axis=1)
    out['lam_w3max'] = W3[lams].max(axis=1)
    globals_ = [m for m in models if m not in fc.RAIN_OCC_LAMS]
    glob_avail = P[globals_].notna().sum(axis=1).astype(float)
    out['glob_frac_wet01'] = (
        (P[globals_] >= 0.1).sum(axis=1) / glob_avail.where(glob_avail > 0)
    )

    # WMO weather codes: drizzle/rain >= 51, showers 80-82, thunder >= 95.
    out['wc_frac_rain'] = (WC >= 51).sum(axis=1) / den
    out['wc_frac_shower'] = ((WC >= 80) & (WC <= 82)).sum(axis=1) / den
    out['wc_frac_thunder'] = (WC >= 95).sum(axis=1) / den
    out['wc_n_thunder'] = (WC >= 95).sum(axis=1).astype(float)

    for var, name in fc.RAIN_OCC_CONTEXT_VARS:
        values = _block(var)
        out[f'{name}_mean'] = values.mean(axis=1)
        if name in ('cc', 'rh', 'mslp'):
            out[f'{name}_std'] = values.std(axis=1)
    out['tdd_mean'] = out['t2m_mean'] - out['td_mean']
    out['mslp_tend3'] = out['mslp_mean'].diff(3)
    out['mslp_tend6'] = out['mslp_mean'].diff(6)
    out['mslp_tend12'] = out['mslp_mean'].diff(12)
    out['rh_tend3'] = out['rh_mean'].diff(3)
    direction = np.radians(_block('wind_direction_10m'))
    speed = _block('wind_speed_10m')
    out['wind_u'] = (-speed * np.sin(direction)).mean(axis=1)
    out['wind_v'] = (-speed * np.cos(direction)).mean(axis=1)

    hour = grid.hour.values
    doy = grid.dayofyear.values
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    for m in models:
        out[f'{m}_p0'] = P[m]
        out[f'{m}_pm1'] = Pm1[m]
        out[f'{m}_pp1'] = Pp1[m]
        out[f'{m}_w3max'] = W3[m]
        out[f'{m}_wc'] = WC[m]

    feats = pd.DataFrame(out, index=grid).replace([np.inf, -np.inf], np.nan)
    feats = feats.reindex(dt.values)
    feats.index = frame.index
    return feats


with contextlib.redirect_stdout(io.StringIO()):
    hist = fc.load_historical_data()
hist = hist.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
times = pd.to_datetime(hist['datetime'])
obs = pd.to_numeric(hist['_derived_precip_obs'], errors='coerce').values
labelled = np.isfinite(obs)
y = (obs >= fc.CORRECTED_RAIN_THRESHOLD_MM).astype(int)
runs = {m: (pd.read_csv(f'previous_runs_data/{m}_previous_runs.csv', parse_dates=['datetime'],
                        low_memory=False).drop_duplicates('datetime').set_index('datetime'))
        for m in PR_MODELS}


def member_frame(lag=None, models=None, variables=None):
    out = {'datetime': times.values}
    for m in models or fc.MODELS:
        if lag is None:
            for c in hist.columns:
                if c.startswith(f'{m}_') and c.endswith('_model'):
                    if variables is None or c[len(m) + 1:-len('_model')] in variables:
                        out[c] = pd.to_numeric(hist[c], errors='coerce').values
        else:
            for v in variables:
                src = f'{v}_{lag}'
                out[f'{m}_{v}_model'] = (
                    pd.to_numeric(runs[m][src], errors='coerce').reindex(times).values
                    if src in runs[m].columns else np.full(len(times), np.nan))
    return pd.DataFrame(out)


def blocks(start, end):
    edges = pd.date_range(start, end + pd.DateOffset(months=3), freq='3MS')
    return [(s, e) for s, e in zip(edges[:-1], edges[1:]) if s < end]


def scores(p, decision, mask):
    p, yy, d = p[mask], y[mask], decision[mask] >= 0.5
    hits, misses = int((d & (yy == 1)).sum()), int((~d & (yy == 1)).sum())
    false_alarms = int((d & (yy == 0)).sum())
    clim = yy.mean()
    brier = float(np.mean((p - yy) ** 2))
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    rainy_day = pd.Series(yy).groupby(times[mask].dt.date.values).transform('max').values == 1
    coin = rainy_day & (p >= 0.3) & (p <= 0.6)
    return {
        'hours': int(mask.sum()), 'wet_hours': int(yy.sum()),
        'log_loss': round(float(-np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc))), 4),
        'brier': round(brier, 5),
        'brier_skill': round(1 - brier / float(np.mean((clim - yy) ** 2)), 4),
        'auc': round(float(roc_auc_score(yy, p)), 4),
        'csi': round(hits / max(hits + misses + false_alarms, 1), 4),
        'pod': round(hits / max(hits + misses, 1), 4),
        'far': round(false_alarms / max(hits + false_alarms, 1), 4),
        'rainy_day_hours_pop_30_60': int(coin.sum()),
        'rainy_day_hit_rate_pop_30_60': round(float(yy[coin].mean()), 3) if coin.any() else None,
    }


def forward(features_full, features_8):
    """Out-of-fold same-day probabilities and day-ahead probabilities by lag."""
    n = len(hist)
    p_same = np.full(n, np.nan)
    p_twin = np.full(n, np.nan)
    p_lag = {lag: np.full(n, np.nan) for lag in features_8 if lag != 'archive'}
    usable_full = (features_full['n_avail'] >= fc.RAIN_OCC_MIN_MODELS).values
    cols8 = [c for c in features_8['archive'].columns
             if features_8['archive'][c].notna().mean() > 0.3
             and features_8['previous_day1'][c].notna().mean() > 0.3]
    for s, e in blocks(FIRST_BLOCK, times[labelled].max()):
        fit = labelled & (times < s - EMBARGO).values
        block = (times >= s).values & (times < e).values
        clf = fc._new_xgb_classifier(**fc.RAIN_OCC_CLF_PARAMS)
        clf.fit(features_full[fit & usable_full], y[fit & usable_full], verbose=False)
        te = block & usable_full
        p_same[te] = clf.predict_proba(features_full[te])[:, 1]
        if s < PREV_RUNS_END:
            twin_fit = fit & (features_8['archive']['n_avail'] >= 5).values
            twin = fc._new_xgb_classifier(**fc.RAIN_OCC_CLF_PARAMS)
            twin.fit(features_8['archive'].loc[twin_fit, cols8], y[twin_fit], verbose=False)
            p_twin[block] = twin.predict_proba(features_8['archive'].loc[block, cols8])[:, 1]
            for lag in p_lag:
                p_lag[lag][block] = twin.predict_proba(features_8[lag].loc[block, cols8])[:, 1]
        print(f'  block {s.date()}', flush=True)
    return p_same, p_twin, p_lag


def prequential_decision(p, tau_source, mask, factor=1.0):
    decision = np.full(len(p), np.nan)
    for s, e in blocks(EVAL_START, times[mask].max()):
        prior = labelled & np.isfinite(tau_source) & (times < s).values
        tau, _ = fc._rain_occ_best_threshold(y[prior], tau_source[prior])
        te = mask & (times >= s).values & (times < e).values
        decision[te] = (p[te] >= tau * factor).astype(float)
    return decision


frames8 = {'archive': member_frame(None, PR_MODELS, PR_VARS),
           'previous_day1': member_frame('previous_day1', PR_MODELS, PR_VARS),
           'previous_day2': member_frame('previous_day2', PR_MODELS, PR_VARS)}
full = member_frame()
versions = {
    'v1': (rain_features_v1(full), {k: rain_features_v1(f, models=PR_MODELS)
                                    for k, f in frames8.items()}),
    'v2': (fc.rain_occurrence_features(full), {k: fc.rain_occurrence_features(f, models=PR_MODELS)
                                               for k, f in frames8.items()}),
}
ok8 = np.ones(len(hist), bool)
for f in versions['v1'][1].values():
    ok8 &= (f['n_avail'] >= 6).values
same_mask = labelled & (times >= EVAL_START).values
lag_mask = labelled & ok8 & (times >= EVAL_START).values & (times < PREV_RUNS_END).values

report = {'eval_start': str(EVAL_START.date()), 'data_end': str(times[labelled].max()),
          'retired_hazard_onset_archive_2025_07': RETIRED_HAZARD_ONSET}
for version, (f_full, f_8) in versions.items():
    print(version, flush=True)
    p_same, p_twin, p_lag = forward(f_full, f_8)
    res = {}
    m = same_mask & np.isfinite(p_same)
    dec_same = prequential_decision(p_same, p_same, m)
    m &= np.isfinite(dec_same)
    res['same_day'] = scores(p_same, dec_same, m)
    res['same_day']['onset'] = fc.rain_onset_decision_scores(
        np.where(m, dec_same, np.nan), times, obs)
    for lag, lead in (('previous_day1', 27.0), ('previous_day2', 51.0)):
        p = p_lag[lag]
        mm = lag_mask & np.isfinite(p)
        dec = prequential_decision(p, p_twin, mm, factor=fc.RAIN_OCC_LONG_LEAD_FACTOR)
        mm &= np.isfinite(dec)
        pop = fc.rain_occurrence_pop(p, np.full(len(p), lead))
        res[lag] = scores(pop, dec, mm)
        res[lag]['onset'] = fc.rain_onset_decision_scores(np.where(mm, dec, np.nan), times, obs)
    # wet-hour amount (median regressor on sqrt mm), same blocks
    wet = labelled & (y == 1)
    amount = np.full(len(hist), np.nan)
    for s, e in blocks(FIRST_BLOCK, times[labelled].max()):
        fit = wet & (times < s - EMBARGO).values
        te = wet & (times >= s).values & (times < e).values & (times >= EVAL_START).values
        if not te.any():
            continue
        reg = fc._new_xgb_regressor(**fc.RAIN_OCC_AMOUNT_PARAMS)
        reg.fit(f_full[fit], np.sqrt(obs[fit]), verbose=False)
        amount[te] = np.clip(np.square(np.clip(reg.predict(f_full[te]), 0, None)),
                             fc.CORRECTED_RAIN_THRESHOLD_MM, 50.0)
    ok = np.isfinite(amount)
    res['wet_hour_amount'] = {'hours': int(ok.sum()),
                              'mae_mm': round(float(np.abs(amount[ok] - obs[ok]).mean()), 4)}
    report[version] = res

for section in ('same_day', 'previous_day1', 'previous_day2'):
    for version in ('v1', 'v2'):
        r = report[version][section]
        o = r['onset']
        print(f"{section:14s} {version} AUC={r['auc']:.4f} Brier={r['brier']:.5f} "
              f"BSS={r['brier_skill']:.3f} CSI={r['csi']:.3f} POD={r['pod']:.3f} FAR={r['far']:.3f} "
              f"| onset FAR={o['event_far']:.3f} +-3h={o['hit_within_3h']:.3f} "
              f"timed CSI={o['timed_csi_3h']:.3f}")
for version in ('v1', 'v2'):
    print(f"amount {version} MAE={report[version]['wet_hour_amount']['mae_mm']:.4f} mm")
with open('analysis_output/rain_v2_backtest.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print('-> analysis_output/rain_v2_backtest.json')
