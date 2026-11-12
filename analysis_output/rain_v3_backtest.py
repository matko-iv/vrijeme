"""Rain classifier feature sets: v2 vs the v3 upper-air / convective sets.

Same protocol as rain_v2_backtest.py: forward-only, every 3-month block is
predicted by a model fit only on older rows (3-day embargo), and its threshold
is the CSI optimum of the out-of-fold hours before the block. Same-day archive
rows, station labels from the gauge accumulation (load_historical_data).

Sets (forecast_48h_v3.RAIN_OCC_FEATURE_SETS):
  v2        surface multi-model features
  v3-gfs    + GFS upper-air / convective block
  v3-multi  + cross-model CAPE/CIN/showers/lightning/TCWV/precip-probability
  v3-both   + both

Gate: ship the v3 set with the lowest same-day Brier that beats v2's Brier
and whose onset timed CSI (+/-3 h) is not below v2's; otherwise v2. The
choice is written to rain_v3_backtest.json ('chosen'), which
train_rain_occurrence_model() reads.

    python analysis_output/rain_v3_backtest.py
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import forecast_48h_v3 as fc  # noqa: E402

EVAL_START = pd.Timestamp('2024-09-01')   # first scored block; tau needs 3 prior months
FIRST_BLOCK = pd.Timestamp('2024-06-01')
EMBARGO = pd.Timedelta(days=fc.RAIN_OCC_EMBARGO_DAYS)
OUT_PATH = os.path.join(ROOT, 'analysis_output', 'rain_v3_backtest.json')


def progress(step, total, label):
    if os.environ.get('FC_PROGRESS') == '1':
        print('##PROGRESS ' + json.dumps({'stage': 'backtest', 'step': step,
                                          'total': total, 'label': label}), flush=True)


with contextlib.redirect_stdout(io.StringIO()):
    hist = fc.load_historical_data()
hist = hist.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
times = pd.to_datetime(hist['datetime'])
obs = pd.to_numeric(hist['_derived_precip_obs'], errors='coerce').values
labelled = np.isfinite(obs)
y = (obs >= fc.CORRECTED_RAIN_THRESHOLD_MM).astype(int)
full = hist[['datetime'] + [c for c in hist.columns if c.endswith('_model')]]


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
    band = (p >= 0.3) & (p <= 0.6)
    rainy_day = pd.Series(yy).groupby(times[mask].dt.date.values).transform('max').values == 1
    return {
        'hours': int(mask.sum()), 'wet_hours': int(yy.sum()),
        'log_loss': round(float(-np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc))), 4),
        'brier': round(brier, 5),
        'brier_skill': round(1 - brier / float(np.mean((clim - yy) ** 2)), 4),
        'auc': round(float(roc_auc_score(yy, p)), 4),
        'csi': round(hits / max(hits + misses + false_alarms, 1), 4),
        'pod': round(hits / max(hits + misses, 1), 4),
        'far': round(false_alarms / max(hits + false_alarms, 1), 4),
        'band_30_60_hours': int(band.sum()),
        'band_30_60_hit_rate': round(float(yy[band].mean()), 3) if band.any() else None,
        'band_30_60_rainy_day_hours': int((band & rainy_day).sum()),
        'wet_hours_in_band_frac': round(float(yy[band].sum() / max(yy.sum(), 1)), 3),
    }


def forward(feats):
    usable = (feats['n_avail'] >= fc.RAIN_OCC_MIN_MODELS).values
    p = np.full(len(hist), np.nan)
    for s, e in blocks(FIRST_BLOCK, times[labelled].max()):
        fit = labelled & usable & (times < s - EMBARGO).values
        te = usable & (times >= s).values & (times < e).values
        if fit.sum() < 5000 or not te.any():
            continue
        clf = fc._new_xgb_classifier(**fc.RAIN_OCC_CLF_PARAMS)
        clf.fit(feats[fit], y[fit], verbose=False)
        p[te] = clf.predict_proba(feats[te])[:, 1]
    return p


def prequential_decision(p, mask):
    decision = np.full(len(p), np.nan)
    for s, e in blocks(EVAL_START, times[mask].max()):
        prior = labelled & np.isfinite(p) & (times < s).values
        if prior.sum() < 500:
            continue
        tau, _ = fc._rain_occ_best_threshold(y[prior], p[prior])
        te = mask & (times >= s).values & (times < e).values
        decision[te] = (p[te] >= tau).astype(float)
    return decision


def stability_coverage():
    cov = {}
    for m in fc.RAIN_OCC_STAB_MODELS:
        c = f'{m}_cape_model'
        cov[m] = round(float(hist[c].notna().mean()), 3) if c in hist.columns else 0.0
    return cov


def main():
    t0 = time.time()
    cov = stability_coverage()
    gfs_ok = cov.get(fc.RAIN_OCC_GFS_MODEL, 0.0) >= 0.2
    multi_ok = sum(v >= 0.2 for v in cov.values()) >= 3
    variants = ['v2']
    if gfs_ok:
        variants.append('v3-gfs')
    if multi_ok:
        variants.append('v3-multi')
    if gfs_ok and multi_ok:
        variants.append('v3-both')
    report = {'eval_start': str(EVAL_START.date()), 'data_end': str(times[labelled].max()),
              'label': 'gauge accumulation >= 0.2 mm/h', 'stability_coverage': cov,
              'variants': {}}
    same_mask = labelled & (times >= EVAL_START).values
    for i, variant in enumerate(variants, 1):
        progress(i, len(variants), variant)
        tv = time.time()
        feats = fc.rain_occurrence_features(full, feature_set=variant)
        p = forward(feats)
        m = same_mask & np.isfinite(p)
        dec = prequential_decision(p, m)
        m &= np.isfinite(dec)
        res = scores(p, dec, m)
        res['features'] = int(feats.shape[1])
        res['onset'] = fc.rain_onset_decision_scores(np.where(m, dec, np.nan), times, obs)
        res['seconds'] = round(time.time() - tv, 1)
        report['variants'][variant] = res
        o = res['onset']
        print(f"{variant:9s} Brier={res['brier']:.5f} AUC={res['auc']:.4f} CSI={res['csi']:.3f} "
              f"FAR={res['far']:.3f} 30-60%={res['band_30_60_hours']}h "
              f"(kisa {res['band_30_60_hit_rate']}) | onset FAR={o['event_far']:.3f} "
              f"+-3h={o['hit_within_3h']} timedCSI={o['timed_csi_3h']:.3f} "
              f"[{res['seconds']:.0f}s]", flush=True)

    base = report['variants']['v2']
    chosen, rule = 'v2', 'no v3 set beat v2 on Brier with onset timed CSI >= v2'
    best = None
    for v, r in report['variants'].items():
        if v == 'v2':
            continue
        if (r['brier'] < base['brier']
                and r['onset']['timed_csi_3h'] >= base['onset']['timed_csi_3h']
                and (best is None or r['brier'] < report['variants'][best]['brier'])):
            best = v
    if best:
        chosen = best
        rule = ('lowest same-day Brier among v3 sets that beat v2 and keep onset '
                'timed CSI >= v2')
    report['chosen'] = chosen
    report['rule'] = rule
    report['seconds'] = round(time.time() - t0, 1)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"CHOSEN: {chosen} ({rule})")
    print(f"-> {OUT_PATH}")


if __name__ == '__main__':
    main()
