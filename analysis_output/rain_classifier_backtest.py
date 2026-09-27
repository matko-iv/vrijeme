"""Walk-forward backtest: multi-model rain classifier vs raw NWP models.

Every season is predicted by a classifier trained only on older data (3-day
embargo), with its threshold picked out-of-fold on the two preceding years --
the same protocol as train_rain_occurrence_model. All rules are scored against
true hourly station rain (>= 0.2 mm over the Open-Meteo interval [t-1, t]).
Each baseline is compared with the classifier on the hours where that baseline
has data, so ICON-2I (archive from 2025-04) is judged on its own period.

    python analysis_output/rain_classifier_backtest.py   (~10 min on 4 CPUs)
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('FC_DEVICE', 'cpu')
import forecast_48h_v3 as fc  # noqa: E402

FIRST_WINDOW = pd.Timestamp('2023-06-01')
WINDOW_MONTHS = 3

with contextlib.redirect_stdout(io.StringIO()):
    hist = fc.load_historical_data()
cols = ['datetime', '_derived_precip_obs'] + [c for c in hist.columns if c.endswith('_model')]
frame = hist[cols].sort_values('datetime').reset_index(drop=True)
feats = fc.rain_occurrence_features(frame)
obs = pd.to_numeric(frame['_derived_precip_obs'], errors='coerce')
labelled = obs.notna().values
y = (obs >= fc.CORRECTED_RAIN_THRESHOLD_MM).astype(int).values
times = pd.to_datetime(frame['datetime'])
baselines = fc.rain_occurrence_baselines(frame)
data_end = times[labelled].max()

rules = ['rain_classifier'] + list(baselines)
decided = {r: np.full(len(frame), np.nan) for r in rules}
windows = []
start = FIRST_WINDOW
while start < data_end:
    end = start + pd.DateOffset(months=WINDOW_MONTHS)
    train_end = start - pd.Timedelta(days=fc.RAIN_OCC_EMBARGO_DAYS)
    oof = fc.rain_occurrence_walk_forward_oof(feats, y, times, labelled, train_end)
    ok = np.isfinite(oof)
    tau, _ = fc._rain_occ_best_threshold(y[ok], oof[ok])
    fit = labelled & (times < train_end).values
    test = labelled & (times >= start).values & (times < end).values
    clf = fc._new_xgb_classifier(**fc.RAIN_OCC_CLF_PARAMS)
    clf.fit(feats[fit], y[fit], verbose=False)
    proba = clf.predict_proba(feats[test])[:, 1]
    decided['rain_classifier'][test] = (proba >= tau).astype(float)
    for r in baselines:
        decided[r][test] = baselines[r][test]
    card = fc._rain_occ_scorecard(
        y[test], {r: decided[r][test] for r in rules
                  if np.isfinite(decided[r][test]).mean() > 0.9})
    windows.append({'start': str(start.date()), 'end': str(end.date()), 'tau': tau,
                    'wet_hours': int(y[test].sum()), 'scorecard': card})
    best_raw = max((r for r in card if r in fc.MODELS), key=lambda r: card[r]['csi'])
    print(f"{start.date()}..{end.date()} tau={tau:.2f} wet={int(y[test].sum()):4d}  "
          f"classifier CSI={card['rain_classifier']['csi']:.3f}  "
          f"best raw {best_raw}={card[best_raw]['csi']:.3f}  "
          f"ensemble={card['ensemble_mean']['csi']:.3f}"
          + (f"  icon2i_gate={card['icon2i_gate']['csi']:.3f}" if 'icon2i_gate' in card else ''))
    start = end

# Paired pooled comparison: each rule vs the classifier on that rule's hours.
pooled = {}
clf_dec = decided['rain_classifier']
for r in baselines:
    both = np.isfinite(decided[r]) & np.isfinite(clf_dec)
    card = fc._rain_occ_scorecard(y[both], {'rain_classifier': clf_dec[both], r: decided[r][both]})
    pooled[r] = {'hours': int(both.sum()), 'wet_hours': int(y[both].sum()),
                 'baseline': card[r], 'rain_classifier': card['rain_classifier']}
print("\nPooled, each rule vs the classifier on the same hours (CSI):")
for r, row in sorted(pooled.items(), key=lambda kv: -kv[1]['baseline']['csi']):
    print(f"  {r:20s} {row['baseline']['csi']:.3f}  vs classifier "
          f"{row['rain_classifier']['csi']:.3f}  ({row['hours']} h, {row['wet_hours']} wet)")

with open('analysis_output/rain_classifier_backtest.json', 'w', encoding='utf-8') as f:
    json.dump({'data_end': str(data_end), 'windows': windows, 'pooled': pooled},
              f, indent=2)
print("-> analysis_output/rain_classifier_backtest.json")
