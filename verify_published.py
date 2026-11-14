"""Score the forecasts the site published against station IBUDVA5.

    python verify_published.py --model v6
    python verify_published.py --since 2026-11-13 --until 2026-12-13

Reads every docs/forecast_data/forecast_48h.json committed to git in the
window, pairs each hourly value with the station hour it describes, and
writes MAE per target and lead window, the MAE of the raw model blend on the
same hours, and rain-hour scores (>= 0.2 mm). Each valid hour counts once per
lead window: the forecasts made for it are averaged first, so days with many
runs do not outweigh days with few.

Station data comes from wu_data/merged_observations.csv. Refresh it first
with `python wu_update.py`. Cloud cover is not scored: the station has no
cloud sensor.

Hours are local time labelled by their start, as on the site and in the
station table: rain at 13:00 is rain from 13:00 to 14:00.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLISHED = 'docs/forecast_data/forecast_48h.json'
OBS_PATH = os.path.join(ROOT, 'wu_data', 'merged_observations.csv')
OUT_DIR = os.path.join(ROOT, 'analysis_output')

# Releases from before the forecast JSON carried "model_version".
# A snapshot without the field gets the last release made before it.
RELEASES = [
    ('v6', pd.Timestamp('2026-08-19 08:05:56')),
    ('v7', pd.Timestamp('2026-11-12 21:53:42')),
]

# published field -> station column
TARGETS = {
    'temperature_2m': 'temp_c',
    'dew_point_2m': 'dewpoint_c',
    'relative_humidity_2m': 'humidity_pct',
    'wind_speed_10m': 'wind_ms',
    'wind_gusts_10m': 'gust_ms',
    'pressure_msl': 'pressure_hpa',
    'shortwave_radiation': 'solar_wm2',
    'precipitation': 'precip_mm',
}
RAIN_MM = 0.2
LEADS = [('0-23h', 0, 23), ('24-47h', 24, 47)]


def git(*args, data=None):
    out = subprocess.run(['git', *args], cwd=ROOT, input=data,
                         capture_output=True, check=True)
    return out.stdout


def read_snapshots(since, until):
    """Every published forecast JSON committed in [since - 1 day, until]."""
    shas = git('log', '--format=%H',
               f'--since={(since - pd.Timedelta(days=1)).isoformat()}',
               f'--until={(until + pd.Timedelta(days=1)).isoformat()}', '--', PUBLISHED).decode().split()
    if not shas:
        return []
    blob = git('cat-file', '--batch',
               data=''.join(f'{s}:{PUBLISHED}\n' for s in shas).encode())
    snaps, pos = [], 0
    for sha in shas:
        end = blob.index(b'\n', pos)
        header = blob[pos:end].split()
        pos = end + 1
        if header[-1] == b'missing':
            continue
        size = int(header[2])
        try:
            snaps.append((sha, json.loads(blob[pos:pos + size].decode('utf-8'))))
        except ValueError:
            pass
        pos += size + 1
    return snaps


def release_at(t):
    name = None
    for version, start in RELEASES:
        if t >= start:
            name = version
    return name


def forecast_rows(snaps, since, until, model):
    rows, used, metrics = [], [], None
    for sha, d in snaps:
        gen = pd.Timestamp(d.get('generated', '')[:19])
        if pd.isna(gen) or not since <= gen < until:
            continue
        version = d.get('model_version') or release_at(gen)
        if model and version != model:
            continue
        used.append(gen)
        if metrics is None or gen >= max(used):
            metrics = d.get('training_metrics') or {}
        issued = gen.floor('h')
        for h in d.get('hourly_forecast', []):
            valid = pd.Timestamp(h['datetime'])
            row = {'valid': valid,
                   'lead': int((valid - issued) / pd.Timedelta(hours=1)),
                   'pop': h.get('precipitation_pop')}
            for t in TARGETS:
                row[t] = h.get(t)
                row[t + '_raw'] = h.get(t + '_raw')
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ('valid', 'lead'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, sorted(used), metrics or {}


def per_hour(df, cols):
    """Average the forecasts made for each valid hour, then over hours."""
    return df.groupby('valid')[cols].mean().mean()


def score(fc, obs):
    df = fc.merge(obs, left_on='valid', right_index=True, how='inner')
    out = {}
    for name, lo, hi in LEADS:
        part = df[(df['lead'] >= lo) & (df['lead'] <= hi)]
        res = {}
        for t, col in TARGETS.items():
            if t == 'precipitation':
                continue
            p = part[part[t].notna() & part[col].notna()].copy()
            if p.empty:
                continue
            p['err'] = p[t] - p[col]
            p['abs'] = p['err'].abs()
            p['raw_abs'] = (p[t + '_raw'] - p[col]).abs()
            m = per_hour(p, ['abs', 'err', 'raw_abs'])
            res[t] = {'hours': int(p['valid'].nunique()),
                      'mae': round(m['abs'], 3),
                      'bias': round(m['err'], 3),
                      'raw_mae': round(m['raw_abs'], 3),
                      'gain_pct': round(100 * (1 - m['abs'] / m['raw_abs']), 1)}
        res['rain'] = rain_scores(part)
        out[name] = res
    return out


def rain_scores(part):
    p = part[part['precipitation'].notna() & part['precip_mm'].notna()].copy()
    if p.empty:
        return {}
    p['obs'] = (p['precip_mm'] >= RAIN_MM).astype(float)
    p['fc'] = (p['precipitation'] >= RAIN_MM).astype(float)
    p['raw'] = (p['precipitation_raw'] >= RAIN_MM).astype(float)
    p['brier'] = (p['pop'] - p['obs']) ** 2
    g = p.groupby('valid').mean(numeric_only=True)

    def table(f):
        hits = (g[f] * g['obs']).sum()
        false_alarms = (g[f] * (1 - g['obs'])).sum()
        misses = ((1 - g[f]) * g['obs']).sum()
        return {'csi': round(hits / max(hits + false_alarms + misses, 1e-9), 3),
                'pod': round(hits / max(hits + misses, 1e-9), 3),
                'far': round(false_alarms / max(hits + false_alarms, 1e-9), 3)}

    return {'hours': int(len(g)), 'rain_hours': int(g['obs'].sum()),
            **table('fc'), 'raw': table('raw'),
            'pop_brier': round(float(g['brier'].mean()), 4)}


def by_month(fc, obs):
    df = fc[fc['lead'] <= 23].merge(obs, left_on='valid', right_index=True)
    df['month'] = df['valid'].dt.strftime('%Y-%m')
    out = {}
    for month, part in df.groupby('month'):
        out[month] = {}
        for t, col in TARGETS.items():
            if t == 'precipitation':
                continue
            p = part[part[t].notna() & part[col].notna()].copy()
            if not p.empty:
                p['abs'] = (p[t] - p[col]).abs()
                out[month][t] = round(per_hour(p, ['abs'])['abs'], 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', help='score one model version, e.g. v6')
    ap.add_argument('--since', help='first issue time (local), e.g. 2026-11-13')
    ap.add_argument('--until', help='end of issue times (local), exclusive')
    ap.add_argument('--out', help='JSON report path')
    a = ap.parse_args()

    since = pd.Timestamp(a.since) if a.since else None
    until = pd.Timestamp(a.until) if a.until else None
    if a.model:
        starts = dict(RELEASES)
        if since is None and a.model in starts:
            since = starts[a.model]
        later = [s for _, s in RELEASES if since is not None and s > since]
        if until is None and later:
            until = later[0]
    since = since or pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    until = until or pd.Timestamp.now()

    obs = pd.read_csv(OBS_PATH, parse_dates=['datetime'], low_memory=False)
    obs = obs.set_index('datetime')[list(TARGETS.values())]
    fc, issued, metrics = forecast_rows(read_snapshots(since, until), since, until, a.model)
    if fc.empty:
        sys.exit(f'No published forecasts from {since} to {until}.')
    fc = fc[(fc['lead'] >= 0) & (fc['lead'] <= LEADS[-1][2])]

    report = {
        'model': a.model,
        'issued_from': str(issued[0]), 'issued_to': str(issued[-1]),
        'forecasts': len(issued),
        'station_data_end': str(obs.index.max()),
        'scores': score(fc, obs),
        'mae_by_month_0_23h': by_month(fc, obs),
        'test_mae': {t: (metrics.get(t) or {}).get('mae') for t in TARGETS},
    }
    label = a.model or f"{since:%Y%m%d}_{until:%Y%m%d}"
    out = a.out or os.path.join(OUT_DIR, f'live_verification_{label}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)

    print(f"{a.model or 'Published'}: {report['forecasts']} forecasts issued "
          f"{issued[0]:%d %b %H:%M} - {issued[-1]:%d %b %Y %H:%M}; "
          f"station data to {obs.index.max():%d %b %H:%M}")
    print(f"{'target':22s}{'lead':>8s}{'hours':>7s}{'MAE':>8s}{'raw':>8s}"
          f"{'gain':>7s}{'bias':>8s}{'test':>8s}")
    for t in TARGETS:
        if t == 'precipitation':
            continue
        for name, _, _ in LEADS:
            r = report['scores'][name].get(t)
            if r:
                test = report['test_mae'][t]
                print(f"{t:22s}{name:>8s}{r['hours']:7d}{r['mae']:8.3f}"
                      f"{r['raw_mae']:8.3f}{r['gain_pct']:6.1f}%{r['bias']:+8.3f}"
                      f"{test if test is not None else float('nan'):8.3f}")
    for name, _, _ in LEADS:
        r = report['scores'][name]['rain']
        if r:
            print(f"rain >= {RAIN_MM} mm {name:>7s}: {r['rain_hours']} of {r['hours']} hours wet; "
                  f"CSI {r['csi']:.3f} (raw {r['raw']['csi']:.3f}), POD {r['pod']:.2f}, "
                  f"FAR {r['far']:.2f}, PoP Brier {r['pop_brier']:.4f}")
    print(f"Written {os.path.relpath(out, ROOT)}")


if __name__ == '__main__':
    main()
