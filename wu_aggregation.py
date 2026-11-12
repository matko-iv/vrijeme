"""Aggregacija sirovih 5-min WU opservacija u satne redove."""

import numpy as np
import pandas as pd

DIR_DEG = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'East': 90, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'South': 180, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'West': 270, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
    'North': 0,
}

HOURLY_COLUMNS = [
    'datetime', 'temp_c', 'dewpoint_c', 'humidity_pct', 'pressure_hpa',
    'wind_ms', 'precip_rate_mm', 'solar_wm2', 'uv', 'gust_ms', 'gust_ms_p95',
    'wind_ms_max', 'wind_ms_p95', 'precip_rate_max', 'solar_wm2_max',
    'precip_accum_mm', 'wind_dir_deg', 'precip_mm',
]

_MEAN_COLS = ('temp_c', 'dewpoint_c', 'humidity_pct', 'pressure_hpa',
              'wind_ms', 'precip_rate_mm', 'solar_wm2', 'uv')


def hourly_precip_from_accum(accum_max, rate_mean=None):
    """Rain that fell in each hour (mm), from the gauge's daily counter.

    ``accum_max`` is the hourly max of ``precip_accum_mm`` indexed by hour
    start. The counter resets at local midnight, so hour 00 is its own max;
    every other hour is its max minus the previous hour's max. When the
    previous hour is missing or the counter dropped (a mid-day reset), the
    hour falls back to ``rate_mean`` (the mean 5-min rate) when given, else
    NaN. The mean of instantaneous rates over-counts the gauge total by
    about 27%, which is why it is only the fallback.
    """
    acc = pd.to_numeric(accum_max, errors='coerce')
    idx = pd.DatetimeIndex(acc.index)
    cur = acc.to_numpy(dtype=float)
    prev = acc.reindex(idx - pd.Timedelta(hours=1)).to_numpy(dtype=float)
    midnight = idx.hour == 0
    diff = cur - prev
    out = np.where(midnight, cur, diff)
    bad = (~midnight) & (~np.isfinite(diff) | (diff < -0.01))
    out = np.where(bad, np.nan, out)
    if rate_mean is not None:
        fb = pd.to_numeric(rate_mean, errors='coerce').reindex(idx).to_numpy(dtype=float)
        out = np.where(np.isfinite(out), out, fb)
    out = np.clip(out, 0.0, None)
    return pd.Series(np.round(out, 3), index=acc.index)


def resample_to_hourly(df):
    """5-min redovi -> satni redovi (label = pocetak sata)."""
    if df is None or df.empty:
        return None

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    for c in (*_MEAN_COLS, 'gust_ms', 'precip_accum_mm'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    hour = df['datetime'].dt.floor('h').rename('hour')
    g = df.groupby(hour, sort=True)

    out = pd.DataFrame(index=g.size().index)
    for c in _MEAN_COLS:
        out[c] = g[c].mean()
    out['gust_ms'] = g['gust_ms'].max()
    out['gust_ms_p95'] = g['gust_ms'].quantile(0.95)
    out['wind_ms_max'] = g['wind_ms'].max()
    out['wind_ms_p95'] = g['wind_ms'].quantile(0.95)
    out['precip_rate_max'] = g['precip_rate_mm'].max()
    out['solar_wm2_max'] = g['solar_wm2'].max()
    out['precip_accum_mm'] = g['precip_accum_mm'].max()

    # Speed-weighted vector mean of the compass direction.
    deg = df['wind_dir'].map(DIR_DEG)
    spd = df['wind_ms']
    ok = deg.notna() & spd.notna()
    rad = np.radians(deg.where(ok).to_numpy(dtype=float))
    w = spd.where(ok).to_numpy(dtype=float)
    parts = pd.DataFrame({
        'ws': w * np.sin(rad), 'wc': w * np.cos(rad),
        's': np.sin(rad), 'c': np.cos(rad), 'w': w, 'n': ok.astype(int),
    }, index=df.index).groupby(hour).sum(min_count=1)
    calm = parts['w'].fillna(0) == 0
    sin_sum = parts['ws'].where(~calm, parts['s'])
    cos_sum = parts['wc'].where(~calm, parts['c'])
    ang = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    out['wind_dir_deg'] = ang.where(parts['n'] > 0)

    out['precip_mm'] = hourly_precip_from_accum(out['precip_accum_mm'],
                                                out['precip_rate_mm'])

    out.index.name = 'datetime'
    out = out.reset_index()
    out['datetime'] = out['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return out[HOURLY_COLUMNS]
