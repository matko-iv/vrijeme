"""Shared Open-Meteo HTTP helper for the history fetch scripts.

Free tier: 600 calls/min, 5,000/h, 10,000/day, where a call's weight is about
locations x max(1, variables/10) x max(1, days/14). Each request sleeps
weight x 60/450 s afterwards (~450 weighted calls/min) and waits 65 s on 429.
"""

import time

import pandas as pd
import requests

LAT, LON = 42.2864, 18.84
TIMEZONE = "Europe/Podgorica"
MODEL_IDS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",
    "UKMO_SEAMLESS": "ukmo_seamless",
    "ECMWF_IFS": "ecmwf_ifs",
    "KNMI_SEAMLESS": "knmi_seamless",
    "DMI_SEAMLESS": "dmi_seamless",
}
UNITS = {"wind_speed_unit": "ms", "precipitation_unit": "mm",
         "temperature_unit": "celsius"}


def call_weight(n_vars, start, end):
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return max(1.0, n_vars / 10.0) * max(1.0, days / 14.0)


def get_json(url, params, weight, attempts=8, session=None):
    """GET with 429 back-off; returns the JSON dict or None."""
    http = session or requests
    for attempt in range(attempts):
        try:
            r = http.get(url, params=params, timeout=180)
        except requests.RequestException as exc:
            print(f"    mreza: {exc}; retry", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            print(f"    429 ({r.text[:100]}); cekam 65 s", flush=True)
            time.sleep(65)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code >= 400:
            print(f"    HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return None
        time.sleep(max(1.0, weight * 60.0 / 450.0))
        try:
            return r.json()
        except ValueError:
            print(f"    neispravan JSON ({len(r.content)} B); retry", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
    return None


def periods(start, end, months):
    """[(start, end)] date strings covering start..end in ``months`` chunks."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    edges = list(pd.date_range(start.replace(day=1), end, freq=f"{months}MS"))
    if not edges or edges[0] > start:
        edges = [start] + edges
    edges[0] = start
    edges.append(end + pd.Timedelta(days=1))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = b - pd.Timedelta(days=1)
        if a <= b:
            out.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return out


def hourly_frame(payload, rename):
    """Open-Meteo ``hourly`` block -> DataFrame with ``datetime`` + renamed columns."""
    h = (payload or {}).get("hourly") or {}
    if "time" not in h:
        return None
    df = pd.DataFrame({"datetime": pd.to_datetime(h["time"])})
    for src, dst in rename.items():
        if src in h:
            df[dst] = pd.to_numeric(pd.Series(h[src]), errors="coerce")
    return df


def merge_new_wins(old, new, replace_from=None, replace_cols=None):
    """Combine two hourly frames; ``new`` wins. When ``replace_from`` is given,
    ``old`` values of ``replace_cols`` at or after it are dropped first, so a
    stale value cannot survive where the new fetch has a gap."""
    # DST fall-back repeats a local hour; keep one row per timestamp.
    new = new.drop_duplicates("datetime", keep="last")
    if old is None or old.empty:
        out = new.copy()
    else:
        old = old.drop_duplicates("datetime", keep="last").copy()
        if replace_from is not None:
            cols = [c for c in (replace_cols or []) if c in old.columns]
            old.loc[old["datetime"] >= pd.Timestamp(replace_from), cols] = float("nan")
        out = (new.set_index("datetime")
               .combine_first(old.set_index("datetime"))
               .reset_index())
    out = out.drop_duplicates("datetime", keep="first").sort_values("datetime")
    return out.reset_index(drop=True)
