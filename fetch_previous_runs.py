"""Previous-runs history (runs from 1 and 2 days earlier) per model.

Writes previous_runs_data/{MODEL}_previous_runs.csv with columns
``{var}_previous_day1`` / ``{var}_previous_day2`` for PREV_RUNS_VARS, wind in
m/s. Incremental: a state file records how far each model is complete; the
first run (no state) rebuilds every model from its start date, because the
older files mixed km/h wind into the day-1/day-2 columns and carry future
long-lead values in their tail.

For a valid time after the fetch moment the API cannot have a run from 24 or
48 h earlier and returns the latest run instead, so rows later than
fetch time - 1 h are dropped.

    python fetch_previous_runs.py [--full]
"""

import argparse
import json
import os
import sys

import pandas as pd

from openmeteo_client import (LAT, LON, MODEL_IDS, TIMEZONE, UNITS, call_weight,
                              get_json, hourly_frame, merge_new_wins, periods)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "previous_runs_data")
STATE_PATH = os.path.join(OUT_DIR, "_fetch_state.json")
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAGS = ("previous_day1", "previous_day2")
# Must equal forecast_48h_v3.PREV_RUNS_VARS (checked in test_forecast_core).
PREV_RUNS_VARS = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_gusts_10m", "pressure_msl",
    "cloud_cover", "precipitation",
    "weather_code", "wind_direction_10m", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "shortwave_radiation",
]
DEFAULT_START = "2024-01-15"
MODEL_START = {"ITALIAMETEO_ICON2I": "2025-05-01", "ECMWF_IFS": "2025-09-01"}
CHUNK_MONTHS = 6
OVERLAP_DAYS = 3


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def fetch_model(name, start, end, cutoff):
    cols = [f"{v}_{lag}" for v in PREV_RUNS_VARS for lag in LAGS]
    parts = []
    for a, b in periods(start, end, CHUNK_MONTHS):
        params = {"latitude": LAT, "longitude": LON, "start_date": a, "end_date": b,
                  "hourly": ",".join(cols), "models": MODEL_IDS[name],
                  "timezone": TIMEZONE, **UNITS}
        payload = get_json(URL, params, call_weight(len(cols), a, b))
        df = hourly_frame(payload, {c: c for c in cols})
        print(f"  {name} {a}..{b}: {'ok' if df is not None else 'FAIL'}", flush=True)
        if df is None:
            return None, cols
        parts.append(df)
    new = pd.concat(parts, ignore_index=True)
    new = new[new["datetime"] <= cutoff]
    return new, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignore state, rebuild all")
    ap.add_argument("--models", default=",".join(MODEL_IDS))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    state = {} if args.full else load_state()
    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
    cutoff = now - pd.Timedelta(hours=1)
    end = now.strftime("%Y-%m-%d")
    failed = []
    for name in args.models.split(","):
        path = os.path.join(OUT_DIR, f"{name}_previous_runs.csv")
        done = state.get(name, {}).get("complete_until")
        if done:
            start = (pd.Timestamp(done) - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
            replace_from = pd.Timestamp(start)
        else:
            start = MODEL_START.get(name, DEFAULT_START)
            replace_from = pd.Timestamp("1900-01-01")
        print(f"{name}: {start} -> {end} (cutoff {cutoff})", flush=True)
        new, cols = fetch_model(name, start, end, cutoff)
        if new is None or new.empty:
            failed.append(name)
            continue
        old = None
        if os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["datetime"], low_memory=False)
        merged = merge_new_wins(old, new, replace_from=replace_from, replace_cols=cols)
        # rows after the cutoff (left from old files) are long-lead values
        tail = merged["datetime"] > cutoff
        merged.loc[tail, [c for c in cols if c in merged.columns]] = float("nan")
        out = merged.copy()
        out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(path, index=False)
        valid = new.dropna(subset=[c for c in cols if c in new.columns], how="all")
        state[name] = {"complete_until": str(valid["datetime"].max() if len(valid) else cutoff),
                       "fetched_at": now.isoformat(), "wind_unit": "m/s"}
        save_state(state)
        n_d1 = int(new.get("precipitation_previous_day1", pd.Series(dtype=float)).notna().sum())
        print(f"{name}: {len(merged)} redova, day1 precip {n_d1} novih sati", flush=True)
    if failed:
        print(f"NEUSPJELO: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
