"""Stability / convective / upper-air history per model.

Writes budva_{MODEL}_stability.csv with ``{MODEL}_{var}_model`` columns for
STABILITY_VARS (CAPE, CIN, showers, lightning, precipitation probability,
850/700/500 hPa wind, humidity, temperature, geopotential, omega), wind in
m/s, from the historical-forecast API. load_historical_data() merges these
files; the live fetch requests the same variables in HOURLY_VARS.

Incremental from each file's last hour (the last day is refetched).

    python fetch_aux_history.py [--seed-dir DIR] [--models A,B]

``--seed-dir`` imports ``{MODEL}_stability.csv`` files with plain variable
names (same API, same units) before fetching, to save API quota.
"""

import argparse
import os
import sys

import pandas as pd

from openmeteo_client import (LAT, LON, MODEL_IDS, TIMEZONE, UNITS, call_weight,
                              get_json, hourly_frame, merge_new_wins, periods)

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# Must equal forecast_48h_v3.STABILITY_VARS (checked in test_forecast_core).
STABILITY_VARS = (
    'cape', 'convective_inhibition', 'lifted_index', 'showers', 'freezing_level_height',
    'wind_speed_850hPa', 'wind_speed_700hPa', 'wind_speed_500hPa',
    'relative_humidity_700hPa', 'relative_humidity_850hPa',
    'temperature_850hPa', 'temperature_500hPa', 'geopotential_height_500hPa',
    'vertical_velocity_700hPa',
    'wind_direction_850hPa', 'wind_direction_700hPa',
    'lightning_potential', 'total_column_integrated_water_vapour',
    'precipitation_probability',
)
STAB_MODELS = ("GFS_SEAMLESS", "ICON_SEAMLESS", "ECMWF_IFS025", "ITALIAMETEO_ICON2I",
               "KNMI_SEAMLESS", "DMI_SEAMLESS", "UKMO_SEAMLESS", "METEOFRANCE")
START = "2021-01-01"
CHUNK_MONTHS = 12


def out_path(name):
    return os.path.join(HERE, f"budva_{name}_stability.csv")


def rename_map(name):
    return {v: f"{name}_{v}_model" for v in STABILITY_VARS}


def read_existing(name):
    path = out_path(name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["datetime"], low_memory=False)


def write(name, df):
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["datetime"] + [c for c in rename_map(name).values() if c in out.columns]
    out[cols].to_csv(out_path(name), index=False)


def seed(name, seed_dir):
    src = os.path.join(seed_dir, f"{name}_stability.csv")
    if not os.path.exists(src):
        return
    d = pd.read_csv(src, parse_dates=["datetime"], low_memory=False)
    d = d.rename(columns=rename_map(name))
    keep = ["datetime"] + [c for c in rename_map(name).values() if c in d.columns]
    merged = merge_new_wins(read_existing(name), d[keep])
    write(name, merged)
    print(f"{name}: seed {len(d)} redova iz {src}", flush=True)


def fetch(name, start, end):
    parts = []
    for a, b in periods(start, end, CHUNK_MONTHS):
        params = {"latitude": LAT, "longitude": LON, "start_date": a, "end_date": b,
                  "hourly": ",".join(STABILITY_VARS), "models": MODEL_IDS[name],
                  "timezone": TIMEZONE, **UNITS}
        payload = get_json(URL, params, call_weight(len(STABILITY_VARS), a, b))
        df = hourly_frame(payload, rename_map(name))
        print(f"  {name} {a}..{b}: {'ok' if df is not None else 'FAIL'}", flush=True)
        if df is None:
            return None
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir")
    ap.add_argument("--models", default=",".join(STAB_MODELS))
    ap.add_argument("--start", default=START)
    args = ap.parse_args()
    end = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).strftime("%Y-%m-%d")
    failed = []
    for name in args.models.split(","):
        if args.seed_dir:
            seed(name, args.seed_dir)
        old = read_existing(name)
        if old is not None and len(old):
            valid = old.dropna(subset=[c for c in old.columns if c != "datetime"], how="all")
            last = valid["datetime"].max() if len(valid) else pd.Timestamp(args.start)
            start = last.strftime("%Y-%m-%d")
        else:
            start = args.start
        if start > end:
            print(f"{name}: azurno", flush=True)
            continue
        print(f"{name}: {start} -> {end}", flush=True)
        new = fetch(name, start, end)
        if new is None:
            failed.append(name)
            continue
        merged = merge_new_wins(old, new)
        write(name, merged)
        cov = merged.drop(columns="datetime").notna().mean().round(2)
        print(f"{name}: {len(merged)} redova do {merged['datetime'].max()}, "
              f"pokrivenost cape={cov.get(f'{name}_cape_model', 0)}, "
              f"pp={cov.get(f'{name}_precipitation_probability_model', 0)}", flush=True)
    if failed:
        print(f"NEUSPJELO: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
