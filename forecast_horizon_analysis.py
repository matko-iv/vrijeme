"""
Forecast Horizon Analysis — Open-Meteo Previous Runs API

Preuzima prognoze sa različitim lead time-ovima (Day0, Day1, Day2)
za analizu degradacije tačnosti sa horizontom, i testira XGBoost poboljšanje.

Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
- temperature_2m = Day0 (najsvježija prognoza, ~0-6h lead)
- temperature_2m_previous_day1 = Day1 (prognoza od jučer, ~24-30h lead)  
- temperature_2m_previous_day2 = Day2 (prognoza od prekjučer, ~48-54h lead)
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os


LAT = 42.29
LON = 18.84
BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2024-01-15"
END_DATE = "2026-02-13"

VARIABLES = [
    "temperature_2m",
    "dew_point_2m", 
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
    "weather_code",
]

LEAD_DAYS = [0, 1, 2]

MODELS = {
    "METEOFRANCE":       "meteofrance_seamless",
    "ECMWF_IFS025":      "ecmwf_ifs025",
    "GFS_SEAMLESS":      "gfs_seamless",
    "ICON_SEAMLESS":     "icon_seamless",
    "UKMO_SEAMLESS":     "ukmo_seamless",
    "ARPEGE_EUROPE":     "meteofrance_arpege_europe",
    "BOM_ACCESS":        "bom_access_global",
}

OBS_CSV = "wu_data/merged_observations.csv"
OUTPUT_DIR = "previous_runs_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_api():
    """Testira API sa malim requestom da vidimo format odgovora."""
    print("=" * 70)
    print("FAZA 1: API TEST")
    print("=" * 70)
    
    hourly_vars = []
    for var in ["temperature_2m"]:
        hourly_vars.append(var)  
        hourly_vars.append(f"{var}_previous_day1")  
        hourly_vars.append(f"{var}_previous_day2")  
    
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2025-01-01",
        "end_date": "2025-01-07",
        "hourly": ",".join(hourly_vars),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "models": "meteofrance_seamless",
    }
    
    print(f"  URL: {BASE_URL}")
    print(f"  Params: {params}")
    
    r = requests.get(BASE_URL, params=params, timeout=60)
    print(f"  Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        if "hourly" in data:
            hourly = data["hourly"]
            print(f"  Ključevi: {list(hourly.keys())}")
            print(f"  Vremenski raspon: {hourly['time'][0]} do {hourly['time'][-1]}")
            print(f"  Broj sati: {len(hourly['time'])}")
            
            for i in range(min(5, len(hourly['time']))):
                t = hourly['time'][i]
                d0 = hourly.get('temperature_2m', [None]*(i+1))[i]
                d1 = hourly.get('temperature_2m_previous_day1', [None]*(i+1))[i]
                d2 = hourly.get('temperature_2m_previous_day2', [None]*(i+1))[i]
                print(f"    {t}: Day0={d0}°C  Day1={d1}°C  Day2={d2}°C")
            return True
        else:
            print(f"  Greška: nema 'hourly' u odgovoru")
            print(f"  Odgovor: {json.dumps(data, indent=2)[:500]}")
            return False
    else:
        print(f"  Greška: {r.text[:300]}")
        return False



def fetch_model_previous_runs(model_name, model_id):
    """Preuzima Day0, Day1, Day2 prognoze za jedan model."""
    csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_previous_runs.csv")
    
    if os.path.exists(csv_path):
        print(f"  ✅ {model_name} — već preuzeto ({csv_path})")
        return pd.read_csv(csv_path, parse_dates=['datetime'])
    
    print(f"  📥 Preuzimam {model_name}...")
    
    hourly_vars = []
    for var in VARIABLES:
        hourly_vars.append(var)  
        hourly_vars.append(f"{var}_previous_day1")  
        hourly_vars.append(f"{var}_previous_day2")  
    
    all_dfs = []
    
    periods = []
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    
    current = start
    while current < end:
        period_end = min(current + pd.DateOffset(months=3), end)
        periods.append((current.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d")))
        current = period_end + pd.Timedelta(days=1)
    
    for i, (sd, ed) in enumerate(periods):
        print(f"    Period {i+1}/{len(periods)}: {sd} do {ed}...")
        
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": sd,
            "end_date": ed,
            "hourly": ",".join(hourly_vars),
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "models": model_id,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.get(BASE_URL, params=params, timeout=120)
                
                if r.status_code == 429:
                    wait = 65
                    print(f"    ⚠️ Rate limit — čekam {wait}s...")
                    time.sleep(wait)
                    continue
                
                if r.status_code != 200:
                    print(f"    ❌ Status {r.status_code}: {r.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(30)
                        continue
                    break
                
                data = r.json()
                
                if "hourly" not in data:
                    print(f"    ❌ Nema 'hourly': {str(data)[:200]}")
                    break
                
                hourly = data["hourly"]
                df_data = {"datetime": pd.to_datetime(hourly["time"])}
                
                for var in hourly_vars:
                    if var in hourly:
                        df_data[var] = hourly[var]
                
                df_period = pd.DataFrame(df_data)
                all_dfs.append(df_period)
                print(f"    ✅ {len(df_period)} redova")
                
                time.sleep(2)
                break
                
            except Exception as e:
                print(f"    ❌ Greška: {e}")
                if attempt < max_retries - 1:
                    time.sleep(30)
    
    if not all_dfs:
        print(f"  ❌ Nema podataka za {model_name}")
        return None
    
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"  ✅ {model_name}: {len(df)} redova → {csv_path}")
    
    return df



def analyze_degradation(obs_df):
    """Analizira kako MAE raste sa lead time-om za svaki model."""
    print()
    print("=" * 70)
    print("FAZA 3: FORECAST HORIZON DEGRADACIJA")
    print("=" * 70)
    
    key_vars = ["temperature_2m", "dew_point_2m", "wind_speed_10m", "pressure_msl", "cloud_cover", "precipitation"]
    obs_mapping = {
        "temperature_2m": "temp_c",
        "dew_point_2m": "dewpoint_c",
        "wind_speed_10m": "wind_ms",
        "pressure_msl": "pressure_hpa",
        "cloud_cover": None,  # nema obs
        "precipitation": "precip_rate_mm",
    }
    
    all_results = []
    
    for model_name in MODELS:
        csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_previous_runs.csv")
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path, parse_dates=['datetime'])
        merged = df.merge(obs_df, on='datetime', how='inner')
        
        print(f"\n  {model_name} ({len(merged)} sati sa opservacijama):")
        print(f"  {'Varijabla':20s} {'Day0 MAE':>10s} {'Day1 MAE':>10s} {'Day2 MAE':>10s} {'Degradacija':>12s}")
        print(f"  {'-'*60}")
        
        for var in key_vars:
            obs_col = obs_mapping.get(var)
            if obs_col is None or obs_col not in merged.columns:
                continue
            
            maes = {}
            for day in LEAD_DAYS:
                if day == 0:
                    col = var
                else:
                    col = f"{var}_previous_day{day}"
                
                if col not in merged.columns:
                    continue
                
                mask = merged[col].notna() & merged[obs_col].notna()
                if mask.sum() < 100:
                    continue
                
                err = np.abs(merged.loc[mask, col] - merged.loc[mask, obs_col])
                maes[day] = err.mean()
            
            if 0 in maes and len(maes) > 1:
                max_day = max(maes.keys())
                degradation = (maes[max_day] - maes[0]) / maes[0] * 100
                
                d0 = f"{maes.get(0, float('nan')):.3f}"
                d1 = f"{maes.get(1, float('nan')):.3f}" if 1 in maes else "—"
                d2 = f"{maes.get(2, float('nan')):.3f}" if 2 in maes else "—"
                
                print(f"  {var:20s} {d0:>10s} {d1:>10s} {d2:>10s} {degradation:>+10.1f}%")
                
                all_results.append({
                    'model': model_name,
                    'variable': var,
                    'mae_day0': round(maes.get(0, None), 4) if 0 in maes else None,
                    'mae_day1': round(maes.get(1, None), 4) if 1 in maes else None,
                    'mae_day2': round(maes.get(2, None), 4) if 2 in maes else None,
                    'degradation_pct': round(degradation, 1)
                })
    
    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n  {'='*60}")
        print(f"  AGREGIRANO (prosjek svih modela):")
        print(f"  {'Varijabla':20s} {'Day0':>10s} {'Day1':>10s} {'Day2':>10s} {'Degradacija':>12s}")
        print(f"  {'-'*60}")
        
        for var in key_vars:
            sub = rdf[rdf['variable'] == var]
            if sub.empty:
                continue
            d0 = sub['mae_day0'].mean()
            d1 = sub['mae_day1'].mean()
            d2 = sub['mae_day2'].mean()
            deg = (d2 - d0) / d0 * 100 if d0 > 0 and not np.isnan(d2) else 0
            print(f"  {var:20s} {d0:10.3f} {d1:10.3f} {d2:10.3f} {deg:>+10.1f}%")
    
    return all_results



def test_xgboost_improvement(obs_df):
    """Testira da li dodavanje Day1/Day2 prognoza poboljšava XGBoost."""
    print()
    print("=" * 70)
    print("FAZA 4: XGBOOST POBOLJŠANJE SA PREVIOUS RUNS")
    print("=" * 70)
    
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        print("  ❌ xgboost ili sklearn nisu instalirani")
        return None
    
    
    target_var = "temperature_2m"
    obs_col = "temp_c"
    
    model_dfs = {}
    for model_name in MODELS:
        csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_previous_runs.csv")
        if os.path.exists(csv_path):
            model_dfs[model_name] = pd.read_csv(csv_path, parse_dates=['datetime'])
    
    if not model_dfs:
        print("  ❌ Nema podataka za XGBoost test")
        return None
    
    base = obs_df[['datetime', obs_col]].copy()
    
    for model_name, mdf in model_dfs.items():
        suffix = model_name.lower()
        for col in mdf.columns:
            if col == 'datetime':
                continue
            mdf = mdf.rename(columns={col: f"{col}_{suffix}"})
        base = base.merge(mdf, on='datetime', how='inner')
    
    print(f"  Merged dataset: {len(base)} redova, {len(base.columns)} kolona")
    
    base = base.dropna(subset=[obs_col])
    
    base['hour'] = base['datetime'].dt.hour
    base['month'] = base['datetime'].dt.month
    base['day_of_year'] = base['datetime'].dt.dayofyear
    base['hour_sin'] = np.sin(2 * np.pi * base['hour'] / 24)
    base['hour_cos'] = np.cos(2 * np.pi * base['hour'] / 24)
    base['month_sin'] = np.sin(2 * np.pi * base['month'] / 12)
    base['month_cos'] = np.cos(2 * np.pi * base['month'] / 12)
    
    baseline_cols = [c for c in base.columns if target_var in c and 'previous' not in c and obs_col not in c]
    baseline_cols += ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    
    enhanced_cols = [c for c in base.columns if target_var in c and obs_col not in c]
    enhanced_cols += ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    
    for model_name in model_dfs:
        suffix = model_name.lower()
        d0_col = f"{target_var}_{suffix}"
        d1_col = f"{target_var}_previous_day1_{suffix}"
        d2_col = f"{target_var}_previous_day2_{suffix}"
        
        if d0_col in base.columns and d1_col in base.columns:
            diff_col = f"diff_d0d1_{suffix}"
            base[diff_col] = base[d0_col] - base[d1_col]
            enhanced_cols.append(diff_col)
        
        if d1_col in base.columns and d2_col in base.columns:
            diff_col = f"diff_d1d2_{suffix}"
            base[diff_col] = base[d1_col] - base[d2_col]
            enhanced_cols.append(diff_col)
    
    d0_temp_cols = [c for c in base.columns if target_var in c and 'previous' not in c and obs_col not in c and 'diff' not in c]
    if len(d0_temp_cols) > 1:
        base['ensemble_mean_d0'] = base[d0_temp_cols].mean(axis=1)
        base['ensemble_std_d0'] = base[d0_temp_cols].std(axis=1)
        baseline_cols += ['ensemble_mean_d0', 'ensemble_std_d0']
        enhanced_cols += ['ensemble_mean_d0', 'ensemble_std_d0']
    
    d1_temp_cols = [c for c in base.columns if f"{target_var}_previous_day1" in c and 'diff' not in c]
    if len(d1_temp_cols) > 1:
        base['ensemble_mean_d1'] = base[d1_temp_cols].mean(axis=1)
        base['ensemble_std_d1'] = base[d1_temp_cols].std(axis=1)
        base['ensemble_change_d0d1'] = base.get('ensemble_mean_d0', 0) - base['ensemble_mean_d1']
        enhanced_cols += ['ensemble_mean_d1', 'ensemble_std_d1', 'ensemble_change_d0d1']
    
    baseline_cols = [c for c in baseline_cols if c in base.columns]
    enhanced_cols = [c for c in enhanced_cols if c in base.columns]
    
    baseline_cols = list(dict.fromkeys(baseline_cols))
    enhanced_cols = list(dict.fromkeys(enhanced_cols))
    
    print(f"  Baseline features: {len(baseline_cols)}")
    print(f"  Enhanced features: {len(enhanced_cols)} (+{len(enhanced_cols)-len(baseline_cols)} novih)")
    
    SPLIT_DATE = pd.Timestamp('2025-07-01')
    
    train = base[base['datetime'] < SPLIT_DATE].copy()
    test = base[base['datetime'] >= SPLIT_DATE].copy()
    
    print(f"  Train: {len(train)} redova ({train['datetime'].min()} do {train['datetime'].max()})")
    print(f"  Test:  {len(test)} redova ({test['datetime'].min()} do {test['datetime'].max()})")
    
    if len(train) < 100 or len(test) < 50:
        print("  ❌ Premalo podataka za test")
        return None
    
    X_train_base = train[baseline_cols].fillna(0)
    X_test_base = test[baseline_cols].fillna(0)
    y_train = train[obs_col]
    y_test = test[obs_col]
    
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'random_state': 42,
    }
    
    print(f"\n  Training BASELINE model (Day0 only)...")
    model_base = xgb.XGBRegressor(**xgb_params)
    model_base.fit(
        X_train_base, y_train,
        eval_set=[(X_test_base, y_test)],
        verbose=False
    )
    pred_base = model_base.predict(X_test_base)
    mae_base = mean_absolute_error(y_test, pred_base)
    bias_base = (pred_base - y_test.values).mean()
    
    X_train_enh = train[enhanced_cols].fillna(0)
    X_test_enh = test[enhanced_cols].fillna(0)
    
    print(f"  Training ENHANCED model (Day0 + Day1 + Day2)...")
    model_enh = xgb.XGBRegressor(**xgb_params)
    model_enh.fit(
        X_train_enh, y_train,
        eval_set=[(X_test_enh, y_test)],
        verbose=False
    )
    pred_enh = model_enh.predict(X_test_enh)
    mae_enh = mean_absolute_error(y_test, pred_enh)
    bias_enh = (pred_enh - y_test.values).mean()
    
    ensemble_col = 'ensemble_mean_d0' if 'ensemble_mean_d0' in test.columns else d0_temp_cols[0]
    mae_ensemble = mean_absolute_error(y_test, test[ensemble_col].fillna(y_test.mean()))
    
    print(f"\n  {'='*55}")
    print(f"  REZULTATI (temperatura, test set {test['datetime'].min().date()} do {test['datetime'].max().date()}):")
    print(f"  {'='*55}")
    print(f"  {'Metoda':35s} {'MAE':>8s} {'Bias':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Ensemble Mean (Day0)':35s} {mae_ensemble:8.3f}°C")
    print(f"  {'XGBoost BASELINE (Day0 only)':35s} {mae_base:8.3f}°C {bias_base:+8.4f}")
    print(f"  {'XGBoost ENHANCED (Day0+Day1+Day2)':35s} {mae_enh:8.3f}°C {bias_enh:+8.4f}")
    
    improvement = (mae_base - mae_enh) / mae_base * 100
    print(f"\n  Poboljšanje: {improvement:+.1f}% ({mae_base:.3f} → {mae_enh:.3f}°C)")
    
    if improvement > 0:
        print(f"  ✅ DA — Previous runs POBOLJŠAVAJU prognozu za {improvement:.1f}%!")
    else:
        print(f"  ❌ NE — Previous runs ne pomažu (ili blago pogoršavaju)")
    
    fi = pd.Series(model_enh.feature_importances_, index=enhanced_cols).sort_values(ascending=False)
    print(f"\n  Top 15 features (enhanced model):")
    for feat, imp in fi.head(15).items():
        marker = " ← NOVO" if 'previous' in feat or 'diff' in feat or 'd1' in feat.lower() else ""
        print(f"    {feat:45s} {imp:.4f}{marker}")
    
    print(f"\n  {'='*55}")
    print(f"  TEST OSTALIH VARIJABLI:")
    print(f"  {'='*55}")
    
    other_vars = {
        "dew_point_2m": "dewpoint_c",
        "wind_speed_10m": "wind_ms",
        "pressure_msl": "pressure_hpa",
    }
    
    results_all = {
        "temperature_2m": {
            "mae_baseline": round(float(mae_base), 4),
            "mae_enhanced": round(float(mae_enh), 4),
            "improvement_pct": round(float(improvement), 1),
            "n_test": len(test),
        }
    }
    
    for var, obs_c in other_vars.items():
        if obs_c not in base.columns:
            continue
        
        b_cols = [c for c in base.columns if var in c and 'previous' not in c and obs_c not in c and 'diff' not in c and 'ensemble' not in c]
        e_cols = [c for c in base.columns if var in c and obs_c not in c and 'ensemble' not in c]
        
        b_cols += ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        e_cols += ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        
        for mn in model_dfs:
            s = mn.lower()
            d0c = f"{var}_{s}"
            d1c = f"{var}_previous_day1_{s}"
            dc = f"diff_{var}_d0d1_{s}"
            if d0c in base.columns and d1c in base.columns:
                if dc not in base.columns:
                    base[dc] = base[d0c] - base[d1c]
                e_cols.append(dc)
        
        b_cols = list(dict.fromkeys([c for c in b_cols if c in base.columns]))
        e_cols = list(dict.fromkeys([c for c in e_cols if c in base.columns]))
        
        train_v = base[base['datetime'] < SPLIT_DATE]
        test_v = base[base['datetime'] >= SPLIT_DATE]
        
        y_tr = train_v[obs_c].dropna()
        common_idx_tr = y_tr.index
        y_te = test_v[obs_c].dropna()
        common_idx_te = y_te.index
        
        if len(common_idx_tr) < 100 or len(common_idx_te) < 50:
            continue
        
        Xb_tr = train_v.loc[common_idx_tr, b_cols].fillna(0)
        Xb_te = test_v.loc[common_idx_te, b_cols].fillna(0)
        Xe_tr = train_v.loc[common_idx_tr, e_cols].fillna(0)
        Xe_te = test_v.loc[common_idx_te, e_cols].fillna(0)
        
        mb = xgb.XGBRegressor(**xgb_params)
        mb.fit(Xb_tr, y_tr, eval_set=[(Xb_te, y_te)], verbose=False)
        mae_b = mean_absolute_error(y_te, mb.predict(Xb_te))
        
        me = xgb.XGBRegressor(**xgb_params)
        me.fit(Xe_tr, y_tr, eval_set=[(Xe_te, y_te)], verbose=False)
        mae_e = mean_absolute_error(y_te, me.predict(Xe_te))
        
        imp = (mae_b - mae_e) / mae_b * 100
        print(f"  {var:25s}  Baseline={mae_b:.3f}  Enhanced={mae_e:.3f}  Δ={imp:+.1f}%")
        
        results_all[var] = {
            "mae_baseline": round(float(mae_b), 4),
            "mae_enhanced": round(float(mae_e), 4),
            "improvement_pct": round(float(imp), 1),
            "n_test": len(common_idx_te),
        }
    
    return results_all



if __name__ == "__main__":
    api_ok = test_api()
    if not api_ok:
        print("\n❌ API test neuspješan. Provjerite vezu i pokušajte ponovo.")
        exit(1)
    
    print("\n📊 Učitavam opservacije...")
    obs = pd.read_csv(OBS_CSV, parse_dates=['datetime'])
    obs = obs[obs['datetime'] >= START_DATE]
    print(f"  {len(obs)} opservacija od {START_DATE}")
    
    print()
    print("=" * 70)
    print("FAZA 2: PREUZIMANJE PREVIOUS RUNS PODATAKA")
    print("=" * 70)
    
    for model_name, model_id in MODELS.items():
        fetch_model_previous_runs(model_name, model_id)
        time.sleep(1)  # pauza između modela
    
    degradation_results = analyze_degradation(obs)
    
    xgb_results = test_xgboost_improvement(obs)
    
    all_results = {
        "degradation": degradation_results,
        "xgboost_improvement": xgb_results,
        "period": f"{START_DATE} to {END_DATE}",
        "generated": pd.Timestamp.now().isoformat(),
    }
    
    with open("forecast_horizon_analysis.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else str(o))
    
    print(f"\n✅ Svi rezultati sačuvani u forecast_horizon_analysis.json")
    print("DONE!")
