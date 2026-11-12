"""Helper tasks the pipeline GUI runs as child processes.

    python pipeline_tasks.py prereq      check/repair the bundled runtime, disk, network, GPU
    python pipeline_tasks.py export      copy the trained model to <drive>/models/<stamp>/

All paths derive from this file's location: <root>/app/gui/pipeline_tasks.py.
"""
import datetime as _dt
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.environ.get("MLFP_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REPO = os.path.join(ROOT, "app", "repo")
WHEELS = os.path.join(ROOT, "runtime", "wheels")
MODELS_OUT = os.path.join(ROOT, "models")
LOGS = os.path.join(ROOT, "logs")
REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "catboost",
            "optuna", "joblib", "requests", "PIL", "tkinter"]
MIN_FREE_GB = 5


def _missing_modules():
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
    return missing


def _pip(args):
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check", *args]
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def prereq():
    print(f"Python {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"Pipeline folder: {ROOT}", flush=True)
    missing = _missing_modules()
    if missing:
        print("Missing packages: " + "; ".join(missing), flush=True)
        lock = os.path.join(WHEELS, "requirements.lock")
        if os.path.exists(lock):
            print("Installing from the offline wheel folder on the drive...", flush=True)
            _pip(["install", "--no-index", "--find-links", WHEELS, "-r", lock])
        missing = _missing_modules()
        if missing:
            print("Trying PyPI...", flush=True)
            _pip(["install", "-r", os.path.join(REPO, "requirements.txt"), "pillow"])
            missing = _missing_modules()
        if missing:
            print("ERROR: packages still missing: " + "; ".join(missing), flush=True)
            return 2
    print("Packages: all importable", flush=True)
    import numpy, pandas, xgboost, lightgbm, catboost, optuna, sklearn  # noqa: E401
    print(f"  numpy {numpy.__version__}, pandas {pandas.__version__}, xgboost {xgboost.__version__}, "
          f"lightgbm {lightgbm.__version__}, catboost {catboost.__version__}, "
          f"optuna {optuna.__version__}, scikit-learn {sklearn.__version__}", flush=True)

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"Free space on the drive: {free_gb:.1f} GB", flush=True)
    if free_gb < MIN_FREE_GB:
        print(f"ERROR: need at least {MIN_FREE_GB} GB free on the drive", flush=True)
        return 3

    for url in ("https://api.open-meteo.com/v1/forecast?latitude=42.29&longitude=18.84&hourly=temperature_2m&forecast_days=1",
                "https://www.wunderground.com/"):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                code = r.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        except Exception as exc:  # noqa: BLE001
            code = f"no connection ({exc.__class__.__name__})"
        print(f"Network {url.split('/')[2]}: {code}", flush=True)

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=20).stdout.strip()
            print(f"GPU: {out or 'nvidia-smi found, no GPU listed'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"GPU: nvidia-smi failed ({exc})", flush=True)
    else:
        print("GPU: no NVIDIA driver found; training runs on the CPU", flush=True)
    return 0


def _git_commit():
    path = os.path.join(REPO, "BUILD_INFO.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def export():
    src = os.path.join(REPO, "trained_models_v2")
    if not os.path.isdir(src):
        print(f"ERROR: no trained model folder at {src}", flush=True)
        return 2
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    dst = os.path.join(MODELS_OUT, stamp)
    os.makedirs(dst, exist_ok=True)
    skip = {"api_cache", "_champion", "__pycache__"}
    shutil.copytree(src, os.path.join(dst, "trained_models_v2"),
                    ignore=lambda d, names: [n for n in names if n in skip])
    extra = {
        "rain_v3_backtest.json": os.path.join(REPO, "analysis_output", "rain_v3_backtest.json"),
        "forecast_48h.json": os.path.join(REPO, "forecast_output", "forecast_48h.json"),
    }
    for name, path in extra.items():
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(dst, name))
    run_log = os.environ.get("MLFP_RUN_LOG")
    if run_log and os.path.exists(run_log):
        shutil.copy2(run_log, os.path.join(dst, "run.log"))

    def _load(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    results = _load(os.path.join(src, "training_results.json")) or {}
    rain = _load(os.path.join(src, "rain_occurrence_meta.json")) or {}
    timings = _load(os.path.join(src, "timings.json")) or []
    manifest = {
        "exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "machine": os.environ.get("COMPUTERNAME", ""),
        "device_requested": os.environ.get("FC_DEVICE", "auto"),
        "optuna_trials": os.environ.get("FC_TRIALS"),
        "tests_passed": os.environ.get("MLFP_TESTS_PASSED"),
        "build": _git_commit(),
        "rain_classifier": {k: rain.get(k) for k in (
            "feature_set", "tau", "oof_csi", "oof_brier", "oof_band_30_60_hours",
            "oof_band_30_60_hit_rate", "data_start", "data_end")},
        "rain_day_ahead": rain.get("day_ahead"),
        "metrics": {k: {m: v.get(m) for m in ("mae", "rmse", "ensemble_mae", "method",
                                              "mae_lead27", "mae_lead51",
                                              "quantile_cov90", "quantile_crps")}
                    for k, v in results.items() if isinstance(v, dict)},
        "timings": timings,
    }
    with open(os.path.join(dst, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    size = sum(os.path.getsize(os.path.join(d, n)) for d, _, ns in os.walk(dst) for n in ns)
    print(f"Saved model: {dst} ({size / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"prereq": prereq, "export": export}.get(task, lambda: 64)())
