"""
XGBoost +48h forecast correction for Budva.
Historical bias tables + multi-model ensemble + XGBoost per parameter.
Run: .venv/Scripts/python.exe forecast_48h_v3.py [--skip-training]
Author: Matija Ivanović (@matko-iv)
"""

import sys, io, os, json, time, warnings, subprocess
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# Do not globally suppress warnings: CUDA/OpenCL fallback and artifact-version
# warnings are operationally important. Individual noisy call sites can filter a
# specific warning locally if one proves harmless.

import pandas as pd
import numpy as np
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV


def meteorological_metrics(y_true, y_pred_binary, p_proba=None):
    """contingency-table metrics for precipitation classification.
    Returns dict with POD (recall), FAR (1-precision), CSI, HSS, SEDI plus
    Brier score and reliability-diagram RMSE if p_proba is provided.
    Use these instead of F1 for precision-first rain detection."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred_binary).astype(int)
    a = int(((y_true == 1) & (y_pred == 1)).sum())  # hits
    b = int(((y_true == 0) & (y_pred == 1)).sum())  # false alarms
    c = int(((y_true == 1) & (y_pred == 0)).sum())  # misses
    d = int(((y_true == 0) & (y_pred == 0)).sum())  # correct rejections
    n = a + b + c + d
    pod = a / (a + c) if (a + c) > 0 else 0.0  # = recall
    far = b / (a + b) if (a + b) > 0 else 0.0  # false-alarm ratio
    precision = a / (a + b) if (a + b) > 0 else 0.0
    csi = a / (a + b + c) if (a + b + c) > 0 else 0.0
    if (a + c) > 0 and (c + d) > 0 and (a + b) > 0 and (b + d) > 0:
        hss_num = 2.0 * (a * d - b * c)
        hss_den = (a + c) * (c + d) + (a + b) * (b + d)
        hss = hss_num / hss_den if hss_den > 0 else 0.0
    else:
        hss = 0.0
    # SEDI: stable for rare events (Ferro & Stephenson 2011)
    H_raw = pod
    F_raw = b / (b + d) if (b + d) > 0 else 0.0
    if (a + c) > 0 and (b + d) > 0:
        # SEDI's logarithms are undefined at exact 0/1. Clipping preserves
        # the correct limiting behavior (perfect -> +1, inverse -> -1).
        eps = 1e-6
        H = float(np.clip(H_raw, eps, 1 - eps))
        F = float(np.clip(F_raw, eps, 1 - eps))
        sedi = (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / \
               (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))
    else:
        sedi = 0.0
    out = {"hits": a, "false_alarms": b, "misses": c, "correct_rejections": d,
           "pod": float(pod), "far": float(far), "precision": float(precision),
           "csi": float(csi), "hss": float(hss), "sedi": float(sedi), "n": n}
    if p_proba is not None:
        p = np.clip(np.asarray(p_proba, dtype=float), 0.0, 1.0)
        # Brier score = MSE between predicted P(rain) and binary outcome
        out["brier"] = float(np.mean((p - y_true) ** 2))
        # Brier skill score vs climatology (base rate predictor)
        base_rate = float(y_true.mean()) if len(y_true) else 0.0
        brier_clim = float(np.mean((base_rate - y_true) ** 2)) if len(y_true) else 1.0
        out["brier_skill_score"] = float(1 - out["brier"] / brier_clim) if brier_clim > 0 else 0.0
        # Reliability-diagram RMSE: bin predictions, compare to observed frequency
        # target < 0.05 annually, < 0.07 in Jun-Sep
        bins = np.linspace(0.0, 1.0, 11)  # 10 bins of width 0.1
        rel_sq_err, total_weight = 0.0, 0
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
            n_bin = int(in_bin.sum())
            if n_bin == 0:
                continue
            obs_freq = float(y_true[in_bin].mean())
            pred_mean = float(p[in_bin].mean())
            rel_sq_err += n_bin * (pred_mean - obs_freq) ** 2
            total_weight += n_bin
        out["reliability_rmse"] = float(np.sqrt(rel_sq_err / total_weight)) if total_weight > 0 else 0.0
    return out


def crps_ensemble(y_true, members):
    """CRPS (continuous ranked probability score) from an ensemble/sample.

    Report A3 / Gneiting & Raftery 2007: CRPS is the strictly-proper
    generalisation of MAE to predictive distributions. Energy-form estimator:
        CRPS = (1/m) Σ|x_i - y| - (1/(2 m^2)) ΣΣ|x_i - x_j|
    with the sorted-sample identity ΣΣ|x_i-x_j| = 2 Σ_i (2i - m - 1) x_(i).
    NaN members are dropped per row. Returns a per-sample array (mean it yourself).
    """
    y = np.asarray(y_true, dtype=float)
    M = np.asarray(members, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    n = len(y)
    out = np.full(n, np.nan)
    for k in range(n):
        row = M[k][np.isfinite(M[k])]
        if row.size == 0 or not np.isfinite(y[k]):
            continue
        m = row.size
        t1 = np.abs(row - y[k]).mean()
        rs = np.sort(row)
        idx = np.arange(1, m + 1)
        t2 = (1.0 / (m * m)) * np.sum((2 * idx - m - 1) * rs)
        out[k] = t1 - t2
    return out


def crps_score(y_true, members):
    """Mean CRPS over all valid samples (lower is better)."""
    vals = crps_ensemble(y_true, members)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')


def pit_values(y_true, members):
    """Probability Integral Transform values (report A3): the continuous analog
    of the rank histogram. PIT = fraction of ensemble members <= obs, in (0,1).
    Flat histogram = calibrated; U-shaped = under-dispersed (overconfident);
    dome = over-dispersed. NaN members dropped per row."""
    y = np.asarray(y_true, dtype=float)
    M = np.asarray(members, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    n = len(y)
    out = np.full(n, np.nan)
    for k in range(n):
        row = M[k][np.isfinite(M[k])]
        if row.size == 0 or not np.isfinite(y[k]):
            continue
        # Deterministic mid-rank PIT. The +0.5 continuity correction avoids
        # impossible zero/one boundary values for finite ensembles.
        less = np.sum(row < y[k])
        equal = np.sum(row == y[k])
        out[k] = (less + 0.5 * equal + 0.5) / (row.size + 1.0)
    return out


def pit_histogram(pit, bins=10):
    """Return (counts, flatness_rmse). flatness_rmse=0 => perfectly calibrated."""
    p = np.asarray(pit, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return [], float('nan')
    counts, _ = np.histogram(p, bins=bins, range=(0.0, 1.0))
    freq = counts / counts.sum()
    uniform = 1.0 / bins
    flatness = float(np.sqrt(np.mean((freq - uniform) ** 2)))
    return counts.tolist(), flatness


def onset_timing_metrics(pred_hours, obs_hours, tolerances=(1, 2, 3)):
    """Event-based onset-timing verification (report B4). Avoids the hourly
    POD/FAR double-penalty. pred_hours / obs_hours are per-case onset LEAD hours
    (np.nan if no onset in the window).
      mae_hours / median_hours : |pred - obs| over cases where BOTH have onset
      hit_within_Nh            : fraction of those within +/-N hours
      onset_pod / onset_far    : event detection (did we call onset at all)
    """
    pred = np.asarray(pred_hours, dtype=float)
    obs = np.asarray(obs_hours, dtype=float)
    has_p, has_o = np.isfinite(pred), np.isfinite(obs)
    both = has_p & has_o
    err = np.abs(pred[both] - obs[both])
    res = {
        'n_cases': int(len(pred)),
        'n_both': int(both.sum()),
        'mae_hours': float(err.mean()) if err.size else float('nan'),
        'median_hours': float(np.median(err)) if err.size else float('nan'),
    }
    for nN in tolerances:
        res[f'hit_within_{nN}h'] = float((err <= nN).mean()) if err.size else float('nan')
    res['onset_pod'] = float(both.sum() / max(int(has_o.sum()), 1))
    res['onset_far'] = float((has_p & ~has_o).sum() / max(int(has_p.sum()), 1))
    return res


def temporal_fss(pred_wet, obs_wet, window=3):
    """Temporal Fractions Skill Score (report B4; Roberts & Lean 2008 in the
    time dimension). pred_wet/obs_wet: binary wet-hour sequences. Compares
    fractional wet coverage in +/-window neighbourhoods. FSS>0.5 ~ 'useful'."""
    p = np.asarray(pred_wet, dtype=float)
    o = np.asarray(obs_wet, dtype=float)
    if len(p) == 0 or len(p) != len(o):
        return float('nan')
    kernel = np.ones(2 * window + 1)
    norm = (2 * window + 1)
    pf = np.convolve(p, kernel, mode='same') / norm
    of = np.convolve(o, kernel, mode='same') / norm
    num = np.mean((pf - of) ** 2)
    den = np.mean(pf ** 2) + np.mean(of ** 2)
    return float(1.0 - num / den) if den > 0 else float('nan')


def threshold_for_precision_at_recall(y_true, p_proba, min_recall=0.5,
                                       fallback_thresh=0.5):
    """precision-first threshold tuning.
    Pick threshold that MAXIMIZES precision subject to recall >= min_recall.
    Fallback to argmax precision if no feasible point."""
    prec, rec, thr = precision_recall_curve(y_true, p_proba)
    if len(thr) == 0:
        return float(fallback_thresh)
    prec_arr, rec_arr = prec[:-1], rec[:-1]
    feasible = rec_arr >= min_recall
    if feasible.any():
        idxs = np.where(feasible)[0]
        best = idxs[np.argmax(prec_arr[idxs])]
        return float(thr[best])
    return float(thr[np.argmax(prec_arr)])


def focal_loss_xgb_objective(gamma: float = 2.0, alpha: float = 0.25):
    """focal loss as XGBoost custom objective for binary classification.

    Wang, Deng & Wang (2020), "Imbalance-XGBoost: leveraging weighted and focal
    losses for binary label-imbalanced classification with XGBoost"
    (Pattern Recognition Letters 136:190-197).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = sigmoid(margin) if y=1 else 1 - sigmoid(margin).

    gamma down-weights easy examples; alpha balances the rare positive class.
    Recommended starting ranges for hourly rain classification:
      gamma in [1.0, 3.0], alpha in [0.2, 0.5].

    Returns a callable suitable for xgb.train's `obj` parameter and for
    XGBClassifier's `objective` argument (modern XGBoost >= 1.6).
    """
    def fobj(preds, dtrain):
        # preds are raw margins (logits) when objective is callable
        try:
            y = dtrain.get_label()
        except Exception:
            y = np.asarray(dtrain)
        y = y.astype(float)
        # Numerically stable sigmoid
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        # pt = probability of the true class; at = class-balance weight
        pt = np.where(y == 1, p, 1 - p)
        at = np.where(y == 1, alpha, 1 - alpha)
        sgn = np.where(y == 1, 1.0, -1.0)
        log_pt = np.log(pt)
        one_minus_pt_g = np.power(1 - pt, gamma)
        # First derivative of focal loss wrt the raw margin
        # dL/dz = at * (1-pt)^gamma * (gamma*pt*log(pt) + pt - 1) * sgn
        grad = at * one_minus_pt_g * (gamma * pt * log_pt + pt - 1) * sgn
        # Second derivative wrt raw margin (correct full derivation; NOT the
        # the simplified textbook formula, which can go negative for small pt and
        # destabilizes XGBoost's Newton step).
        # d^2L/dz^2 = at * pt * (1-pt)^gamma * [ gamma*log(pt)*(1 - pt*(gamma+1))
        #                                       - pt*(2*gamma + 1) + 2*gamma + 1 ]
        bracket = gamma * log_pt * (1 - pt * (gamma + 1)) - pt * (2 * gamma + 1) + 2 * gamma + 1
        hess = at * pt * one_minus_pt_g * bracket
        # Focal loss is non-convex in margin space; hessian can be negative for
        # very hard misclassified examples (e.g. y=1 with p << 0.5). XGBoost's
        # Newton step -grad/hess would blow up. Clip negative curvature to a
        # small positive floor; taking abs() would invent strong curvature and
        # produce a different optimization problem.
        hess = np.maximum(hess, 1e-3)
        # xgb.train does not apply DMatrix weights to a custom objective. Honor
        # the seasonal rain weights explicitly in both first and second order.
        try:
            sample_weight = np.asarray(dtrain.get_weight(), dtype=float)
        except Exception:
            sample_weight = np.array([], dtype=float)
        if sample_weight.size == y.size:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess
    return fobj


def focal_loss_xgb_feval(gamma: float = 2.0, alpha: float = 0.25):
    """Evaluation metric matching focal_loss_xgb_objective so early-stopping
    sees a meaningful, monotone signal. Returns (name, value). XGBoost minimizes
    by default with custom feval."""
    def feval(preds, dmatrix):
        try:
            y = dmatrix.get_label()
        except Exception:
            y = np.asarray(dmatrix)
        y = y.astype(float)
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        pt = np.where(y == 1, p, 1 - p)
        at = np.where(y == 1, alpha, 1 - alpha)
        loss = -at * np.power(1 - pt, gamma) * np.log(pt)
        try:
            sample_weight = np.asarray(dmatrix.get_weight(), dtype=float)
        except Exception:
            sample_weight = np.array([], dtype=float)
        value = (np.average(loss, weights=sample_weight)
                 if sample_weight.size == y.size and sample_weight.sum() > 0
                 else np.mean(loss))
        return 'focal_loss', float(value)
    return feval


def seasonal_sample_weights(months, y, summer_months=(6, 7, 8, 9)):
    """asymmetric sample weights conditioned on month.

    Relax positives in summer (we already over-forecast), penalize negatives in
    summer more (false alarms hurt user trust most). Push positives in winter
    (real frontal rain is what we mainly miss).

      Summer (Jun-Sep): rain pos -> 0.7, dry neg -> 1.5
      Other      (W):   rain pos -> 1.4, dry neg -> 1.0
    """
    months = np.asarray(months)
    y = np.asarray(y).astype(int)
    is_summer = np.isin(months, list(summer_months))
    w = np.where(
        y == 1,
        np.where(is_summer, 0.7, 1.4),
        np.where(is_summer, 1.5, 1.0),
    ).astype(float)
    return w
import catboost as cb
import lightgbm as lgb
import requests
import optuna
import prob_forecast as pf
optuna.logging.set_verbosity(optuna.logging.WARNING)


RUN_AUX_DIAGNOSTICS = (
    '--aux-diagnostics' in sys.argv or
    os.environ.get('FC_AUX_DIAGNOSTICS', '').strip().lower()
    in {'1', 'true', 'yes', 'on'}
)


class _AuxDiagnosticsDisabled(Exception):
    """Internal control flow for optional non-production base learners."""


# ---------------------------------------------------------------------------
# ML compute device
# ---------------------------------------------------------------------------
# XGBoost >= 2 uses ``device='cuda'`` together with ``tree_method='hist'``.
# Keep device selection in one place: this file creates models in many separate
# training paths, and a missed constructor would otherwise silently run on CPU.
#
# Selection (highest precedence first):
#   --gpu / --cpu
#   FC_DEVICE=cuda|cpu|auto (default: auto)
#   FC_GPU_ID=0             (default: first GPU)
if '--gpu' in sys.argv and '--cpu' in sys.argv:
    raise ValueError("--gpu i --cpu se ne mogu koristiti istovremeno.")

_DEVICE_REQUEST = (
    'cuda' if '--gpu' in sys.argv else
    'cpu' if '--cpu' in sys.argv else
    os.environ.get('FC_DEVICE', 'auto').strip().lower()
)
if _DEVICE_REQUEST not in {'auto', 'cuda', 'cpu'}:
    raise ValueError("FC_DEVICE mora biti 'auto', 'cuda' ili 'cpu'.")

_GPU_ID = 0
if _DEVICE_REQUEST != 'cpu':
    try:
        _GPU_ID = int(os.environ.get('FC_GPU_ID', '0'))
        if _GPU_ID < 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError('FC_GPU_ID mora biti cijeli broj >= 0.') from exc


def _probe_xgboost_cuda(gpu_id=0):
    """Return (available, detail) after a real one-tree CUDA training probe.

    Looking only for ``nvidia-smi`` is insufficient: the Python XGBoost wheel
    itself must include CUDA support. Reading build flags is also insufficient
    on machines where CUDA was compiled in but no usable GPU is visible.
    """
    try:
        major = int(str(xgb.__version__).split('.', 1)[0])
    except (TypeError, ValueError):
        major = 0
    if major < 2:
        return False, f'XGBoost {xgb.__version__} je prestar; potreban je >= 2.0'

    requested = f'cuda:{gpu_id}'
    try:
        probe_X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        probe_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        booster = xgb.train(
            {'device': requested, 'tree_method': 'hist',
             'objective': 'reg:squarederror', 'verbosity': 0},
            xgb.DMatrix(probe_X, label=probe_y),
            num_boost_round=1,
        )
        config = json.loads(booster.save_config())
        actual = str(config['learner']['generic_param']['device'])
        if not actual.startswith('cuda'):
            return False, f'XGBoost je izabrao {actual} umjesto {requested}'
        return True, actual
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


_DEVICE_DETAIL = 'forced CPU'
if _DEVICE_REQUEST == 'cpu':
    ML_DEVICE = 'cpu'
else:
    _CUDA_OK, _DEVICE_DETAIL = _probe_xgboost_cuda(_GPU_ID)
    if _CUDA_OK:
        ML_DEVICE = f'cuda:{_GPU_ID}'
    elif _DEVICE_REQUEST == 'cuda':
        raise RuntimeError(
            'GPU je izricito zatrazen, ali XGBoost CUDA provjera nije uspjela: '
            f'{_DEVICE_DETAIL}'
        )
    else:
        ML_DEVICE = 'cpu'

USING_GPU = ML_DEVICE.startswith('cuda')
XGB_DEVICE_PARAMS = {'tree_method': 'hist', 'device': ML_DEVICE}
CATBOOST_DEVICE_PARAMS = (
    {'task_type': 'GPU', 'devices': str(_GPU_ID), 'allow_writing_files': False}
    if USING_GPU else
    {'task_type': 'CPU', 'allow_writing_files': False}
)
# LightGBM's Windows GPU backend is OpenCL (``device_type='gpu'``), not its
# Linux-only CUDA backend. Official Windows wheels include this GPU learner.
_LGB_GPU_DEVICE_ID = _GPU_ID
_LGB_GPU_PLATFORM_ID = None
if USING_GPU:
    try:
        _LGB_GPU_DEVICE_ID = int(os.environ.get('FC_LGB_GPU_DEVICE_ID', str(_GPU_ID)))
        if _LGB_GPU_DEVICE_ID < 0:
            raise ValueError
        if os.environ.get('FC_LGB_GPU_PLATFORM_ID', '').strip():
            _LGB_GPU_PLATFORM_ID = int(os.environ['FC_LGB_GPU_PLATFORM_ID'])
            if _LGB_GPU_PLATFORM_ID < 0:
                raise ValueError
    except ValueError as exc:
        raise ValueError(
            'FC_LGB_GPU_DEVICE_ID/FC_LGB_GPU_PLATFORM_ID moraju biti cijeli brojevi >= 0.'
        ) from exc

LIGHTGBM_DEVICE_PARAMS = (
    {'device_type': 'gpu', 'gpu_device_id': _LGB_GPU_DEVICE_ID,
     **({'gpu_platform_id': _LGB_GPU_PLATFORM_ID}
        if _LGB_GPU_PLATFORM_ID is not None else {})}
    if USING_GPU else
    {'device_type': 'cpu'}
)


def _new_xgb_regressor(**params):
    """Create an XGBoost regressor on the selected device."""
    return xgb.XGBRegressor(**{**params, **XGB_DEVICE_PARAMS})


def _new_xgb_classifier(**params):
    """Create an XGBoost classifier on the selected device."""
    return xgb.XGBClassifier(**{**params, **XGB_DEVICE_PARAMS})


def _new_xgb_booster():
    """Create a raw Booster on the selected device (used when reloading)."""
    return xgb.Booster(params=XGB_DEVICE_PARAMS)


def _train_xgb_booster(params, dtrain, *args, **kwargs):
    """Run raw ``xgb.train`` without allowing a call site to miss the GPU."""
    return xgb.train({**params, **XGB_DEVICE_PARAMS}, dtrain, *args, **kwargs)


def _restore_xgb_device(model):
    """Re-apply runtime device after load_model (saved config may differ)."""
    if isinstance(model, xgb.Booster):
        model.set_param(XGB_DEVICE_PARAMS)
    else:
        model.set_params(**XGB_DEVICE_PARAMS)
    return model


def _new_catboost_regressor(**params):
    """Create a CatBoost regressor on the selected training device."""
    return cb.CatBoostRegressor(**{**params, **CATBOOST_DEVICE_PARAMS})


def _new_catboost_cpu_regressor(**params):
    """CPU fallback used only by auto mode after a backend-specific failure."""
    return cb.CatBoostRegressor(
        **{**params, 'task_type': 'CPU', 'allow_writing_files': False}
    )


def _catboost_predict(model, X):
    """Use GPU inference for the numerical CatBoost models when requested."""
    return model.predict(X, task_type='GPU' if USING_GPU else 'CPU')


def _new_lgbm_regressor(**params):
    """Create a LightGBM regressor on CPU or the Windows OpenCL GPU backend."""
    return lgb.LGBMRegressor(**{**params, **LIGHTGBM_DEVICE_PARAMS})


def _new_lgbm_cpu_regressor(**params):
    """CPU fallback used only by auto mode after an OpenCL-specific failure."""
    return lgb.LGBMRegressor(**{**params, 'device_type': 'cpu'})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "forecast_output")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models_v2")
PREV_RUNS_DIR = os.path.join(BASE_DIR, "previous_runs_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _remove_if_exists(*paths):
    """Invalidate optional artifacts that were absent from the newest fit."""
    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            print(f"  WARN: ne mogu ukloniti zastarjeli artifact {path}: {exc}")


def _write_json_atomic(path, payload, *, indent=None, ensure_ascii=True,
                       allow_nan=True):
    """Write JSON without exposing readers to a truncated partial file."""
    temporary = f'{path}.{os.getpid()}.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=indent, ensure_ascii=ensure_ascii,
                      allow_nan=allow_nan)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


_QUANTILE_ARTIFACT_VERSION = 1


def _quantile_legacy_prefix(param):
    return os.path.join(MODEL_DIR, f'qmod_{param}')


def _quantile_manifest_path(param):
    return f'{_quantile_legacy_prefix(param)}_active.json'


def _quantile_bundle_paths(prefix):
    """Return every file that makes up one quantile bundle generation."""
    alphas = set(float(a) for a in pf.DEFAULT_ALPHAS)
    meta_path = f'{prefix}_cqr.json'
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as handle:
                alphas.update(float(a) for a in json.load(handle).get('alphas', []))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    model_paths = [
        f'{prefix}_q{int(round(alpha * 100)):02d}.txt'
        for alpha in sorted(alphas)
    ]
    return model_paths + [meta_path, f'{prefix}_features.json']


def _remove_quantile_bundle(prefix):
    if prefix:
        _remove_if_exists(*_quantile_bundle_paths(prefix))


def _active_quantile_prefix(param):
    """Resolve the committed generation, retaining legacy-bundle support.

    Once a manifest exists, it is authoritative: an invalid or corrupt
    manifest must never fall back to possibly stale legacy qmod files.
    """
    legacy_prefix = _quantile_legacy_prefix(param)
    manifest_path = _quantile_manifest_path(param)
    if not os.path.exists(manifest_path):
        return legacy_prefix
    try:
        with open(manifest_path, encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"  Kvantili {param}: neispravan active manifest ({exc})")
        return None
    if (manifest.get('version') != _QUANTILE_ARTIFACT_VERSION
            or manifest.get('state') != 'active'):
        return None
    generation = manifest.get('generation')
    expected_prefix = f'qmod_{param}.__gen__.'
    if (not isinstance(generation, str)
            or os.path.basename(generation) != generation
            or not generation.startswith(expected_prefix)):
        print(f"  Kvantili {param}: odbijen neispravan generation manifest")
        return None
    return os.path.join(MODEL_DIR, generation)


def _invalidate_quantile_artifacts(param, reason):
    """Atomically prevent any older quantile generation from being reloaded."""
    legacy_prefix = _quantile_legacy_prefix(param)
    manifest_path = _quantile_manifest_path(param)
    previous_prefix = _active_quantile_prefix(param)
    invalid_manifest = {
        'version': _QUANTILE_ARTIFACT_VERSION,
        'state': 'invalid',
        'reason': str(reason)[:500],
        'updated_utc': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    try:
        # Commit invalidation first. Readers then stop before old files are
        # cleaned, so they can never mistake a partial cleanup for a bundle.
        _write_json_atomic(manifest_path, invalid_manifest, ensure_ascii=False)
    except Exception as exc:
        # If even the marker cannot be committed, remove every prefix that a
        # loader could currently resolve before removing the broken manifest.
        print(f"  WARN: quantile invalidation manifest nije upisan ({exc})")
        if previous_prefix and previous_prefix != legacy_prefix:
            _remove_quantile_bundle(previous_prefix)
        _remove_quantile_bundle(legacy_prefix)
        _remove_if_exists(manifest_path)
        return
    if previous_prefix and previous_prefix != legacy_prefix:
        _remove_quantile_bundle(previous_prefix)
    _remove_quantile_bundle(legacy_prefix)


def _promote_quantile_artifacts(param, models, offsets, features):
    """Save, verify, then atomically activate a complete bundle generation.

    The old manifest is left untouched while LightGBM writes seven model files,
    CQR metadata, and the feature schema. The single manifest replacement is
    the commit point, so an interrupted save cannot expose a mixed generation.
    """
    legacy_prefix = _quantile_legacy_prefix(param)
    manifest_path = _quantile_manifest_path(param)
    previous_prefix = _active_quantile_prefix(param)
    token = f'{time.time_ns()}-{os.getpid()}'
    generation = f'qmod_{param}.__gen__.{token}'
    candidate_prefix = os.path.join(MODEL_DIR, generation)
    try:
        pf.save_quantile_bundle(models, offsets, candidate_prefix)
        _write_json_atomic(f'{candidate_prefix}_features.json', list(features))

        # Reload every staged model before publishing the manifest. Existence
        # checks alone do not detect a truncated LightGBM text model or JSON.
        check_models, check_offsets = pf.load_quantile_bundle(candidate_prefix)
        if check_models is None or check_offsets is None:
            raise RuntimeError('staged quantile bundle se ne moze ponovo ucitati')
        expected_alphas = {float(a) for a in models}
        if {float(a) for a in check_models} != expected_alphas:
            raise RuntimeError('staged quantile bundle nema ocekivane alfa modele')
        with open(f'{candidate_prefix}_features.json', encoding='utf-8') as handle:
            if json.load(handle) != list(features):
                raise RuntimeError('staged quantile feature schema nije vjerodostojna')

        _write_json_atomic(
            manifest_path,
            {
                'version': _QUANTILE_ARTIFACT_VERSION,
                'state': 'active',
                'generation': generation,
                'alphas': sorted(expected_alphas),
                'updated_utc': pd.Timestamp.now(tz='UTC').isoformat(),
            },
            ensure_ascii=False,
        )
    except Exception:
        _remove_quantile_bundle(candidate_prefix)
        raise

    # Promotion is complete. Old/legacy files are now unreachable and can be
    # cleaned without risking the active bundle during an interrupted retrain.
    if previous_prefix and previous_prefix not in (legacy_prefix, candidate_prefix):
        _remove_quantile_bundle(previous_prefix)
    _remove_quantile_bundle(legacy_prefix)
    return candidate_prefix


def _invalidate_onset_artifacts():
    """Prevent a failed/disabled retrain from reloading an older onset bundle."""
    _remove_if_exists(
        os.path.join(MODEL_DIR, 'onset_hazard.json'),
        os.path.join(MODEL_DIR, 'onset_meta.json'),
        os.path.join(MODEL_DIR, 'onset_iso.joblib'),
    )

LAT, LON = 42.2864, 18.84  # E viva!!
# Open-Meteo defaults to GMT when timezone is omitted. Keep every live API
# request and every output timestamp explicitly in Budva local time.
FORECAST_TIMEZONE = "Europe/Podgorica"

MODELS = ["ARPEGE_EUROPE", "GFS_SEAMLESS", "ICON_SEAMLESS", "METEOFRANCE", "ECMWF_IFS025", "ITALIAMETEO_ICON2I", "UKMO_SEAMLESS", "ECMWF_IFS", "KNMI_SEAMLESS", "DMI_SEAMLESS"]
TRUSTED_MODELS = ["ITALIAMETEO_ICON2I"]

# Per-target feature subset (matches MARINE_WIND_MODELS philosophy: high-res LAMs only).
# For wind/gusts, train XGB using ONLY features from these 3 LAMs (plus generic
# engineered features). Set to None to disable filtering for that target.
# To revert to full features for wind: set FEATURE_MODEL_SUBSET = {} (empty dict).
FEATURE_MODEL_SUBSET = {
    "wind_speed_10m":  ["GFS_SEAMLESS", "KNMI_SEAMLESS", "DMI_SEAMLESS", "ICON_SEAMLESS", "ECMWF_IFS025"],
    "wind_gusts_10m":  ["GFS_SEAMLESS", "KNMI_SEAMLESS", "DMI_SEAMLESS", "ICON_SEAMLESS", "ECMWF_IFS025"],
}
TRUSTED_RAIN_MODEL = "ITALIAMETEO_ICON2I"
TRUSTED_CONVECTIVE_VARS = (
    'showers', 'cape', 'convective_inhibition', 'lightning_potential'
)
CORRECTED_RAIN_THRESHOLD_MM = 0.2
# Sustained wind (m/s) above which RAIN is shown as a gusty/stormy code instead
# of "light/drizzle": in Budva a few mm with strong wind (or thunder) reads as a
# storm, not "Slaba/Sitna kiša". See escalate_storm_code().
STORM_WIND_MS = 5.0
TRUSTED_RAIN_THRESHOLD = 0.1
LOCAL_DRY_NOWCAST_HOURS = 4
LOCAL_DRY_LIGHT_RAIN_MAX_MM = 0.7

# Isolated warm-season ICON-2I rain is normally precision-filtered, but native
# convective diagnostics can rescue a real cell.  These are deliberately
# conservative: CAPE alone never opens the gate; it must pair with weak CIN and
# either native showers or wet neighborhood support.  Native lightning/thunder
# is direct evidence and can rescue without the thermodynamic pair.
CONVECTIVE_CAPE_MIN_JKG = 500.0
CONVECTIVE_CIN_MAX_JKG = 75.0
CONVECTIVE_SHOWER_MIN_MM = 0.2
CONVECTIVE_NBR_WET_MIN = 0.12

# Short-burst wind boost (microcellular convection signature in trusted model).
# A "burst" is a contiguous wet run in the trusted model (>= TRUSTED_RAIN_THRESHOLD).
# Up to 2h: always treated as burst. 3-4h: burst only if the run carries
# enough intensity (max hourly >= BURST_EXTENDED_MAX_MM or sum >= BURST_EXTENDED_SUM_MM)
# — heavy slow-movers still count as convective. Longer runs are stratiform.
BURST_MAX_HOURS = 2
BURST_MAX_HOURS_EXTENDED = 4
BURST_EXTENDED_MAX_MM = 2.0    # mm/h needed in run for 3-4h extension
BURST_EXTENDED_SUM_MM = 5.0    # OR total run sum needed for 3-4h extension

# Wind boost activates per-hour only when italia precip >= this threshold
# (slabe kise <2 mm/h ne dirau vjetar — previse nepredvidivo).
BURST_BOOST_PRECIP_MM = 2.0

# Dynamic floors keyed to italia precip in that hour:
#   floor = base + slope * (italia_precip - BURST_BOOST_PRECIP_MM), capped.
# At 2 mm: wind=4.0, gust=7.0. At 8 mm: wind=5.8, gust=11.2.
BURST_WIND_FLOOR_BASE = 4.0
BURST_WIND_FLOOR_SLOPE = 0.3
BURST_WIND_FLOOR_CAP = 8.0
BURST_GUST_FLOOR_BASE = 7.0
BURST_GUST_FLOOR_SLOPE = 0.7
BURST_GUST_FLOOR_CAP = 18.0

# Halo: weaker additive boost on +/-1h around boost hours.
BURST_HALO_WIND_DELTA = 1.0
BURST_HALO_GUST_DELTA = 1.5

BURST_GUST_MAX = 40.0          # absolute cap (safety)
BURST_WIND_MAX = 25.0

# A downdraft scales off the wind field the cell falls into; rain rate alone
# cannot conjure one. Keyed to italia precip only, the floors saturated on any
# heavy hour: an 11.6 mm cell over a dead-calm bay published gust 18.0 m/s for
# an hour whose own band read q90 = 3.2 and P(gust > 17) = 0.00, which is a red
# wind warning on the page. Bound each floor by that q90 band, which the boost
# never touches, while still allowing the base floor.
BURST_AMBIENT_GUST_FACTOR = 2.5
BURST_AMBIENT_WIND_FACTOR = 2.0


# Single wave location: at this resolution (~5km grid for DWD EWAM, ~10km for
# MeteoFrance WAM) close coastal points snap to the same offshore grid cells,
# so it would be dishonest to show "different" wave heights for Bečići vs
# otvoreno more. We fetch ONE representative offshore point and reuse it.
MARINE_WAVE_LOCATION = {
    "lat": 42.252626,
    "lon": 18.831183,
    "name": "Pomorska zona Budve",
    "desc": "Reprezentativna offshore tačka iza Sv. Nikole",
}

# Two wind locations — wind models (ICON-2I 2.2km, AROME 1.3km, ICON-EU 7km)
# have fine enough resolution to actually distinguish sheltered bay from
# open sea, unlike the wave models.
MARINE_WIND_LOCATIONS = [
    {
        "id": "becici",
        "name": "Bečićki zaliv",
        "lat": 42.270259,
        "lon": 18.870885,
        "desc": "Bečićka plaža",
    },
    {
        "id": "open_sea",
        "name": "Otvoreno more",
        "lat": 42.252626,
        "lon": 18.831183,
        "desc": "Iza Svetog Nikole, otvoreno more",
    },
]
# Wave models. ewam (DWD EWAM) is high-res but only ~4 days; meteofrance_wave
# (MFWAM) and gwam (DWD GWAM) are global ~7+ days. TWO globals on purpose:
# day-5+ coverage survives if one global fails/short-fetches on a given run
# (the daily summary uses a NaN-skipping mean across whatever succeeded).
MARINE_MODELS = ["ewam", "meteofrance_wave", "gwam"]
MARINE_VARS = [
    "wave_height", "wave_period", "wave_direction",
    "wind_wave_height", "wind_wave_period",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "sea_surface_temperature",
]
MARINE_WIND_MODELS = [
    "italia_meteo_arpae_icon_2i",
    "icon_eu",
    "arpege_europe",
    "ecmwf_ifs025",                  # global ~7d (day-5+ backbone)
    "gfs_seamless",                  # 2nd global ~7d — day-5+ redundancy
    "knmi_harmonie_arome_europe",
    "dmi_harmonie_arome_europe",
]

# Fallback TTL when upstream meta is unavailable (some "seamless" wrappers
# and certain wave models don't expose /data/<id>/static/meta.json). Source:
# open-meteo.com docs "every N hours" column.
MODEL_UPDATE_HOURS = {
    'arpege_europe': 1,
    'gfs_seamless': 1,
    'icon_seamless': 3,
    'meteofrance_seamless': 1,
    'ecmwf_ifs025': 6,
    'italia_meteo_arpae_icon_2i': 12,
    'ukmo_seamless': 1,
    'bom_access_global': 6,
    'ecmwf_ifs': 6,
    'knmi_seamless': 1,
    'dmi_seamless': 3,
    'ewam': 12,
    'meteofrance_wave': 12,
    'gwam': 12,
    'meteofrance_arpege_europe': 1,
    'knmi_harmonie_arome_europe': 1,
    'dmi_harmonie_arome_europe': 3,
    'icon_eu': 3,
}
DEFAULT_UPDATE_HOURS = 1

# Map each model_id we use to the primary model whose /static/meta.json
# exposes the real upstream `last_run_modification_time`. None = no meta
# endpoint exists; fall back to TTL caching. Verified empirically on
# 2026-05-19 against api.open-meteo.com and marine-api.open-meteo.com.
#
# Value format: (api_host_key, primary_model_id_or_None)
META_PRIMARY = {
    'arpege_europe':              ('forecast', 'meteofrance_arpege_europe'),
    'gfs_seamless':               ('forecast', None),
    'icon_seamless':              ('forecast', 'dwd_icon_d2'),
    'meteofrance_seamless':       ('forecast', 'meteofrance_arpege_europe'),
    'ecmwf_ifs025':               ('forecast', 'ecmwf_ifs025'),
    'italia_meteo_arpae_icon_2i': ('forecast', 'italia_meteo_arpae_icon_2i'),
    'ukmo_seamless':              ('forecast', None),
    'bom_access_global':          ('forecast', 'bom_access_global'),
    'ecmwf_ifs':                  ('forecast', 'ecmwf_ifs'),
    'knmi_seamless':              ('forecast', 'knmi_harmonie_arome_europe'),
    'dmi_seamless':               ('forecast', 'dmi_harmonie_arome_europe'),
    'ewam':                       ('marine',   'dwd_ewam'),
    'meteofrance_wave':           ('marine',   None),
    'meteofrance_arpege_europe':  ('forecast', 'meteofrance_arpege_europe'),
    'knmi_harmonie_arome_europe': ('forecast', 'knmi_harmonie_arome_europe'),
    'dmi_harmonie_arome_europe':  ('forecast', 'dmi_harmonie_arome_europe'),
    'icon_eu':                    ('forecast', 'dwd_icon_eu'),
}
META_HOSTS = {
    'forecast': 'https://api.open-meteo.com/data/',
    'marine':   'https://marine-api.open-meteo.com/data/',
}

_upstream_time_cache = {}  # per-process memoization, avoid double meta fetches


def _get_upstream_update_time(model_id):
    """Return Unix `last_run_modification_time` for the primary model that
    backs our model_id, or None if no meta endpoint is available."""
    if model_id in _upstream_time_cache:
        return _upstream_time_cache[model_id]
    primary = META_PRIMARY.get(model_id)
    if not primary or primary[1] is None:
        _upstream_time_cache[model_id] = None
        return None
    host_key, primary_id = primary
    url = META_HOSTS[host_key] + primary_id + '/static/meta.json'
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        t = r.json().get('last_run_modification_time')
        _upstream_time_cache[model_id] = t
        return t
    except Exception:
        _upstream_time_cache[model_id] = None
        return None

API_CACHE_DIR = os.path.join(MODEL_DIR, 'api_cache')
os.makedirs(API_CACHE_DIR, exist_ok=True)


def _cache_path(endpoint, model_id, lat, lon):
    """Stable path on disk for a (endpoint, model, location) cache entry."""
    safe = endpoint.replace('/', '_').replace(':', '').replace('.', '_').strip('_')
    return os.path.join(API_CACHE_DIR, f"{safe}__{model_id}__{lat}_{lon}.json")


def _cache_age_hours(path):
    """Cache age in hours from the stored fetch time (`.fetched` sidecar), NOT
    file mtime. On GitHub Actions this matters: `git checkout` rewrites every file's
    mtime to the checkout time, so a committed cache always looked 0.0h fresh and
    the TTL check NEVER refetched (frozen stale data — the long-range bug). The
    sidecar stores the real fetch unix time inside the repo content, which git
    preserves. If the sidecar is missing (old caches), return None so the cache
    is treated as expired and refetched once (which then writes the sidecar)."""
    meta = path + '.fetched'
    if not os.path.exists(meta):
        return None
    try:
        with open(meta) as f:
            ts = float(f.read().strip())
        return (time.time() - ts) / 3600.0
    except Exception:
        return None


def _load_fresh_cache(path, max_age_hours):
    """Return parsed JSON if cache exists and is within max_age_hours, else None."""
    if not os.path.exists(path):
        return None
    age_h = _cache_age_hours(path)
    if age_h is None or age_h > max_age_hours:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_stale_cache(path):
    """Return parsed JSON regardless of age (last-resort fallback)."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, _cache_age_hours(path)
    except Exception:
        return None, None


def _save_cache(path, payload):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        # Record the real fetch time so age survives git checkout (see
        # _cache_age_hours). MUST be committed alongside the cache .json.
        with open(path + '.fetched', 'w') as f:
            f.write(str(int(time.time())))
    except Exception as e:
        print(f"  [cache] WARN save failed: {e}")


def _save_upstream_meta(path, upstream_time):
    """Record the upstream model run timestamp next to the cache file."""
    if upstream_time is None:
        return
    try:
        with open(path + '.upstream', 'w') as f:
            f.write(str(int(upstream_time)))
    except Exception:
        pass


def _read_upstream_meta(path):
    p = path + '.upstream'
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _cache_current_for_upstream(path, upstream_time):
    """True if our cache was saved for the current upstream model run."""
    if upstream_time is None or not os.path.exists(path):
        return False
    cached = _read_upstream_meta(path)
    return cached is not None and cached >= upstream_time


MODEL_IDS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",
    "UKMO_SEAMLESS": "ukmo_seamless",
    "BOM_ACCESS": "bom_access_global",
    "ECMWF_IFS": "ecmwf_ifs",
    "KNMI_SEAMLESS": "knmi_seamless",
    "DMI_SEAMLESS": "dmi_seamless",
}

PREV_RUNS_MODELS = [m for m in MODELS if m not in ("ITALIAMETEO_ICON2I", "ECMWF_IFS")]
PREV_RUNS_VARS = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_gusts_10m", "pressure_msl",
    "cloud_cover", "precipitation",
]
PREV_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
WU_STATION_ID = os.environ.get("WU_STATION_ID", "IBUDVA5")
WU_API_KEY = os.environ.get("WU_API_KEY", "")

# --- Monograph-2 aux data ---
# Grid must match fetch_neighborhood.py exactly (row-major, lats S->N, lons W->E,
# Budva = p12, ~12 km spacing).
NBR_LATS = [42.0664, 42.1764, 42.2864, 42.3964, 42.5064]
NBR_LONS = [18.55, 18.695, 18.84, 18.985, 19.13]
NBR_GRID_LATS = [la for la in NBR_LATS for _ in NBR_LONS]
NBR_GRID_LONS = [lo for _ in NBR_LATS for lo in NBR_LONS]
NEIGHBORHOOD_MODELS = ["ITALIAMETEO_ICON2I", "ECMWF_IFS025", "GFS_SEAMLESS",
                       "METEOFRANCE", "KNMI_SEAMLESS"]
PG_LAT, PG_LON = 42.4411, 19.2626  # Podgorica — inland side of the Lovcen ridge

# SKALA radar status — the budva-radar pipeline
# mirrors its radar_status.json into our docs/; the repo path is the fallback.
RADAR_STATUS_CANDIDATES = [
    os.path.join(BASE_DIR, 'docs', 'radar_status.json'),
    r'C:\Users\Matija\Documents\GitHub\budva-radar\docs\radar_status.json',
]

# HP budget — 10-15 Optuna trials is too small for XGBoost's
# learning_rate/n_estimators/min_child_weight; ~50 with early stopping.
# Override per run: FC_TRIALS=15 for a quick retrain.
N_TRIALS = int(os.environ.get('FC_TRIALS', '50'))


def local_now():
    """Naive timestamp in the forecast timezone used by Open-Meteo rows."""
    return pd.Timestamp.now(tz=FORECAST_TIMEZONE).tz_localize(None)

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "showers", "snowfall",
    "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cape", "convective_inhibition", "lightning_potential"
]

TARGET_PARAMS = {
    "temperature_2m":       {"obs": "temperature_2m_obs",       "unit": "\u00b0C",   "display": "Temperatura"},
    "dew_point_2m":         {"obs": "dew_point_2m_obs",         "unit": "\u00b0C",   "display": "Tacka rose"},
    "relative_humidity_2m": {"obs": "relative_humidity_2m_obs", "unit": "%",    "display": "Vlaznost"},
    "wind_speed_10m":       {"obs": "wind_speed_10m_obs",       "unit": "m/s",  "display": "Brzina vjetra"},
    "wind_gusts_10m":       {"obs": "wind_gusts_10m_obs",       "unit": "m/s",  "display": "Udari vjetra"},
    "pressure_msl":         {"obs": "pressure_msl_obs",         "unit": "hPa",  "display": "Pritisak"},
    "cloud_cover":          {"obs": "_derived_cloud_obs",       "unit": "%",    "display": "Oblacnost"},
    "precipitation":        {"obs": "_derived_precip_obs",      "unit": "mm",   "display": "Padavine"},
    "shortwave_radiation":  {"obs": "shortwave_radiation_obs",  "unit": "W/m\u00b2", "display": "Solar. radijacija"},
}

SPLIT_DATE = pd.Timestamp('2025-07-01')

print("=" * 72)
print("  XGBoost +48h v3 --- Bias Correction Pipeline --- Budva")
print("  Models:", len(MODELS), "| Obs: merged (2020-2026) | Split:", SPLIT_DATE.date())
print("  Previous Runs: +Day1/Day2 forecasts for", len(PREV_RUNS_MODELS), "models")
print(f"  ML device: {ML_DEVICE} (requested={_DEVICE_REQUEST}, XGBoost={xgb.__version__})")
if _DEVICE_REQUEST == 'auto' and not USING_GPU:
    print(f"  GPU auto-detect fallback -> CPU: {_DEVICE_DETAIL}")
print("=" * 72)


def compute_clear_sky(dt_series):
    """Approximate clear-sky GHI using *local solar time* for Budva.

    The old curve assumed that the Sun crossed the meridian at 12:00 wall
    time.  In Budva solar noon is later, especially during daylight-saving
    time.  That phase error made the station look artificially dim in the
    afternoon and taught the cloud model a false 17-18h cloud signal.
    """
    dt = pd.to_datetime(dt_series)
    doy = dt.dt.dayofyear
    wall_hour = dt.dt.hour + dt.dt.minute / 60.0

    # NOAA equation-of-time approximation.  The longitude correction uses the
    # local standard meridian (15 degrees per UTC-offset hour), so DST is
    # handled explicitly rather than being baked into an hour-of-day rule.
    b = 2 * np.pi * (doy - 81) / 364.0
    equation_of_time_min = (
        9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    )
    try:
        localized = pd.DatetimeIndex(dt).tz_localize(
            FORECAST_TIMEZONE, ambiguous='NaT', nonexistent='shift_forward'
        )
        utc_offset_h = pd.Series(
            [x.utcoffset().total_seconds() / 3600.0 if not pd.isna(x) else 1.0
             for x in localized],
            index=dt.index,
        )
    except Exception:
        utc_offset_h = pd.Series(1.0, index=dt.index)
    standard_meridian = 15.0 * utc_offset_h
    time_correction_min = 4.0 * (LON - standard_meridian) + equation_of_time_min
    solar_hour = wall_hour + time_correction_min / 60.0

    lat_rad = np.radians(LAT)
    dec = np.radians(23.45 * np.sin(np.radians(360 / 365.25 * (doy - 81))))
    ha = np.radians(15 * (solar_hour - 12))
    sin_e = (np.sin(lat_rad) * np.sin(dec) +
             np.cos(lat_rad) * np.cos(dec) * np.cos(ha)).clip(lower=0)
    return (1361 * sin_e * 0.75).clip(lower=0)


def derive_cloud_cover_from_station_solar(datetimes, solar_wm2, fit_mask=None,
                                          clear_quantile=0.95,
                                          min_clear_envelope_wm2=75.0,
                                          min_station_transmission=0.70):
    """Create a station-calibrated daytime cloud proxy from solar radiation.

    IBUDVA5 is terrain/sensor shaded, most visibly late in the day.  Comparing
    it directly with an unobstructed theoretical or NWP irradiance curve makes
    clear afternoons look cloudy.  A pre-test-period monthly/hourly upper
    envelope represents what *this station* measures on a clear day.  The
    envelope is fit only on ``fit_mask`` rows to avoid test-set leakage.

    Returns ``(cloud_percent, envelope_wm2)``.  Hours with too little usable
    irradiance remain NaN because a pyranometer is not a trustworthy cloud
    sensor close to sunrise/sunset.
    """
    dt = pd.to_datetime(datetimes)
    solar = pd.to_numeric(solar_wm2, errors='coerce')
    if fit_mask is None:
        fit_mask = pd.Series(True, index=dt.index)
    else:
        fit_mask = pd.Series(fit_mask, index=dt.index).fillna(False).astype(bool)

    theoretical_clear = compute_clear_sky(dt)
    work = pd.DataFrame({
        'month': dt.dt.month,
        'hour': dt.dt.hour,
        'solar': solar,
        'theoretical_clear': theoretical_clear,
    }, index=dt.index)
    # Fit station transmission rather than raw W/m2.  Solar geometry then
    # supplies the within-month change in day length/elevation, avoiding a new
    # edge-of-month bias at 10-12h and 17-18h.
    work['station_transmission'] = (
        work['solar'] / work['theoretical_clear'].clip(lower=1.0)
    ).clip(0.0, 2.0)
    fit = work.loc[
        fit_mask & work['solar'].notna() & (work['theoretical_clear'] >= 20.0)
    ]
    transmission_table = fit.groupby(['month', 'hour'])[
        'station_transmission'
    ].quantile(clear_quantile)
    keys = pd.MultiIndex.from_arrays([work['month'], work['hour']])
    transmission = pd.Series(
        transmission_table.reindex(keys).to_numpy(), index=dt.index, dtype=float
    )
    envelope = theoretical_clear * transmission

    reliable = (
        (envelope >= min_clear_envelope_wm2)
        & (transmission >= min_station_transmission)
    )
    cloud = ((1.0 - solar / envelope.clip(lower=1.0)).clip(0.0, 1.0) * 100.0)
    cloud.loc[~reliable | solar.isna()] = np.nan
    return cloud, envelope


def fetch_sst_data(start_date, end_date):
    """Fetch sea surface temperature from Open-Meteo Marine API.
    The Marine API only provides recent + forecast data (~16 days), not historical.
    For historical training use ERA5 archive instead."""
    sst_cache = os.path.join(BASE_DIR, 'budva_sst_cache.csv')
    if os.path.exists(sst_cache):
        sst_df = pd.read_csv(sst_cache, parse_dates=['datetime'])
        # Check that cache covers the requested range (both ends)
        covers_end = sst_df['datetime'].max() >= end_date - pd.Timedelta(days=2)
        covers_start = sst_df['datetime'].min() <= start_date + pd.Timedelta(days=2)
        if covers_end and covers_start:
            print(f"  SST: using cached data ({len(sst_df)} rows)")
            return sst_df

    print(f"  SST: fetching from Marine API...")
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        'latitude': LAT, 'longitude': LON,
        'hourly': 'sea_surface_temperature',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'timezone': FORECAST_TIMEZONE,
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            print(f"  SST: API error {r.status_code}, skipping")
            return None
        data = r.json()
        hourly = data.get('hourly', {})
        if 'time' not in hourly or 'sea_surface_temperature' not in hourly:
            print(f"  SST: no data in response, skipping")
            return None
        sst_df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'sst': hourly['sea_surface_temperature'],
        })
        sst_df.to_csv(sst_cache, index=False)
        print(f"  SST: fetched {len(sst_df)} rows")
        return sst_df
    except Exception as e:
        print(f"  SST: fetch failed ({e}), skipping")
        return None


def fetch_current_station_observation():
    """Fetch current Weather Underground PWS observation if an API key exists."""
    if not WU_API_KEY:
        return None

    url = "https://api.weather.com/v2/pws/observations/current"
    params = {
        "stationId": WU_STATION_ID,
        "format": "json",
        "units": "m",
        "numericPrecision": "decimal",
        "apiKey": WU_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 204:
            print(f"  WU nowcast: nema svjezeg zapisa za {WU_STATION_ID}")
            return None
        r.raise_for_status()
        observations = r.json().get("observations", [])
        if not observations:
            return None

        obs = observations[0]
        metric = obs.get("metric", {}) or {}
        epoch = obs.get("epoch")
        age_min = None
        if epoch:
            age_min = (time.time() - float(epoch)) / 60.0

        return {
            "station": obs.get("stationID", WU_STATION_ID),
            "obs_time_local": obs.get("obsTimeLocal"),
            "age_min": age_min,
            "precip_rate_mm": metric.get("precipRate"),
            "precip_total_mm": metric.get("precipTotal"),
            "humidity": obs.get("humidity"),
            "solar_radiation": obs.get("solarRadiation"),
        }
    except Exception as e:
        print(f"  WU nowcast: ne mogu ucitati trenutni zapis ({e})")
        return None


def station_says_dry_now(obs):
    if not obs:
        return False
    age_min = obs.get("age_min")
    if age_min is None or age_min > 90:
        return False
    precip_rate = obs.get("precip_rate_mm")
    if precip_rate is None:
        return False
    return float(precip_rate) <= 0.05


def read_radar_nowcast(max_age_min=25):
    """SKALA radar nowcast for 0-6h precip blending.

    Reads radar_status.json produced by the budva-radar pipeline (mirrored
    into our docs/). Uses sources.dhmz.approaching.nowcast_details.p_by_lead
    {15,30,60,120 min} + the dominant-cell receding flag. Returns a dict or
    None when the file is missing/stale — callers must degrade gracefully.
    The SKALA verification log has no overlap with the training period
    yet, so the radar is inference-time evidence only ("collect now, train
    later" for the onset model)."""
    for path in RADAR_STATUS_CANDIDATES:
        try:
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                st = json.load(f)
            gen = pd.Timestamp(st.get('generated'))
            age_min = (local_now() - gen).total_seconds() / 60.0
            if age_min > max_age_min or age_min < -10:
                continue
            dh = st.get('sources', {}).get('dhmz', {})
            ap = dh.get('approaching') or {}
            nd = ap.get('nowcast_details') or {}
            pbl = nd.get('p_by_lead') or {}
            if '60' not in pbl:
                continue
            p60 = float(pbl['60'])
            dom = nd.get('dominant') or {}
            return {
                'p15': float(pbl.get('15', p60)),
                'p30': float(pbl.get('30', p60)),
                'p60': p60,
                'p120': float(pbl.get('120', p60)),
                'raining_now': bool(ap.get('rain_at_location', False)),
                'receding': bool(dom.get('receding') or False),
                'dist_km': dom.get('dist_km'),
                'age_min': round(age_min, 1),
                'source_path': path,
            }
        except Exception:
            continue
    return None


def _validate_precip_observations(frame, precipitation, source,
                                  min_year_rows=500):
    """Fail closed when hourly rain labels look like wet-only missing data.

    Weather Underground historically rendered some dry values as ``--``.
    A stale detailed archive consequently contains recent years where almost
    every dry hour is NaN while temperature/RH/pressure are present. Training
    after dropping those NaNs produces a severely rain-biased classifier.
    """
    dt = pd.to_datetime(frame['datetime'], errors='coerce')
    values = pd.to_numeric(precipitation, errors='coerce')
    core_cols = [c for c in ('temperature_2m_obs',
                             'relative_humidity_2m_obs',
                             'pressure_msl_obs') if c in frame.columns]
    if core_cols:
        required = min(2, len(core_cols))
        station_present = frame[core_cols].notna().sum(axis=1) >= required
    else:
        station_present = pd.Series(True, index=frame.index)

    qa = pd.DataFrame({
        'year': dt.dt.year,
        'station_present': station_present.values,
        'precip': values.values,
    }).dropna(subset=['year'])
    rows = []
    suspicious = []
    for year, group in qa.groupby('year', sort=True):
        valid = group['precip'].replace([np.inf, -np.inf], np.nan).dropna()
        completeness = len(valid) / len(group)
        wet_fraction = (
            float((valid >= CORRECTED_RAIN_THRESHOLD_MM).mean())
            if len(valid) else np.nan
        )
        missing_with_station = int(
            (group['station_present'] & group['precip'].isna()).sum()
        )
        row = {
            'year': int(year), 'rows': int(len(group)),
            'completeness': float(completeness),
            'wet_fraction': wet_fraction,
            'missing_with_station': missing_with_station,
        }
        rows.append(row)
        print(f"    rain-label QA {int(year)}: valid={completeness:.1%}, "
              f"wet>={CORRECTED_RAIN_THRESHOLD_MM:.1f}mm={wet_fraction:.1%} "
              f"({len(valid)}/{len(group)})")
        if len(group) >= min_year_rows and (
                completeness < 0.95 or missing_with_station > 0 or
                not np.isfinite(wet_fraction) or
                wet_fraction < 0.005 or wet_fraction > 0.30):
            suspicious.append(row)

    if suspicious:
        details = '; '.join(
            f"{r['year']}: valid={r['completeness']:.1%}, "
            f"wet={r['wet_fraction']:.1%}" for r in suspicious
        )
        raise RuntimeError(
            f"Precipitation observations from {source!r} failed QA ({details}). "
            "Refusing to train on likely wet-only/missing-dry labels."
        )
    return rows


def load_historical_data():
    print("\n[1/6] Ucitavanje istorijskih podataka...")
    all_dfs = {}
    available_models = []
    for m in MODELS:
        path = os.path.join(BASE_DIR, f"budva_{m}_detailed.csv")
        if not os.path.exists(path):
            print(f"  {m}: NEMA FAJLA - preskačem (pokreni fetch_new_models.py)")
            continue
        all_dfs[m] = pd.read_csv(path, parse_dates=['datetime'], low_memory=False)
        available_models.append(m)
        print(f"  {m}: {all_dfs[m].shape[0]} redova")

    if not available_models:
        raise RuntimeError("Nema nijednog model fajla!")

    forecast_cols = [c for c in all_dfs[available_models[0]].columns if c.endswith('_model')]
    base = all_dfs[available_models[0]].copy()
    base.rename(columns={c: f"{available_models[0]}_{c}" for c in forecast_cols}, inplace=True)
    for m in available_models[1:]:
        other_cols = [c for c in all_dfs[m].columns if c.endswith('_model')]
        other = all_dfs[m][['datetime'] + other_cols].copy()
        other.rename(columns={c: f"{m}_{c}" for c in other_cols}, inplace=True)
        base = base.merge(other, on='datetime', how='left')
    base.sort_values('datetime', inplace=True)
    base.reset_index(drop=True, inplace=True)

    # Detailed model CSVs are a stale denormalized observation snapshot. In
    # addition to missing recent dry-rain labels, they retain the old hourly-
    # mean gust bug and even a 9934 W/m2 solar outlier that corrupts cloud
    # derivation. Replace every observation target from the canonical hourly WU
    # table; never combine_first with stale values for a missing station hour.
    canonical_obs_path = os.path.join(
        BASE_DIR, 'wu_data', 'merged_observations.csv'
    )
    if not os.path.exists(canonical_obs_path):
        raise FileNotFoundError(
            f"Canonical WU observations are required for safe retraining: "
            f"{canonical_obs_path}"
        )
    try:
        canonical_obs = pd.read_csv(
            canonical_obs_path, parse_dates=['datetime'], low_memory=False
        )
        required_obs = {'datetime', 'temp_c', 'dewpoint_c', 'humidity_pct',
                        'wind_ms', 'gust_ms', 'pressure_hpa',
                        'precip_rate_mm', 'solar_wm2'}
        missing_obs = sorted(required_obs - set(canonical_obs.columns))
        if missing_obs:
            raise ValueError(f"nedostaju kolone: {missing_obs}")
        if canonical_obs['datetime'].isna().any():
            raise ValueError('canonical observations sadrze neispravan datetime')
        if canonical_obs['datetime'].duplicated().any():
            duplicates = int(canonical_obs['datetime'].duplicated(keep=False).sum())
            raise ValueError(f'canonical observations imaju {duplicates} duplicate timestampa')
        if (canonical_obs['datetime'].min() > base['datetime'].min()
                or canonical_obs['datetime'].max() < base['datetime'].max()):
            raise ValueError(
                'canonical observations ne pokrivaju puni model archive raspon '
                f"({canonical_obs['datetime'].min()}..{canonical_obs['datetime'].max()} "
                f"vs {base['datetime'].min()}..{base['datetime'].max()})"
            )

        canonical_map = {
            'temp_c': 'temperature_2m_obs',
            'dewpoint_c': 'dew_point_2m_obs',
            'humidity_pct': 'relative_humidity_2m_obs',
            'wind_ms': 'wind_speed_10m_obs',
            'gust_ms': 'wind_gusts_10m_obs',
            'pressure_hpa': 'pressure_msl_obs',
            'precip_rate_mm': '_canonical_precip_rate_mm',
            'solar_wm2': 'shortwave_radiation_obs',
        }
        optional_map = {
            'precip_accum_mm': 'precipitation_obs',
            'uv': 'uv_index_obs',
            'wind_dir_deg': 'wind_direction_10m_obs',
        }
        canonical_map.update({k: v for k, v in optional_map.items()
                              if k in canonical_obs.columns})
        overlay = canonical_obs[['datetime'] + list(canonical_map)].rename(
            columns=canonical_map
        )
        # Avoid merge suffixes, then replace all stale observation values.
        target_cols = list(canonical_map.values())
        base.drop(columns=[c for c in target_cols if c in base.columns], inplace=True)
        base = base.merge(overlay, on='datetime', how='left', validate='one_to_one')
        print(f"  Canonical WU observation targets: {canonical_obs_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Ne mogu ucitati canonical WU observations: {exc}"
        ) from exc

    solar = pd.to_numeric(
        base.get('shortwave_radiation_obs', pd.Series(dtype=float)),
        errors='coerce',
    )
    # Fit the clear-day envelope strictly before the untouched report period.
    # This removes IBUDVA5's hour-specific terrain/sensor attenuation without
    # learning anything from the held-out target rows.
    cloud, station_clear_envelope = derive_cloud_cover_from_station_solar(
        base['datetime'], solar, fit_mask=base['datetime'] < SPLIT_DATE,
    )
    base['_derived_cloud_obs'] = cloud
    print(f"  Cloud cover derived: {cloud.notna().sum()} valid "
          f"(station-calibrated monthly/hourly clear envelope; "
          f"median={station_clear_envelope.dropna().median():.0f} W/m2)")

    # Use canonical precipitation RATE (mm/hr), not accumulated precipitation.
    vals = pd.to_numeric(base['_canonical_precip_rate_mm'], errors='coerce')
    if (vals.dropna() < 0).any() or np.isinf(vals.to_numpy(dtype=float)).any():
        raise RuntimeError('Canonical precipitation contains negative/infinite values')
    _validate_precip_observations(base, vals, '_canonical_precip_rate_mm')
    base['_derived_precip_obs'] = vals
    base.drop(columns=['_canonical_precip_rate_mm'], inplace=True)
    n_valid = int(vals.notna().sum())
    n_nonzero = int((vals > 0).sum())
    print(f"  Hourly precip from canonical WU rate: {n_nonzero} non-zero, "
          f"{n_valid} valid")

    print("  Ucitavanje previous runs podataka (Day1/Day2)...")
    prev_merged = 0
    for m in PREV_RUNS_MODELS:
        prev_path = os.path.join(PREV_RUNS_DIR, f"{m}_previous_runs.csv")
        if not os.path.exists(prev_path):
            print(f"    {m}: nema fajla - preskačem")
            continue
        prev = pd.read_csv(prev_path, parse_dates=['datetime'])
        rename_map = {}
        for v in PREV_RUNS_VARS:
            for lag in ['previous_day1', 'previous_day2']:
                old_col = f"{v}_{lag}"
                new_col = f"{m}_{v}_{lag}"
                if old_col in prev.columns:
                    rename_map[old_col] = new_col
        prev_keep = ['datetime'] + list(rename_map.keys())
        prev = prev[[c for c in prev_keep if c in prev.columns]].rename(columns=rename_map)
        base = base.merge(prev, on='datetime', how='left')
        prev_merged += 1
        n_valid = base[f'{m}_temperature_2m_previous_day1'].notna().sum() if f'{m}_temperature_2m_previous_day1' in base.columns else 0
        print(f"    {m}: merged ({n_valid} valid Day1 rows)")
    print(f"  Previous runs: {prev_merged} modela merged")

    # --- SST integration ---
    # Open-Meteo Marine API ne daje istorijske SST vrijednosti (samo forecast horizon
    # od oko 16 dana). Ranija logika je dovodila do "0 valid rows merged" greske.
    # SST se sad fetch-uje samo za forecast horizon u _fetch_marine_waves i prikazuje
    # u marine_forecast (i u daily cards preko sst_avg).
    # Za korišćenje SST kao istorijski feature za treniranje, trebalo bi koristiti
    # ERA5 archive (drugi izvor) - nije implementirano.

    # --- Stability/synoptic features ---
    # Merge in CAPE/CIN/lifted_index/wind@500-700hPa per model. Generated by
    # fetch_stability_features.py. Missing files are silently skipped (no break).
    stab_loaded = 0
    for m in MODELS:
        stab_path = os.path.join(BASE_DIR, f"budva_{m}_stability.csv")
        if not os.path.exists(stab_path):
            continue
        sdf = pd.read_csv(stab_path, parse_dates=['datetime'])
        # Drop columns that already exist (idempotency)
        new_cols = [c for c in sdf.columns if c != 'datetime' and c not in base.columns]
        if not new_cols:
            continue
        base = base.merge(sdf[['datetime'] + new_cols], on='datetime', how='left')
        stab_loaded += 1
    if stab_loaded > 0:
        print(f"  Stability features: merged {stab_loaded} model stability CSV(s)")

    # --- Neighborhood 5x5 precip ---
    # Generated by fetch_neighborhood.py. Columns get the model prefix here so
    # multiple models' grids can coexist ({M}_nbr_p00..p24).
    nbr_dir = os.path.join(BASE_DIR, 'neighborhood_data')
    nbr_loaded = 0
    if os.path.isdir(nbr_dir):
        for m in MODELS:
            nbr_path = os.path.join(nbr_dir, f"{m}_neighborhood.csv")
            if not os.path.exists(nbr_path):
                continue
            ndf = pd.read_csv(nbr_path, parse_dates=['datetime'])
            ren = {c: f"{m}_{c}" for c in ndf.columns if c.startswith('nbr_p')}
            ndf = ndf.rename(columns=ren)
            new_cols = [c for c in ndf.columns if c != 'datetime' and c not in base.columns]
            if not new_cols:
                continue
            base = base.merge(ndf[['datetime'] + new_cols], on='datetime', how='left')
            nbr_loaded += 1
    if nbr_loaded > 0:
        print(f"  Neighborhood precip: merged {nbr_loaded} model grid CSV(s)")

    # --- Podgorica MSLP ---
    pg_dir = os.path.join(BASE_DIR, 'podgorica_data')
    pg_loaded = 0
    if os.path.isdir(pg_dir):
        for m in MODELS:
            pg_path = os.path.join(pg_dir, f"{m}_podgorica.csv")
            if not os.path.exists(pg_path):
                continue
            pdf_ = pd.read_csv(pg_path, parse_dates=['datetime'])
            new_cols = [c for c in pdf_.columns if c != 'datetime' and c not in base.columns]
            if not new_cols:
                continue
            base = base.merge(pdf_[['datetime'] + new_cols], on='datetime', how='left')
            pg_loaded += 1
    if pg_loaded > 0:
        print(f"  Podgorica MSLP: merged {pg_loaded} model CSV(s)")

    # --- Observation QC ---
    # Flag physically impossible values as NaN to prevent training on bad data.
    qc_limits = {
        'temperature_2m_obs': (-20, 50),
        'dew_point_2m_obs': (-30, 40),
        'relative_humidity_2m_obs': (0, 100),
        'wind_speed_10m_obs': (0, 60),
        'wind_gusts_10m_obs': (0, 100),
        'pressure_msl_obs': (940, 1070),
        'shortwave_radiation_obs': (0, 1400),
    }
    total_flagged = 0
    for col, (lo, hi) in qc_limits.items():
        if col not in base.columns:
            continue
        vals = pd.to_numeric(base[col], errors='coerce')
        bad = (vals < lo) | (vals > hi)
        n_bad = bad.sum()
        if n_bad > 0:
            base.loc[bad, col] = np.nan
            total_flagged += n_bad
    if total_flagged > 0:
        print(f"  Observation QC: flagged {total_flagged} values as NaN")

    print(f"  Merged: {base.shape[0]} x {base.shape[1]}")
    return base


def compute_bias_tables(df):
    print("\n  Kreiranje tabela istorijskog biasa (samo na train podacima)...")
    train = df[df['datetime'] < SPLIT_DATE].copy()
    train['month'] = train['datetime'].dt.month
    train['hour'] = train['datetime'].dt.hour

    bias_tables = {}
    for param, info in TARGET_PARAMS.items():
        obs_col = info['obs']
        if obs_col not in train.columns:
            continue
        obs = pd.to_numeric(train[obs_col], errors='coerce')

        for m in MODELS:
            fcst_col = f"{m}_{param}_model"
            if fcst_col not in train.columns:
                continue
            fcst = pd.to_numeric(train[fcst_col], errors='coerce')
            err = fcst - obs
            tmp = pd.DataFrame({'err': err, 'month': train['month'], 'hour': train['hour']})
            table = tmp.groupby(['month', 'hour'])['err'].agg(['mean', 'std']).reset_index()
            table.columns = ['month', 'hour', 'bias_mean', 'bias_std']
            key = f"{m}_{param}"
            bias_tables[key] = table

    print(f"  Tabele biasa: {len(bias_tables)} (model x param kombinacija)")
    return bias_tables


def apply_bias_features(df, bias_tables):
    df = df.copy()
    df['_month'] = df['datetime'].dt.month
    df['_hour'] = df['datetime'].dt.hour

    for key, table in bias_tables.items():
        merged = df[['_month', '_hour']].merge(
            table, left_on=['_month', '_hour'], right_on=['month', 'hour'], how='left'
        )
        df[f'{key}_hist_bias'] = merged['bias_mean'].values
        df[f'{key}_hist_bias_std'] = merged['bias_std'].values

    df.drop(columns=['_month', '_hour'], inplace=True)
    return df


def _rain_consensus_stats(rain_values, wet_threshold=0.1,
                          consensus_fraction=0.5):
    """Row-wise wet votes with missing members excluded from the denominator.

    Returns ``(wet_count, available_count, agreement, consensus_wet_hour)``.
    Rows with no available model retain NaN agreement/hour instead of being
    silently interpreted as dry.
    """
    values = rain_values.apply(pd.to_numeric, errors='coerce')
    available_count = values.notna().sum(axis=1)
    wet_count = (values >= wet_threshold).sum(axis=1)
    agreement = wet_count.div(available_count.where(available_count > 0))
    wet_hour = (agreement >= consensus_fraction).astype(float)
    wet_hour.loc[available_count == 0] = np.nan
    return wet_count, available_count, agreement, wet_hour


def _convective_rescue_mask(frame, isolated_signal):
    """Return isolated ICON-2I hours with independent convective support.

    The rescue is intentionally precision-first.  CAPE by itself is common on
    false-alarm summer days, so it only counts with weak inhibition plus native
    ICON-2I showers or a spatially wet neighborhood.  Native lightning or an
    explicit thunderstorm weather code is already direct convective evidence.
    Missing diagnostics never become positive evidence.
    """
    idx = frame.index

    def _series(name):
        return pd.to_numeric(
            frame.get(name, pd.Series(np.nan, index=idx)), errors='coerce'
        )

    model = TRUSTED_RAIN_MODEL
    cape = _series(f'{model}_cape_model').combine_first(
        _series('cape_ens_max')
    )
    cin_strength = _series(f'{model}_convective_inhibition_model').abs()
    cin_strength = cin_strength.combine_first(_series('cin_ens_mean').abs())
    showers = _series(f'{model}_showers_model')
    lightning = _series(f'{model}_lightning_potential_model')
    neighborhood = _series(f'{model}_nbr_fwet').combine_first(
        _series('nbr_ens_fwet')
    )
    thunder_votes = _series('storm_wc_count').fillna(0.0)

    thermodynamic_support = (
        (cape >= CONVECTIVE_CAPE_MIN_JKG)
        & (cin_strength <= CONVECTIVE_CIN_MAX_JKG)
    )
    direct_support = (lightning > 0.0) | (thunder_votes >= 1.0)
    shower_support = (
        (showers >= CONVECTIVE_SHOWER_MIN_MM) & thermodynamic_support
    )
    neighborhood_support = (
        (neighborhood >= CONVECTIVE_NBR_WET_MIN) & thermodynamic_support
    )
    isolated = pd.Series(np.asarray(isolated_signal, dtype=bool), index=idx)
    return (isolated & (direct_support | shower_support | neighborhood_support)).values


def engineer_features(df):
    out = df.copy()

    model_cols = [c for c in out.columns if c.endswith('_model')]
    for c in model_cols:
        if out[c].dtype == 'object':
            out[c] = pd.to_numeric(out[c], errors='coerce')

    out['hour'] = out['datetime'].dt.hour
    out['month'] = out['datetime'].dt.month
    out['day_of_year'] = out['datetime'].dt.dayofyear
    # harmonics must follow the SUN, not the wall clock. Local
    # (Europe/Podgorica) hours jump by 1h at each DST transition, which shifts
    # the learned diurnal phase between winter and summer. Convert to UTC for
    # the cyclic encodings; 'hour'/'month'/'season' stay local for the
    # human-schedule-like splits (bias tables, seasonal weights).
    try:
        _utc_hour = (out['datetime']
                     .dt.tz_localize(FORECAST_TIMEZONE, ambiguous='NaT',
                                     nonexistent='shift_forward')
                     .dt.tz_convert('UTC').dt.hour)
        _utc_hour = _utc_hour.fillna(out['hour'])  # NaT only on the repeated DST hour
    except Exception:
        _utc_hour = out['hour']
    out['utc_hour'] = _utc_hour.astype(float)
    out['hour_sin'] = np.sin(2 * np.pi * out['utc_hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['utc_hour'] / 24)
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    out['doy_sin'] = np.sin(2 * np.pi * out['day_of_year'] / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * out['day_of_year'] / 365.25)
    # 2nd harmonics (report A1): capture semidiurnal + semiannual structure
    # (land/sea-breeze, twice-yearly transition) more efficiently than dummies.
    out['hour_sin2'] = np.sin(4 * np.pi * out['utc_hour'] / 24)
    out['hour_cos2'] = np.cos(4 * np.pi * out['utc_hour'] / 24)
    out['doy_sin2'] = np.sin(4 * np.pi * out['day_of_year'] / 365.25)
    out['doy_cos2'] = np.cos(4 * np.pi * out['day_of_year'] / 365.25)
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out['season'] = out['month'].map(season_map)

    clear = compute_clear_sky(out['datetime'])
    out['is_daytime'] = (clear > 20).astype(float)
    out['clear_sky_rad'] = clear

    # --- Missingness indicators per model ---
    # XGBoost's sparsity-aware splits handle NaN natively; indicators let it learn
    # that model availability itself carries information.
    for m in MODELS:
        # Use temperature as proxy for model availability
        ref_col = f"{m}_temperature_2m_model"
        if ref_col in out.columns:
            out[f'is_{m}_available'] = pd.to_numeric(out[ref_col], errors='coerce').notna().astype(float)
    # Count of available models (how many NWP runs exist for this hour)
    avail_cols = [f'is_{m}_available' for m in MODELS if f'is_{m}_available' in out.columns]
    if avail_cols:
        out['n_models_available'] = out[avail_cols].sum(axis=1)

    ensemble_params = ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
                       'apparent_temperature', 'wind_speed_10m', 'wind_gusts_10m',
                       'wind_direction_10m', 'pressure_msl', 'surface_pressure',
                       'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                       'cloud_cover_high', 'precipitation', 'shortwave_radiation',
                       'direct_radiation', 'diffuse_radiation', 'rain', 'showers',
                       'lightning_potential']

    for param in ensemble_params:
        mcols = [f"{m}_{param}_model" for m in MODELS if f"{m}_{param}_model" in out.columns]
        if len(mcols) < 2:
            continue
        vals = out[mcols].apply(pd.to_numeric, errors='coerce')
        out[f'{param}_ens_mean'] = vals.mean(axis=1)
        out[f'{param}_ens_std'] = vals.std(axis=1)
        out[f'{param}_ens_range'] = vals.max(axis=1) - vals.min(axis=1)
        out[f'{param}_ens_median'] = vals.median(axis=1)
        out[f'{param}_ens_min'] = vals.min(axis=1)
        out[f'{param}_ens_max'] = vals.max(axis=1)

        if len(mcols) >= 4:
            sorted_vals = np.sort(vals.values, axis=1)
            out[f'{param}_ens_trimmed_mean'] = np.nanmean(sorted_vals[:, 1:-1], axis=1)
        for m in MODELS:
            c = f"{m}_{param}_model"
            if c in out.columns:
                out[f'{m}_{param}_dev'] = pd.to_numeric(out[c], errors='coerce') - out[f'{param}_ens_mean']

    if 'temperature_2m_ens_mean' in out.columns and 'dew_point_2m_ens_mean' in out.columns:
        out['temp_dew_spread'] = out['temperature_2m_ens_mean'] - out['dew_point_2m_ens_mean']
        # Dew point deficit feature
        out['dew_point_deficit'] = out['temperature_2m_ens_mean'] - out['dew_point_2m_ens_mean']

    # --- Clear-sky index (CSI) features for solar radiation ---
    # CSI = GHI / GHI_clearsky normalizes out the diurnal cycle and solar geometry,
    # constraining the target to approximately [0, 1.2]. 15-25% MAE reduction expected.
    if 'shortwave_radiation_ens_mean' in out.columns and 'clear_sky_rad' in out.columns:
        cs = out['clear_sky_rad'].clip(lower=1)
        out['csi_ens_mean'] = (out['shortwave_radiation_ens_mean'] / cs).clip(0, 1.5)
        out['csi_ens_std'] = out.get('shortwave_radiation_ens_std', 0) / cs
        for m in MODELS:
            sw_col = f"{m}_shortwave_radiation_model"
            if sw_col in out.columns:
                out[f'{m}_csi'] = (pd.to_numeric(out[sw_col], errors='coerce') / cs).clip(0, 1.5)

    if 'wind_speed_10m_ens_mean' in out.columns and 'pressure_msl_ens_mean' in out.columns:
        out['wind_pressure_idx'] = out['wind_speed_10m_ens_mean'] / (out['pressure_msl_ens_mean'] / 1013.25).clip(lower=0.9)
    if 'cloud_cover_ens_mean' in out.columns:
        out['cloud_solar_discrepancy'] = out.get('shortwave_radiation_ens_mean', 0) / (out['cloud_cover_ens_mean'].clip(lower=1))
    if 'pressure_msl_ens_mean' in out.columns:
        out['pres_tend_3h'] = out['pressure_msl_ens_mean'].diff(3)
        out['pres_tend_6h'] = out['pressure_msl_ens_mean'].diff(6)
    if 'temperature_2m_ens_mean' in out.columns:
        out['temp_tend_3h'] = out['temperature_2m_ens_mean'].diff(3)
        out['temp_tend_6h'] = out['temperature_2m_ens_mean'].diff(6)

    rain_mcols = [f"{m}_precipitation_model" for m in MODELS if f"{m}_precipitation_model" in out.columns]
    if rain_mcols:
        rain_vals = out[rain_mcols].apply(pd.to_numeric, errors='coerce')
        (out['rain_model_count'], out['rain_model_available_count'],
         out['rain_agreement'], _) = _rain_consensus_stats(rain_vals)

        # Precision-first false-alarm fingerprint features.
        # High-res LAMs vs global models. Real frontal rain shows up in both;
        # weakly-forced summer convection often only triggers in high-res LAMs.
        HIGH_RES = ["ITALIAMETEO_ICON2I", "ICON_SEAMLESS", "KNMI_SEAMLESS", "DMI_SEAMLESS"]
        GLOBAL_M = ["GFS_SEAMLESS", "ECMWF_IFS025", "ECMWF_IFS", "UKMO_SEAMLESS",
                    "ARPEGE_EUROPE", "METEOFRANCE"]
        hr_cols = [f"{m}_precipitation_model" for m in HIGH_RES if f"{m}_precipitation_model" in out.columns]
        gl_cols = [f"{m}_precipitation_model" for m in GLOBAL_M if f"{m}_precipitation_model" in out.columns]
        if hr_cols:
            _, _, out['frac_high_res_wet'], _ = _rain_consensus_stats(out[hr_cols])
        if gl_cols:
            _, _, out['frac_global_wet'], _ = _rain_consensus_stats(out[gl_cols])
        if hr_cols and gl_cols:
            # Large positive = "only high-res sees it" - classic false-alarm pattern
            out['regional_minus_global_wet'] = out['frac_high_res_wet'] - out['frac_global_wet']

        # italiameteo_isolated = ICON-2I says rain but <= 30% of others do.
        # Direct false-alarm fingerprint per NOT a gate, just a feature.
        icon2i_col = "ITALIAMETEO_ICON2I_precipitation_model"
        if icon2i_col in out.columns:
            icon2i_amount = pd.to_numeric(out[icon2i_col], errors='coerce')
            icon2i_wet = icon2i_amount >= TRUSTED_RAIN_THRESHOLD
            # Dedicated trusted-rain features.  Keeping these explicit lets the
            # classifier learn ICON-2I intensity, onset, persistence, and
            # isolation differently instead of relying on deep splits over the
            # raw amount plus generic ensemble statistics.
            out['italiameteo_rain_amount'] = icon2i_amount
            out['italiameteo_rain_log_amount'] = np.log1p(icon2i_amount.clip(lower=0))
            out['italiameteo_rain_signal'] = icon2i_wet.astype(float)
            out.loc[icon2i_amount.isna(), 'italiameteo_rain_signal'] = np.nan
            prev_wet = icon2i_wet.shift(1, fill_value=False)
            out['italiameteo_rain_onset'] = (icon2i_wet & ~prev_wet).astype(float)
            out.loc[icon2i_amount.isna(), 'italiameteo_rain_onset'] = np.nan
            out['italiameteo_rain_3h_max'] = icon2i_amount.rolling(
                3, min_periods=1, center=True
            ).max()
            out['italiameteo_rain_3h_sum'] = icon2i_amount.rolling(
                3, min_periods=1, center=True
            ).sum()
            out['italiameteo_isolated'] = (icon2i_wet & (out['rain_agreement'] <= 0.30)).astype(float)
            out.loc[icon2i_amount.isna(), 'italiameteo_isolated'] = np.nan

        # Precipitation quantile features — tails matter more than mean for intermittent precip
        if len(rain_mcols) >= 4:
            out['precip_ens_p10'] = rain_vals.quantile(0.1, axis=1)
            out['precip_ens_p25'] = rain_vals.quantile(0.25, axis=1)
            out['precip_ens_p75'] = rain_vals.quantile(0.75, axis=1)
            out['precip_ens_p90'] = rain_vals.quantile(0.9, axis=1)
            # Conditional ensemble mean: mean only of models predicting rain (avoids dilution by zeros)
            rain_only = rain_vals.where(rain_vals > 0.1)
            out['precip_ens_mean_rainy'] = rain_only.mean(axis=1).fillna(0)

        # Ensemble dry consensus: all models predict < 0.1mm = strong dry signal
        _rain_available = rain_vals.notna().any(axis=1)
        out['ens_all_dry'] = (rain_vals.max(axis=1) < 0.1).astype(float)
        out.loc[~_rain_available, 'ens_all_dry'] = np.nan
        # Max model precipitation — captures extreme predictions the mean smooths out
        out['precip_ens_max_single'] = rain_vals.max(axis=1)

        # --- LAM-pair agreement (report A1/B3) ---
        # High-res LAMs (ICON-2I 2.2km, KNMI/DMI HARMONIE-AROME) are the best
        # timing signal at this scale; their mutual agreement is a strong,
        # sharper-than-global rain predictor/anchor.
        LAM_MODELS = ["ITALIAMETEO_ICON2I", "KNMI_SEAMLESS", "DMI_SEAMLESS"]
        lam_cols = [f"{m}_precipitation_model" for m in LAM_MODELS
                    if f"{m}_precipitation_model" in out.columns]
        if len(lam_cols) >= 2:
            lam_vals = out[lam_cols].apply(pd.to_numeric, errors='coerce')
            (lam_wet_count, lam_available_count, out['lam_frac_wet'],
             _) = _rain_consensus_stats(lam_vals)
            out['lam_all_wet'] = (
                (lam_available_count == len(lam_cols)) &
                (lam_wet_count == lam_available_count)
            ).astype(float)
            out.loc[lam_available_count == 0, 'lam_all_wet'] = np.nan
            out['lam_precip_median'] = lam_vals.median(axis=1)
            out['lam_precip_spread'] = lam_vals.std(axis=1)

    wc_feat_cols = [f"{m}_weather_code_model" for m in MODELS if f"{m}_weather_code_model" in out.columns]
    if wc_feat_cols:
        wc_vals = out[wc_feat_cols].apply(pd.to_numeric, errors='coerce')
        out['rain_wc_count'] = (wc_vals >= 51).sum(axis=1)
        out['storm_wc_count'] = (wc_vals >= 95).sum(axis=1)

    if 'precipitation_ens_mean' in out.columns and 'temperature_2m_ens_mean' in out.columns:
        out['precip_x_temp'] = out['precipitation_ens_mean'] * out['temperature_2m_ens_mean']
        out['precip_x_temp_std'] = out['precipitation_ens_mean'] * out.get('temperature_2m_ens_std', 0)

    if 'cloud_cover_ens_mean' in out.columns:
        out['cloud_tend_3h'] = out['cloud_cover_ens_mean'].diff(3)
        out['cloud_tend_6h'] = out['cloud_cover_ens_mean'].diff(6)

    if 'precipitation_ens_mean' in out.columns:
        out['precip_tend_3h'] = out['precipitation_ens_mean'].diff(3)
        out['precip_tend_6h'] = out['precipitation_ens_mean'].diff(6)

    if 'relative_humidity_2m_ens_mean' in out.columns:
        out['humidity_tend_3h'] = out['relative_humidity_2m_ens_mean'].diff(3)

    if 'temp_dew_spread' in out.columns:
        out['dew_spread_tend_3h'] = out['temp_dew_spread'].diff(3)
        out['dew_spread_tend_6h'] = out['temp_dew_spread'].diff(6)

    for m in MODELS:
        wd = f"{m}_wind_direction_10m_model"
        ws = f"{m}_wind_speed_10m_model"
        if wd in out.columns and ws in out.columns:
            d = pd.to_numeric(out[wd], errors='coerce')
            s = pd.to_numeric(out[ws], errors='coerce')
            # Widened bura detection: full NE quadrant 0-90° at 7 m/s
            out[f'{m}_bura'] = (((d >= 315) | (d <= 90)) & (s >= 7)).astype(float)
    bura_cols = [f'{m}_bura' for m in MODELS if f'{m}_bura' in out.columns]
    if bura_cols:
        out['bura_agreement'] = out[bura_cols].sum(axis=1)

    # --- Wind u/v components (report A1) ---
    # Raw direction in degrees wraps at 0/360, which trees split badly and which
    # makes naive averaging wrong. Decompose to u/v (meteorological convention:
    # direction is where the wind comes FROM) and build a circular-correct
    # ensemble mean + a directional-consistency ratio (vector speed / scalar
    # speed: 1 = all models agree on direction, <1 = directional disagreement).
    u_cols, v_cols = [], []
    for m in MODELS:
        wd = f"{m}_wind_direction_10m_model"
        ws = f"{m}_wind_speed_10m_model"
        if wd in out.columns and ws in out.columns:
            d = np.radians(pd.to_numeric(out[wd], errors='coerce'))
            s = pd.to_numeric(out[ws], errors='coerce')
            out[f'{m}_wind_u'] = -s * np.sin(d)
            out[f'{m}_wind_v'] = -s * np.cos(d)
            u_cols.append(f'{m}_wind_u')
            v_cols.append(f'{m}_wind_v')
    if u_cols:
        out['wind_u_ens_mean'] = out[u_cols].mean(axis=1)
        out['wind_v_ens_mean'] = out[v_cols].mean(axis=1)
        out['wind_speed_vec_ens'] = np.sqrt(out['wind_u_ens_mean'] ** 2 + out['wind_v_ens_mean'] ** 2)
        out['wind_dir_vec_ens'] = (np.degrees(np.arctan2(-out['wind_u_ens_mean'],
                                                          -out['wind_v_ens_mean'])) % 360)
        if 'wind_speed_10m_ens_mean' in out.columns:
            out['wind_dir_consistency'] = (
                out['wind_speed_vec_ens'] / out['wind_speed_10m_ens_mean'].clip(lower=0.1)
            ).clip(0, 1)

    if rain_mcols:
        rain_vals = out[rain_mcols].apply(pd.to_numeric, errors='coerce')
        ens_precip = rain_vals.mean(axis=1)

        out['precip_running_6h'] = ens_precip.rolling(6, min_periods=1).sum()
        out['precip_running_12h'] = ens_precip.rolling(12, min_periods=1).sum()
        out['precip_running_24h'] = ens_precip.rolling(24, min_periods=1).sum()

        (_wet_model_count, available_model_count, agreement,
         rain_hours) = _rain_consensus_stats(rain_vals)
        # Count consensus-wet *hours*, not wet model votes. Previously three
        # wet members in one timestamp were mislabelled as three rain hours.
        out['rain_hours_6h'] = rain_hours.rolling(6, min_periods=1).sum()
        out['rain_hours_12h'] = rain_hours.rolling(12, min_periods=1).sum()
        out['rain_hours_24h'] = rain_hours.rolling(24, min_periods=1).sum()

        out['rain_agreement_6h'] = agreement.rolling(6, min_periods=1).mean()
        out['rain_agreement_12h'] = agreement.rolling(12, min_periods=1).mean()

        out['persistent_rain'] = (
            (out['rain_hours_6h'] >= 3) &
            (agreement >= 0.5)
        ).astype(float)

        out['sustained_rain_12h'] = (
            (out['rain_hours_12h'] >= 6) &
            (out['rain_agreement_12h'] >= 0.4)
        ).astype(float)

    is_winter = out['month'].isin([11, 12, 1, 2, 3]).astype(float)
    out['is_winter'] = is_winter

    if 'temp_dew_spread' in out.columns:
        out['dew_saturated'] = (out['temp_dew_spread'] < 2.0).astype(float)
        if rain_mcols:
            out['winter_rain_signal'] = (
                is_winter *
                out.get('persistent_rain', 0) *
                out['dew_saturated']
            )

    if 'relative_humidity_2m_ens_mean' in out.columns:
        rh = out['relative_humidity_2m_ens_mean']
        out['humidity_above_90'] = (rh > 90).astype(float)
        out['humidity_above_90_6h'] = out['humidity_above_90'].rolling(6, min_periods=1).sum()

        if rain_mcols:
            out['humid_rain_persistence'] = (
                out['humidity_above_90_6h'] *
                out.get('rain_agreement_6h', 0)
            )

    if 'cloud_cover_ens_mean' in out.columns:
        cc = out['cloud_cover_ens_mean']
        out['overcast_6h'] = (cc > 80).rolling(6, min_periods=1).mean()
        out['overcast_12h'] = (cc > 80).rolling(12, min_periods=1).mean()

    if 'precipitation_ens_mean' in out.columns:
        pem = out['precipitation_ens_mean']
        out['precip_ens_running_6h'] = pem.rolling(6, min_periods=1).sum()
        out['precip_ens_nonzero_6h'] = (pem > 0.05).rolling(6, min_periods=1).sum()
        out['precip_ens_nonzero_12h'] = (pem > 0.05).rolling(12, min_periods=1).sum()
        out['precip_ens_intensity'] = pem.rolling(6, min_periods=1).mean()

    if 'precipitation_ens_std' in out.columns:
        out['precip_model_certainty'] = 1.0 / (1.0 + out['precipitation_ens_std'])

    # ICON-2I MICROFISICA-NEW regime change flag.
    # ItaliaMeteo confirmed on 26-May-2025 that pre-fix ICON-2I systematically
    # overestimated weakly-forced summer convection. Training data straddles
    # both regimes; let XGB learn the bias change.
    out['icon2i_era'] = (out['datetime'] >= pd.Timestamp('2025-05-26')).astype(float)

    # --- stability/synoptic features (require stability CSVs) ---
    # CAPE: ensemble across all models that have it.
    cape_cols = [f"{m}_cape_model" for m in MODELS if f"{m}_cape_model" in out.columns]
    if cape_cols:
        cape_vals = out[cape_cols].apply(pd.to_numeric, errors='coerce')
        out['cape_ens_mean'] = cape_vals.mean(axis=1)
        out['cape_ens_max'] = cape_vals.max(axis=1)
        out['cape_ens_std'] = cape_vals.std(axis=1)

    # CIN: only 4 models have it (GFS, ITALIA, DMI, UKMO).
    cin_cols = [f"{m}_convective_inhibition_model" for m in MODELS
                if f"{m}_convective_inhibition_model" in out.columns]
    if cin_cols:
        cin_vals = out[cin_cols].apply(pd.to_numeric, errors='coerce')
        # APIs/models differ on CIN sign convention.  Inhibition strength is a
        # magnitude, so normalize with abs() before comparing thresholds.
        cin_strength = cin_vals.abs()
        out['cin_ens_mean'] = cin_strength.mean(axis=1)
        # Low CIN means little cap despite high CAPE.
        out['low_cin_indicator'] = (
            out['cin_ens_mean'] <= CONVECTIVE_CIN_MAX_JKG
        ).astype(float)
        if 'cape_ens_mean' in out.columns:
            # CAPE × low_CIN_indicator = genuine triggering potential
            out['cape_x_low_cin'] = out['cape_ens_mean'] * out['low_cin_indicator']

    # Lifted index: only GFS. LI < -2 with low CIN = real convection.
    li_col = "GFS_SEAMLESS_lifted_index_model"
    if li_col in out.columns:
        out['lifted_index_gfs'] = pd.to_numeric(out[li_col], errors='coerce')

    # Synoptic forcing: mean wind at 500-700hPa.
    # low values (<5 m/s) flag weakly-forced regime = worst-FAR.
    # Open-Meteo returns wind in km/h by default; convert to m/s.
    KMH_TO_MS = 1.0 / 3.6
    w500_cols = [f"{m}_wind_speed_500hPa_model" for m in MODELS
                 if f"{m}_wind_speed_500hPa_model" in out.columns]
    w700_cols = [f"{m}_wind_speed_700hPa_model" for m in MODELS
                 if f"{m}_wind_speed_700hPa_model" in out.columns]
    if w500_cols:
        out['wind_500_ens_mean'] = out[w500_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1) * KMH_TO_MS
    if w700_cols:
        out['wind_700_ens_mean'] = out[w700_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1) * KMH_TO_MS
    if w500_cols and w700_cols:
        out['mean_wind_500_700'] = (out['wind_500_ens_mean'] + out['wind_700_ens_mean']) / 2
        # Weakly-forced flag: low synoptic wind (now correctly in m/s)
        out['weakly_forced_regime'] = (out['mean_wind_500_700'] < 5.0).astype(float)

    # Precipitable water (ECMWF only).
    pw_col = "ECMWF_IFS025_total_column_integrated_water_vapour_model"
    if pw_col in out.columns:
        out['precipitable_water_ecmwf'] = pd.to_numeric(out[pw_col], errors='coerce')

    # Convection regime classifier:
    # "high CAPE + low ensemble agreement + low synoptic forcing" = canonical
    # FAR pattern (models hallucinate convection that won't materialize).
    if 'cape_ens_mean' in out.columns and 'mean_wind_500_700' in out.columns:
        hallucination_regime = (
            (out['cape_ens_mean'] > 500) &
            (out['mean_wind_500_700'] < 5.0) &
            (out.get('rain_agreement', pd.Series(0, index=out.index)) < 0.30)
        )
        out['hallucination_convection_flag'] = hallucination_regime.astype(float)

    # High CAPE + strong CIN (capping) + low ensemble
    # agreement = the OTHER canonical "models hallucinate" pattern. Strong
    # CIN means a thermal cap prevents convection from triggering despite
    # plenty of CAPE; without an external lift mechanism (front, orographic),
    # the storm never fires. Complementary to weakly_forced_regime which
    # captures lack of synoptic forcing rather than thermodynamic capping.
    if 'cape_ens_mean' in out.columns and 'cin_ens_mean' in out.columns:
        out['high_cin_indicator'] = (out['cin_ens_mean'] > 100).astype(float)
        out['hallucination_via_cin_flag'] = (
            (out['cape_ens_mean'] > 500) &
            (out['cin_ens_mean'] > 100) &
            (out.get('rain_agreement', pd.Series(0, index=out.index)) < 0.30)
        ).astype(float)

    # dry_spell_length: hours since last observed rain.
    # Long dry spells in summer should raise the bar for predicting onset.
    # Must not use the current hour's obs: that would leak the target.
    # We shift wet by 1 so the count is "consecutive dry hours immediately
    # preceding hour t (not including t itself)". For a wet hour t this
    # reports the length of the dry spell that just ENDED — informative,
    # not leaking.
    if '_derived_precip_obs' in out.columns:
        precip_obs = pd.to_numeric(out['_derived_precip_obs'], errors='coerce').fillna(0)
        wet = (precip_obs >= 0.1)
        wet_lag = wet.shift(1).fillna(False)
        grp = wet_lag.cumsum()
        out['dry_spell_length'] = wet_lag.groupby(grp).cumcount().clip(upper=168).astype(float)

    # Monthly climatological rain frequency.
    # P(rain | month, hour) from training data only. Strong corrector of
    # summer over-forecasts. Computed on train portion (datetime < SPLIT_DATE)
    # to avoid leakage, then broadcast to all rows by (month, hour) lookup.
    if '_derived_precip_obs' in out.columns:
        train_mask = out['datetime'] < SPLIT_DATE
        obs_tr = pd.to_numeric(out.loc[train_mask, '_derived_precip_obs'],
                                errors='coerce').fillna(0)
        rain_ind_tr = (obs_tr >= 0.1).astype(float)
        clim_df = pd.DataFrame({
            'month': out.loc[train_mask, 'month'].values,
            'hour':  out.loc[train_mask, 'hour'].values,
            'rain':  rain_ind_tr.values,
        })
        clim_table = clim_df.groupby(['month', 'hour'])['rain'].mean()
        keys = list(zip(out['month'].astype(int), out['hour'].astype(int)))
        out['monthly_clim_rain_freq'] = pd.Series(
            [clim_table.get(k, np.nan) for k in keys], index=out.index
        )

    if 'pressure_msl_ens_mean' in out.columns:
        pres = out['pressure_msl_ens_mean']

        pres_change_3h = pres.diff(3)
        pres_change_6h = pres.diff(6)
        pres_change_12h = pres.diff(12)

        out['pres_change_12h'] = pres_change_12h

        out['pres_rapidly_falling'] = (pres_change_3h < -3.0).astype(float)
        out['pres_falling'] = (pres_change_3h < -1.0).astype(float)
        out['pres_rising'] = (pres_change_3h > 1.0).astype(float)
        out['pres_rapidly_rising'] = (pres_change_3h > 3.0).astype(float)

        out['pres_anomaly'] = pres - 1015.0

        out['low_pressure_regime'] = (pres < 1010.0).astype(float)
        out['very_low_pressure'] = (pres < 1005.0).astype(float)
        out['high_pressure_regime'] = (pres > 1020.0).astype(float)

        out['pres_stability_6h'] = pres.rolling(6, min_periods=2).std()
        out['pres_stability_12h'] = pres.rolling(12, min_periods=3).std()

        if 'pressure_msl_ens_std' in out.columns:
            out['pres_model_disagreement'] = out['pressure_msl_ens_std']
            out['pres_high_uncertainty'] = (out['pressure_msl_ens_std'] > 1.5).astype(float)

        if 'cloud_cover_ens_mean' in out.columns:
            out['frontal_signal'] = (
                (pres_change_6h < -2.0) &
                (out['cloud_cover_ens_mean'] > 70)
            ).astype(float)

            if rain_mcols:
                out['active_front'] = (
                    out['frontal_signal'] *
                    out.get('rain_agreement', 0)
                )

    if 'relative_humidity_2m_ens_mean' in out.columns:
        rh = out['relative_humidity_2m_ens_mean']

        out['humidity_above_80'] = (rh > 80).astype(float)
        out['humidity_above_95'] = (rh > 95).astype(float)
        out['sustained_humid_6h'] = out['humidity_above_80'].rolling(6, min_periods=1).sum()
        out['sustained_humid_12h'] = out['humidity_above_80'].rolling(12, min_periods=1).sum()
        out['sustained_humid_24h'] = out['humidity_above_80'].rolling(24, min_periods=1).sum()

        out['rh_tend_1h'] = rh.diff(1)
        out['rh_tend_3h'] = rh.diff(3)
        out['rh_tend_6h'] = rh.diff(6)
        out['rh_rising'] = (out['rh_tend_3h'] > 3.0).astype(float)
        out['rh_falling'] = (out['rh_tend_3h'] < -3.0).astype(float)

        if 'relative_humidity_2m_ens_std' in out.columns:
            out['rh_model_disagreement'] = out['relative_humidity_2m_ens_std']

        if 'cloud_cover_ens_mean' in out.columns:
            cc = out['cloud_cover_ens_mean']
            out['humid_overcast'] = (rh * cc / 100.0)
            out['humid_overcast_flag'] = ((rh > 80) & (cc > 80)).astype(float)
            out['dry_clear_flag'] = ((rh < 50) & (cc < 30)).astype(float)

        if 'wind_speed_10m_ens_mean' in out.columns:
            out['rh_wind_interaction'] = rh / (1.0 + out['wind_speed_10m_ens_mean'])

        out['night_humid'] = (
            out.get('is_daytime', pd.Series(0, index=out.index)).eq(0) &
            (rh > 75)
        ).astype(float)

    if 'temperature_2m_ens_mean' in out.columns and 'dew_point_2m_ens_mean' in out.columns:
        temp = out['temperature_2m_ens_mean']
        dew = out['dew_point_2m_ens_mean']
        spread = temp - dew

        out['near_saturation'] = (spread < 1.5).astype(float)
        out['moderate_spread'] = ((spread >= 1.5) & (spread < 5.0)).astype(float)
        out['dry_spread'] = (spread >= 8.0).astype(float)

        out['near_sat_6h'] = out['near_saturation'].rolling(6, min_periods=1).sum()
        out['near_sat_12h'] = out['near_saturation'].rolling(12, min_periods=1).sum()

        if 'wind_speed_10m_ens_mean' in out.columns:
            out['fog_risk'] = (
                (spread < 2.0) &
                (out['wind_speed_10m_ens_mean'] < 3.0)
            ).astype(float)

        out['spread_tend_3h'] = spread.diff(3)
        out['spread_closing'] = (out['spread_tend_3h'] < -1.0).astype(float)
        out['spread_opening'] = (out['spread_tend_3h'] > 1.0).astype(float)

    if 'temperature_2m_ens_mean' in out.columns:
        temp = out['temperature_2m_ens_mean']

        if 'temperature_2m_ens_std' in out.columns:
            out['temp_high_uncertainty'] = (out['temperature_2m_ens_std'] > 1.0).astype(float)

        if 'is_daytime' in out.columns:
            out['dtr_proxy'] = temp.rolling(24, min_periods=6).max() - temp.rolling(24, min_periods=6).min()

            if 'cloud_cover_ens_mean' in out.columns:
                out['dtr_x_cloud'] = out['dtr_proxy'] * (1.0 - out['cloud_cover_ens_mean'] / 100.0)

        out['temp_near_zero'] = ((temp > -2.0) & (temp < 5.0)).astype(float)

        if 'precipitation_ens_mean' in out.columns:
            pem = out['precipitation_ens_mean']
            out['temp_x_precip'] = temp * pem.clip(upper=5.0)
            out['cold_rain'] = ((temp < 8.0) & (pem > 0.1)).astype(float)
            out['warm_rain'] = ((temp > 20.0) & (pem > 0.1)).astype(float)

        month = out['month']
        sea_temp_approx = 13.0 + 6.0 * np.sin(2 * np.pi * (month - 3) / 12)
        out['sea_air_diff'] = sea_temp_approx - temp
        out['marine_warming'] = (out['sea_air_diff'] > 3.0).astype(float)  # sea warms air
        out['marine_cooling'] = (out['sea_air_diff'] < -3.0).astype(float)  # sea cools air

    if all(c in out.columns for c in ['is_winter', 'cloud_cover_ens_mean',
           'relative_humidity_2m_ens_mean']):
        cc = out['cloud_cover_ens_mean']
        rh = out['relative_humidity_2m_ens_mean']

        winter_overcast = (
            (out['is_winter'] > 0) &
            (cc > 75) &
            (rh > 70)
        ).astype(float)
        out['winter_overcast_regime'] = winter_overcast

        out['winter_overcast_6h'] = winter_overcast.rolling(6, min_periods=1).sum()
        out['winter_overcast_12h'] = winter_overcast.rolling(12, min_periods=1).sum()

        if 'precipitation_ens_mean' in out.columns:
            pem = out['precipitation_ens_mean']
            out['winter_overcast_rain'] = (
                winter_overcast *
                (pem > 0.1).astype(float)
            )
            out['winter_overcast_rain_12h'] = out['winter_overcast_rain'].rolling(12, min_periods=1).sum()

        if 'pressure_msl_ens_mean' in out.columns:
            pres = out['pressure_msl_ens_mean']
            out['winter_pres_above_1020'] = (
                (out['is_winter'] > 0) &
                (pres > 1020.0)
            ).astype(float)

            out['winter_low_pres_rain'] = (
                (out['is_winter'] > 0) &
                (pres < 1010.0) &
                (out.get('rain_agreement', pd.Series(0, index=out.index)) > 0.3)
            ).astype(float)

    pres_mcols = [f"{m}_pressure_msl_model" for m in MODELS
                  if f"{m}_pressure_msl_model" in out.columns]
    if len(pres_mcols) >= 2 and 'pressure_msl_ens_mean' in out.columns:
        pres_ens = out['pressure_msl_ens_mean']
        for m in MODELS:
            pc = f"{m}_pressure_msl_model"
            if pc in out.columns:
                dev = pd.to_numeric(out[pc], errors='coerce') - pres_ens
                out[f'{m}_pres_bias'] = dev

        pres_vals = out[pres_mcols].apply(pd.to_numeric, errors='coerce')
        out['pres_max_spread'] = pres_vals.max(axis=1) - pres_vals.min(axis=1)

    rh_mcols = [f"{m}_relative_humidity_2m_model" for m in MODELS
                if f"{m}_relative_humidity_2m_model" in out.columns]
    if len(rh_mcols) >= 2 and 'relative_humidity_2m_ens_mean' in out.columns:
        rh_ens = out['relative_humidity_2m_ens_mean']
        for m in MODELS:
            rhc = f"{m}_relative_humidity_2m_model"
            if rhc in out.columns:
                dev = pd.to_numeric(out[rhc], errors='coerce') - rh_ens
                out[f'{m}_rh_bias'] = dev

        rh_vals = out[rh_mcols].apply(pd.to_numeric, errors='coerce')
        out['rh_max_spread'] = rh_vals.max(axis=1) - rh_vals.min(axis=1)

        out['rh_above85_count'] = (rh_vals > 85).sum(axis=1)
        out['rh_above85_ratio'] = out['rh_above85_count'] / len(rh_mcols)

    if all(c in out.columns for c in ['cloud_cover_ens_mean',
           'relative_humidity_2m_ens_mean', 'shortwave_radiation_ens_mean']):
        cc = out['cloud_cover_ens_mean']
        rh = out['relative_humidity_2m_ens_mean']
        sw = out['shortwave_radiation_ens_mean']
        clear = out.get('clear_sky_rad', pd.Series(1, index=out.index))

        out['cloud_rh_inconsistent'] = ((cc > 80) & (rh < 60)).astype(float)

        out['humid_clear_sky'] = ((cc < 30) & (rh > 85)).astype(float)

        solar_ratio = sw / clear.clip(lower=1)
        out['solar_cloud_mismatch'] = (
            (cc > 80) & (solar_ratio > 0.5) |
            (cc < 20) & (solar_ratio < 0.3)
        ).astype(float)

    if 'precipitation_ens_mean' in out.columns:
        pem = out['precipitation_ens_mean']

        out['precip_24h_total'] = pem.rolling(24, min_periods=1).sum()
        out['precip_48h_total'] = pem.rolling(48, min_periods=1).sum()

        out['heavy_rain_event'] = (pem > 3.0).astype(float)
        out['heavy_rain_6h'] = out['heavy_rain_event'].rolling(6, min_periods=1).sum()

        out['precip_decreasing'] = (pem.diff(3) < -0.5).astype(float)
        out['post_rain_clearing'] = (
            (out['precip_24h_total'] > 5.0) &
            (pem < 0.1) &
            out['precip_decreasing'].astype(bool)
        ).astype(float)

    wd_mcols = [f"{m}_wind_direction_10m_model" for m in MODELS
                if f"{m}_wind_direction_10m_model" in out.columns]
    ws_mcols = [f"{m}_wind_speed_10m_model" for m in MODELS
                if f"{m}_wind_speed_10m_model" in out.columns]
    if wd_mcols and ws_mcols:
        wd_vals = out[wd_mcols].apply(pd.to_numeric, errors='coerce')
        ws_vals = out[ws_mcols].apply(pd.to_numeric, errors='coerce')

        wd_mean = np.degrees(np.arctan2(
            np.sin(np.radians(wd_vals)).mean(axis=1),
            np.cos(np.radians(wd_vals)).mean(axis=1)
        )) % 360
        ws_mean = ws_vals.mean(axis=1)

        out['is_jugo'] = (
            ((wd_mean >= 100) & (wd_mean <= 170)) &
            (ws_mean > 5.0)
        ).astype(float)

        out['is_maestral'] = (
            ((wd_mean >= 280) & (wd_mean <= 340)) &
            (ws_mean > 3.0) &
            (out['month'].isin([5, 6, 7, 8, 9]).astype(float) > 0)
        ).astype(float)

        out['jugo_6h'] = out['is_jugo'].rolling(6, min_periods=1).sum()

        out['winter_jugo'] = (
            out['is_jugo'] * out.get('is_winter', pd.Series(0, index=out.index))
        )

    for v in PREV_RUNS_VARS:
        day1_cols = [f"{m}_{v}_previous_day1" for m in PREV_RUNS_MODELS
                     if f"{m}_{v}_previous_day1" in out.columns]
        day2_cols = [f"{m}_{v}_previous_day2" for m in PREV_RUNS_MODELS
                     if f"{m}_{v}_previous_day2" in out.columns]

        if len(day1_cols) >= 2:
            d1_vals = out[day1_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_prev_day1_ens_mean'] = d1_vals.mean(axis=1)
            out[f'{v}_prev_day1_ens_std'] = d1_vals.std(axis=1)

        if len(day2_cols) >= 2:
            d2_vals = out[day2_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_prev_day2_ens_mean'] = d2_vals.mean(axis=1)

        rev_cols = []
        for m in PREV_RUNS_MODELS:
            d0 = f"{m}_{v}_model"
            d1 = f"{m}_{v}_previous_day1"
            if d0 in out.columns and d1 in out.columns:
                col_name = f'{m}_{v}_revision'
                out[col_name] = (pd.to_numeric(out[d0], errors='coerce') -
                                 pd.to_numeric(out[d1], errors='coerce'))
                rev_cols.append(col_name)

        d1d2_rev_cols = []
        for m in PREV_RUNS_MODELS:
            d1 = f"{m}_{v}_previous_day1"
            d2 = f"{m}_{v}_previous_day2"
            if d1 in out.columns and d2 in out.columns:
                col_name = f'{m}_{v}_d1d2_revision'
                out[col_name] = (pd.to_numeric(out[d1], errors='coerce') -
                                 pd.to_numeric(out[d2], errors='coerce'))
                d1d2_rev_cols.append(col_name)

        if len(rev_cols) >= 2:
            rv = out[rev_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_revision_ens_mean'] = rv.mean(axis=1)
            out[f'{v}_revision_ens_std'] = rv.std(axis=1)
            out[f'{v}_revision_ens_abs_mean'] = rv.abs().mean(axis=1)

        if len(d1d2_rev_cols) >= 2:
            d1d2 = out[d1d2_rev_cols].apply(pd.to_numeric, errors='coerce')
            out[f'{v}_d1d2_revision_ens_mean'] = d1d2.mean(axis=1)

        if f'{v}_prev_day1_ens_mean' in out.columns and f'{v}_ens_mean' in out.columns:
            out[f'{v}_day0_vs_day1_ens'] = out[f'{v}_ens_mean'] - out[f'{v}_prev_day1_ens_mean']

    for param in ['temperature_2m', 'dew_point_2m', 'pressure_msl', 'wind_speed_10m',
                  'relative_humidity_2m', 'cloud_cover']:
        ens_col = f'{param}_ens_mean'
        if ens_col in out.columns:
            ser = out[ens_col]
            out[f'{param}_ens_lag1'] = ser.shift(1)
            out[f'{param}_ens_lag3'] = ser.shift(3)
            out[f'{param}_ens_ma6'] = ser.rolling(6, min_periods=1).mean()
            out[f'{param}_ens_ma12'] = ser.rolling(12, min_periods=1).mean()
            out[f'{param}_ens_ma24'] = ser.rolling(24, min_periods=1).mean()
            out[f'{param}_ens_std6'] = ser.rolling(6, min_periods=2).std()
            out[f'{param}_ens_std24'] = ser.rolling(24, min_periods=3).std()
            out[f'{param}_ens_anom24'] = ser - out[f'{param}_ens_ma24']

    if 'precipitation_ens_mean' in out.columns:
        pem = out['precipitation_ens_mean']
        out['precip_sqrt'] = np.sqrt(pem.clip(lower=0))
        out['precip_log1p'] = np.log1p(pem.clip(lower=0))
        out['precip_is_zero'] = (pem < 0.05).astype(float)
        out['precip_dry_hours'] = out['precip_is_zero'].rolling(12, min_periods=1).sum()

    for param in ['temperature_2m', 'precipitation', 'wind_speed_10m', 'cloud_cover']:
        mcols = [f"{m}_{param}_model" for m in MODELS if f"{m}_{param}_model" in out.columns]
        if len(mcols) >= 4:
            vals = out[mcols].apply(pd.to_numeric, errors='coerce')
            q25 = vals.quantile(0.25, axis=1)
            q75 = vals.quantile(0.75, axis=1)
            out[f'{param}_ens_iqr'] = q75 - q25
            out[f'{param}_ens_skew'] = vals.skew(axis=1)

    if 'hour_sin' in out.columns and 'season' in out.columns:
        out['hour_sin_x_season'] = out['hour_sin'] * out['season']
        out['hour_cos_x_season'] = out['hour_cos'] * out['season']
    if 'hour_sin' in out.columns and 'doy_sin' in out.columns:
        out['hour_x_doy'] = out['hour_sin'] * out['doy_sin']

    if 'temperature_2m_ens_std' in out.columns and 'temperature_2m_ens_mean' in out.columns:
        out['temp_cv'] = out['temperature_2m_ens_std'] / (out['temperature_2m_ens_mean'].abs().clip(lower=0.1))

    # Multi-factor bias interactions: model errors correlate with other
    # conditions, and multi-factor NWP bias correction beats single-factor.

    # Temperature bias conditioned on humidity regime
    for m in MODELS:
        t_bias = f'{m}_temperature_2m_hist_bias'
        rh_col = f'{m}_relative_humidity_2m_model'
        p_col = f'{m}_pressure_msl_model'
        cc_col = f'{m}_cloud_cover_model'

        if t_bias in out.columns and rh_col in out.columns:
            rh_v = pd.to_numeric(out[rh_col], errors='coerce')
            out[f'{m}_temp_bias_x_humid'] = out[t_bias] * (rh_v / 100.0).fillna(0.5)
        if t_bias in out.columns and cc_col in out.columns:
            cc_v = pd.to_numeric(out[cc_col], errors='coerce')
            out[f'{m}_temp_bias_x_cloud'] = out[t_bias] * (cc_v / 100.0).fillna(0.5)
        if t_bias in out.columns and p_col in out.columns:
            p_v = pd.to_numeric(out[p_col], errors='coerce')
            out[f'{m}_temp_bias_x_pres'] = out[t_bias] * ((p_v - 1013.25) / 20.0).fillna(0)

    # Ensemble disagreement × bias magnitude (high disagreement + high bias → less trustworthy)
    for param in ['temperature_2m', 'pressure_msl', 'wind_speed_10m', 'cloud_cover']:
        std_col = f'{param}_ens_std'
        if std_col in out.columns:
            bias_cols = [f'{m}_{param}_hist_bias' for m in MODELS if f'{m}_{param}_hist_bias' in out.columns]
            if bias_cols:
                mean_abs_bias = out[bias_cols].abs().mean(axis=1)
                out[f'{param}_disagree_x_bias'] = out[std_col] * mean_abs_bias

    # Diurnal bias pattern: bias tends to be systematic at certain hours
    for param in ['temperature_2m', 'cloud_cover', 'shortwave_radiation']:
        bias_cols = [f'{m}_{param}_hist_bias' for m in MODELS if f'{m}_{param}_hist_bias' in out.columns]
        if bias_cols and 'hour_sin' in out.columns:
            mean_bias = out[bias_cols].mean(axis=1)
            out[f'{param}_bias_x_hour_sin'] = mean_bias * out['hour_sin']
            out[f'{param}_bias_x_hour_cos'] = mean_bias * out['hour_cos']

    # Extra ensemble statistics to enrich the feature representation.
    for param in ['temperature_2m', 'dew_point_2m', 'wind_speed_10m', 'pressure_msl',
                  'cloud_cover', 'relative_humidity_2m']:
        mcols = [f"{m}_{param}_model" for m in MODELS if f"{m}_{param}_model" in out.columns]
        if len(mcols) >= 4:
            vals = out[mcols].apply(pd.to_numeric, errors='coerce')
            # Kurtosis: measures tail heaviness of model distribution
            out[f'{param}_ens_kurtosis'] = vals.kurtosis(axis=1)
            # Coefficient of variation
            ens_mean_col = f'{param}_ens_mean'
            if ens_mean_col in out.columns:
                out[f'{param}_ens_cv'] = out.get(f'{param}_ens_std', vals.std(axis=1)) / out[ens_mean_col].abs().clip(lower=0.01)
            # Ratio of range to IQR (measures outlier severity)
            iqr_col = f'{param}_ens_iqr'
            range_col = f'{param}_ens_range'
            if iqr_col in out.columns and range_col in out.columns:
                out[f'{param}_range_iqr_ratio'] = out[range_col] / out[iqr_col].clip(lower=0.01)

    # Lag-error features: error_lag_k = obs[t-k] - forecast[t-k]. Recent
    # error tends to persist, so these carry short-range correction.
    for param in ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
                  'wind_speed_10m', 'pressure_msl', 'cloud_cover']:
        obs_col = f'{param}_obs'
        ens_col = f'{param}_ens_mean'
        if obs_col in out.columns and ens_col in out.columns:
            obs_vals = pd.to_numeric(out[obs_col], errors='coerce')
            ens_vals = pd.to_numeric(out[ens_col], errors='coerce')
            error_series = obs_vals - ens_vals
            for lag in [1, 3, 6, 24]:
                out[f'{param}_error_lag{lag}'] = error_series.shift(lag)
            # Running mean of PAST errors (shift(1) excludes current timestep to prevent target leakage)
            out[f'{param}_error_ma6'] = error_series.shift(1).rolling(6, min_periods=1).mean()
            out[f'{param}_error_ma24'] = error_series.shift(1).rolling(24, min_periods=1).mean()

    # SST features: SST moderates coastal Budva temps; the land-sea
    # gradient drives sea breeze / onshore flow.
    if 'sst' in out.columns:
        sst = pd.to_numeric(out['sst'], errors='coerce')
        out['sst_ma24'] = sst.rolling(24, min_periods=1).mean()
        out['sst_tendency_24h'] = sst.diff(24)
        # Climatological SST anomaly (rough: SST - 30-day running mean)
        out['sst_anomaly'] = sst - sst.rolling(720, min_periods=24).mean()
        # Land-sea temperature gradient — drives onshore/offshore flow
        if 'temperature_2m_ens_mean' in out.columns:
            out['land_sea_gradient'] = pd.to_numeric(out['temperature_2m_ens_mean'], errors='coerce') - sst
            out['land_sea_gradient_abs'] = out['land_sea_gradient'].abs()

    # Kalman-style bias tracking: EWMA of model error; Q (process noise)
    # and R (observation noise) control the filter gain.
    # Higher Q → more responsive; higher R → smoother. We use Q/R ≈ 0.1 for stability.
    for param in ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
                  'wind_speed_10m', 'pressure_msl', 'cloud_cover']:
        obs_col = f'{param}_obs'
        ens_col = f'{param}_ens_mean'
        if obs_col not in out.columns or ens_col not in out.columns:
            continue
        obs_vals = pd.to_numeric(out[obs_col], errors='coerce').values
        ens_vals = pd.to_numeric(out[ens_col], errors='coerce').values
        innovation = obs_vals - ens_vals  # observation - forecast = error

        # Simple Kalman filter (scalar): x_k = x_{k-1} + K*(obs - x_{k-1})
        Q, R = 0.1, 1.0  # process / measurement noise
        x = 0.0  # initial state (no bias)
        P = 1.0  # initial covariance
        kalman_bias = np.full(len(out), np.nan)
        for i in range(len(innovation)):
            kalman_bias[i] = x  # store PRIOR (before absorbing obs[i]) to prevent target leakage
            if not np.isnan(innovation[i]):
                P_pred = P + Q
                K = P_pred / (P_pred + R)
                x = x + K * (innovation[i] - x)
                P = (1 - K) * P_pred
        out[f'{param}_kalman_bias'] = kalman_bias

    # Monograph-2 features (neighborhood, bura gradient, sea breeze,
    # NWP-proxy regimes). Computed identically at train and inference time
    # because both paths run through this function.

    # Neighborhood 5x5 precipitation stats.
    # Columns {M}_nbr_p00..p24 come from neighborhood_data CSVs (training) or
    # the live multi-point fetch. Grid is row-major, lats S->N, lons W->E;
    # Budva itself is p12. "Upwind" = the western band (p05,p06,p10,p11,p15,p16)
    # — the dominant approach corridor for Adriatic rain bands.
    _NBR_UPWIND = ['p05', 'p06', 'p10', 'p11', 'p15', 'p16']
    _nbr_fwet_cols, _nbr_max_cols = [], []
    for m in MODELS:
        mcols = [f'{m}_nbr_p{i:02d}' for i in range(25) if f'{m}_nbr_p{i:02d}' in out.columns]
        if len(mcols) < 20:
            continue
        nv = out[mcols].apply(pd.to_numeric, errors='coerce')
        out[f'{m}_nbr_mean'] = nv.mean(axis=1)
        out[f'{m}_nbr_max'] = nv.max(axis=1)
        out[f'{m}_nbr_fwet'] = (nv >= 0.1).mean(axis=1)
        up = [f'{m}_nbr_{p}' for p in _NBR_UPWIND if f'{m}_nbr_{p}' in out.columns]
        if up:
            out[f'{m}_nbr_upwind_mean'] = out[up].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        _nbr_fwet_cols.append(f'{m}_nbr_fwet')
        _nbr_max_cols.append(f'{m}_nbr_max')
    if _nbr_fwet_cols:
        out['nbr_ens_fwet'] = out[_nbr_fwet_cols].mean(axis=1)
        out['nbr_ens_max'] = out[_nbr_max_cols].max(axis=1)

    # --- Wind-direction sector fractions (direction-only, unlike {m}_bura
    # which also requires >=7 m/s) ---
    _ne_flags, _onshore_flags = [], []
    for m in MODELS:
        wd = f"{m}_wind_direction_10m_model"
        if wd in out.columns:
            d = pd.to_numeric(out[wd], errors='coerce')
            _ne_flags.append(((d >= 315) | (d <= 90)).astype(float).where(d.notna()))
            # Budva coastline faces SSW: onshore (sea->land) sector ~120-250 deg
            _onshore_flags.append(((d >= 120) & (d <= 250)).astype(float).where(d.notna()))
    if _ne_flags:
        out['ne_sector_frac'] = pd.concat(_ne_flags, axis=1).mean(axis=1)
        out['onshore_frac'] = pd.concat(_onshore_flags, axis=1).mean(axis=1)
        if 'is_daytime' in out.columns:
            out['sea_breeze_ix'] = out['onshore_frac'] * out['is_daytime']

    # Transdinaric pressure gradient (bura forcing).
    # Positive gradient (Podgorica > Budva MSL) pushes air over the Lovcen
    # ridge -> katabatic NE flow on the coast.
    _grad_cols = []
    for m in MODELS:
        pg = f'{m}_pressure_msl_pg_model'
        bu = f'{m}_pressure_msl_model'
        if pg in out.columns and bu in out.columns:
            out[f'{m}_mslp_grad'] = (pd.to_numeric(out[pg], errors='coerce')
                                     - pd.to_numeric(out[bu], errors='coerce'))
            _grad_cols.append(f'{m}_mslp_grad')
    if _grad_cols:
        out['mslp_grad_ens_mean'] = out[_grad_cols].mean(axis=1)
        out['mslp_grad_ens_std'] = out[_grad_cols].std(axis=1)
        if 'ne_sector_frac' in out.columns:
            out['mslp_grad_x_ne'] = out['mslp_grad_ens_mean'] * out['ne_sector_frac']

    # Gust factor G = U_gust / U_mean, stability-conditioned.
    _gf_cols = []
    for m in MODELS:
        gu = f'{m}_wind_gusts_10m_model'
        sp = f'{m}_wind_speed_10m_model'
        if gu in out.columns and sp in out.columns:
            g = pd.to_numeric(out[gu], errors='coerce')
            s = pd.to_numeric(out[sp], errors='coerce').clip(lower=0.5)
            out[f'{m}_gust_factor'] = (g / s).clip(0, 10)
            _gf_cols.append(f'{m}_gust_factor')
    if _gf_cols:
        out['gust_factor_ens_mean'] = out[_gf_cols].mean(axis=1)
        out['gust_factor_ens_std'] = out[_gf_cols].std(axis=1)
        if 'cape_ens_mean' in out.columns:
            out['gf_x_cape'] = out['gust_factor_ens_mean'] * (out['cape_ens_mean'] / 1000.0)

    # NWP-proxy circulation regimes (no ERA5 Lamb needed).
    # Used directly as features, by the regime x model meta-learner,
    # and by the regime-conditional rain-gate evaluation.
    out['is_summer'] = out['month'].isin([6, 7, 8, 9]).astype(float)
    if 'ne_sector_frac' in out.columns:
        out['regime_ne'] = (out['ne_sector_frac'] > 0.5).astype(float)
        out['regime_onshore'] = (out['onshore_frac'] > 0.5).astype(float)
    if 'rain_agreement' in out.columns:
        out['regime_wet'] = (pd.to_numeric(out['rain_agreement'], errors='coerce')
                             .fillna(0) >= 0.5).astype(float)

    # --- Lead-time interactions ---
    # 'lead_time' is set by the caller: 12 for day-0 archive rows, 36/60 for the
    # previous-runs stacked rows, true hours-ahead at inference.
    if 'lead_time' in out.columns:
        lt = pd.to_numeric(out['lead_time'], errors='coerce')
        if 'is_daytime' in out.columns:
            out['lead_x_daytime'] = lt * out['is_daytime']
        for p in ('temperature_2m', 'precipitation', 'wind_speed_10m'):
            sc = f'{p}_ens_std'
            if sc in out.columns:
                out[f'lead_x_{p}_std'] = lt * pd.to_numeric(out[sc], errors='coerce')

    return out


def get_feature_columns(df):
    exclude = set([
        'datetime', 'date',
        'temp_f', 'dewpoint_f', 'wind_mph', 'gust_mph', 'pressure_in',
        'precip_rate_in', 'precip_accum_in', 'precip_rate_mm', 'precip_accum_mm',
        'temp_obs', 'wind_ms', 'solar_wm2', 'uv',
        'temp_c', 'dewpoint_c', 'humidity_pct', 'wind_dir', 'gust_ms', 'pressure_hpa',
        # Re-scrape sub-hourly aggregates from the station: these are OBSERVATIONS
        # of the target itself (wind / precip / solar), so using them as features
        # for predicting the corresponding *_obs target is direct leakage.
        # Stations don't expose these at forecast time anyway, so the model
        # would train on something it can never see in production.
        'wind_ms_max', 'wind_ms_p95', 'gust_ms_p95',
        'precip_rate_max', 'solar_wm2_max', 'wind_dir_deg',
        'day_night', 'time_of_day', 'has_rain', 'light_rain', 'heavy_rain',
        'strong_wind', 'very_strong_wind', 'is_bura', 'winter_bura',
        'cloudy', 'extreme_cold', 'extreme_hot',
        '_derived_cloud_obs', '_derived_precip_obs',
        '_canonical_precip_rate_mm',
        # Legacy observation-derived columns: unavailable live and historically
        # biased because missing station observations were treated as dry.
        'dry_spell_length', 'monthly_clim_rain_freq',
        'date_str', 'time_str', '_h',
        # Derived display direction in degrees — has the 0/360 wrap; the model
        # should use the wind_u/wind_v components instead (report A1).
        'wind_dir_vec_ens',
    ])
    obs_suffix = '_obs'
    # Exclude features that require station observations — these are unavailable at
    # inference time because fetch_live_forecasts() only returns model forecasts.
    # Without this exclusion, the model trains on features that are always NaN in
    # production, causing systematic bias (train/serve skew).
    obs_dependent_suffixes = ('_error_lag1', '_error_lag3', '_error_lag6', '_error_lag24',
                              '_error_ma6', '_error_ma24', '_kalman_bias')

    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.endswith(obs_suffix):
            continue
        if any(col.endswith(s) for s in obs_dependent_suffixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def _make_val_split(X_tr, y_tr, val_frac=0.05):
    """Split last val_frac of training data as validation (respects time order)."""
    n = len(X_tr)
    split_idx = int(n * (1 - val_frac))
    return (X_tr.iloc[:split_idx], y_tr.iloc[:split_idx],
            X_tr.iloc[split_idx:], y_tr.iloc[split_idx:])


def _compute_sample_weights(y, datetime_index=None, decay_half_life_days=365):
    """Exponential temporal decay weights: recent samples get higher weight.
    Based on research showing NWP model updates make older biases less relevant."""
    n = len(y)
    if datetime_index is not None and len(datetime_index) == n:
        days_ago = (datetime_index.max() - datetime_index).dt.total_seconds() / 86400
    else:
        days_ago = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2) * days_ago / decay_half_life_days)
    weights = weights / weights.mean()  # normalize to mean=1
    return weights


def _timestamp_grouped_cv_splits(datetimes, n_splits=3, embargo_hours=72):
    """Expanding CV folds that keep duplicate valid-times in one partition."""
    dt = pd.Series(pd.to_datetime(np.asarray(datetimes))).reset_index(drop=True)
    unique_times = pd.Index(dt.dropna().sort_values().unique())
    if len(unique_times) <= n_splits:
        return []
    splits = []
    for train_time_idx, val_time_idx in TimeSeriesSplit(n_splits=n_splits).split(unique_times):
        val_times = unique_times[val_time_idx]
        cutoff = pd.Timestamp(val_times.min()) - pd.Timedelta(hours=embargo_hours)
        train_times = unique_times[train_time_idx]
        train_times = train_times[train_times <= cutoff]
        train_idx = np.flatnonzero(dt.isin(train_times).values)
        val_idx = np.flatnonzero(dt.isin(val_times).values)
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def _optuna_tune_hp(X_tr, y_tr, param_name, n_trials=15, base_objective='reg:quantileerror',
                    train_datetimes=None):
    """Bayesian hyperparameter optimization using Optuna with TimeSeriesSplit CV.
    Uses reg:quantileerror α=0.5 which directly minimizes MAE.
    3-fold TimeSeriesSplit with embargo gap.
    Wider search bounds + more trials.
    Optionally tunes decay_half_life_days."""
    # Build folds on unique valid timestamps, not rows. Lead-stacked training can
    # contain 12/36/60h copies of the same observation; row-based splitting can
    # otherwise place the same target timestamp on both sides of a fold.
    datetime_series = None
    cv_splits = []
    if train_datetimes is not None and len(train_datetimes) == len(X_tr):
        datetime_series = pd.Series(pd.to_datetime(np.asarray(train_datetimes))).reset_index(drop=True)
        cv_splits = _timestamp_grouped_cv_splits(datetime_series)
    if not cv_splits:
        cv_splits = list(TimeSeriesSplit(n_splits=3, gap=72).split(X_tr))

    # Variable-specific objective selection
    def get_objective_for_param(trial, param):
        if param in ('temperature_2m', 'dew_point_2m', 'relative_humidity_2m'):
            # Huber: robust to occasional extreme errors from bura/Saharan events
            obj = trial.suggest_categorical('obj_type', ['quantile', 'huber'])
            if obj == 'huber':
                hs = trial.suggest_float('huber_slope', 0.5, 5.0)
                return 'reg:pseudohubererror', {'huber_slope': hs}
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        elif param in ('wind_speed_10m', 'wind_gusts_10m'):
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        elif param == 'pressure_msl':
            # Pressure errors are near-Gaussian: MSE is appropriate
            obj = trial.suggest_categorical('obj_type', ['quantile', 'mse'])
            if obj == 'mse':
                return 'reg:squarederror', {}
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        elif param in ('cloud_cover', 'shortwave_radiation'):
            return 'reg:quantileerror', {'quantile_alpha': 0.5}
        else:
            return 'reg:quantileerror', {'quantile_alpha': 0.5}

    def objective(trial):
        obj_name, obj_params = get_objective_for_param(trial, param_name)
        # Tunable temporal decay half-life
        decay_hl = trial.suggest_categorical('decay_half_life', [90, 180, 365, 545, 730])
        hp = {
            'n_estimators': 1500,  # Use early stopping to find optimal count
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'objective': obj_name,
            'tree_method': 'hist',
            'max_bin': 512,  # Higher bins for constrained trees
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 30,
        }
        hp.update(obj_params)
        scores = []
        for train_idx, val_idx in cv_splits:
            X_t, X_v = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            # Compute sample weights with tuned half-life
            if datetime_series is not None:
                dt_t = datetime_series.iloc[train_idx]
                sw = _compute_sample_weights(y_t, dt_t, decay_half_life_days=decay_hl)
            elif train_datetimes is not None:
                dt_t = train_datetimes.iloc[train_idx] if hasattr(train_datetimes, 'iloc') else train_datetimes[train_idx]
                sw = _compute_sample_weights(y_t, dt_t, decay_half_life_days=decay_hl)
            else:
                sw = None
            model = _new_xgb_regressor(**hp)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False, sample_weight=sw)
            y_pred = model.predict(X_v)
            scores.append(mean_absolute_error(y_v, y_pred))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, warn_independent_sampling=False))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Extract tuned half-life before popping categorical params
    best_decay_hl = best.pop('decay_half_life', 365)
    # Reconstruct objective from best trial
    obj_type = best.pop('obj_type', 'quantile')
    if obj_type == 'huber':
        best['objective'] = 'reg:pseudohubererror'
    elif obj_type == 'mse':
        best['objective'] = 'reg:squarederror'
    else:
        best['objective'] = 'reg:quantileerror'
        best['quantile_alpha'] = 0.5
    best['tree_method'] = 'hist'
    best['max_bin'] = 512
    best['n_estimators'] = 1500
    best['random_state'] = 42
    best['n_jobs'] = -1
    best['early_stopping_rounds'] = 30
    print(f"    Optuna ({param_name}): best MAE={study.best_value:.4f} "
          f"(depth={best['max_depth']}, lr={best['learning_rate']:.4f}, "
          f"obj={best['objective']}, sub={best['subsample']:.2f}, "
          f"decay_hl={best_decay_hl}d)")
    best['_decay_half_life'] = best_decay_hl
    return best


def _select_features_by_importance(model, feature_cols, X_tr, y_tr, X_val, y_val,
                                   min_features=80, importance_type='gain'):
    """SHAP-based feature pruning: uses SHAP values for more accurate importance.
    Falls back to gain-based if SHAP fails. Removes bottom 5% of features."""
    try:
        import shap
        # Use TreeExplainer for efficient SHAP computation on tree models
        # Sample up to 500 rows for speed
        sample_size = min(300, len(X_tr))
        X_sample = X_tr.iloc[:sample_size]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        importances = np.abs(shap_values).mean(axis=0)
    except Exception:
        # Fallback to gain-based importance
        importances = model.feature_importances_

    nonzero_mask = importances > 0
    n_nonzero = nonzero_mask.sum()

    if n_nonzero <= min_features:
        return feature_cols

    # Remove bottom 5% of nonzero-importance features (conservative)
    nonzero_imps = importances[nonzero_mask]
    threshold = np.percentile(nonzero_imps, 5)
    selected = [f for f, imp in zip(feature_cols, importances)
                if imp >= threshold]

    if len(selected) < min_features or len(selected) >= len(feature_cols) * 0.92:
        return feature_cols

    return selected


def _train_xgb(X_tr, y_tr, X_val, y_val, hp, sample_weight=None):
    """Two-pass training: find best n_estimators on val, retrain on all data."""
    model_val = _new_xgb_regressor(**hp)
    model_val.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False,
                  sample_weight=sample_weight[:len(X_tr)] if sample_weight is not None else None)
    best_iteration = getattr(model_val, 'best_iteration', None)
    best_n = (int(best_iteration) + 1 if best_iteration is not None
              else int(hp.get('n_estimators', 500)))
    # An early optimum at (say) five trees is evidence for a small model, not
    # permission to jump back to the original 500-1500 tree ceiling. Keep a
    # modest numerical floor while respecting early stopping.
    best_n = max(best_n, 10)

    hp_final = {k: v for k, v in hp.items() if k != 'early_stopping_rounds'}
    hp_final['n_estimators'] = best_n
    X_full = pd.concat([X_tr, X_val], axis=0)
    y_full = pd.concat([y_tr, y_val], axis=0)
    w_full = sample_weight if sample_weight is not None else None
    model = _new_xgb_regressor(**hp_final)
    model.fit(X_full, y_full, verbose=False, sample_weight=w_full)
    return model, list(X_tr.columns)


def _find_optimal_blend(y_pred, y_te, ens_te):
    """Find optimal alpha: final = alpha*xgb + (1-alpha)*ensemble."""
    base_mae = mean_absolute_error(y_te, y_pred)
    best_alpha, best_mae = 1.0, base_mae
    best_pred = y_pred.copy()
    for alpha in np.arange(0.50, 1.01, 0.025):
        blend = alpha * y_pred + (1 - alpha) * ens_te
        bm = mean_absolute_error(y_te, blend)
        if bm < best_mae:
            best_mae, best_alpha = bm, alpha
            best_pred = blend.copy()
    return best_alpha, best_mae, best_pred


def _train_residual_blended(X_tr, y_tr, X_te, y_te, hp, param, ens_col, df_v_tr, df_v_te,
                            use_optuna=True, sample_weight=None, train_datetimes=None):
    """Train direct + residual (Huber) + multi-objective stacked models.
    Incorporates: Optuna HP tuning, feature selection, multi-loss stacking,
    temporal sample weighting."""
    ens_tr = pd.to_numeric(df_v_tr[ens_col], errors='coerce').fillna(0) if ens_col in df_v_tr.columns else pd.Series(0, index=y_tr.index)
    ens_te = pd.to_numeric(df_v_te[ens_col], errors='coerce').fillna(0) if ens_col in df_v_te.columns else pd.Series(0, index=y_te.index)

    # --- Optuna hyperparameter tuning ---
    tuned_hl = 365
    if use_optuna:
        hp = _optuna_tune_hp(X_tr, y_tr, param, n_trials=N_TRIALS, base_objective='reg:absoluteerror',
                             train_datetimes=train_datetimes)
        # Use the Optuna-tuned decay half-life for sample weights
        tuned_hl = hp.pop('_decay_half_life', 365)
        if train_datetimes is not None:
            sample_weight = _compute_sample_weights(y_tr, train_datetimes, decay_half_life_days=tuned_hl)
            # keep the half-weight of stacked long-lead rows after
            # the tuned-half-life recompute
            if 'lead_time' in X_tr.columns:
                _lt_rb = pd.to_numeric(X_tr['lead_time'], errors='coerce').fillna(12).values
                sample_weight = sample_weight * np.where(_lt_rb > 24, 0.5, 1.0)
            print(f"    Using tuned decay half-life: {tuned_hl} days")

    # Monotonic constraints will be set after feature selection (below)

    X_train_a, y_train_a, X_val_a, y_val_a = _make_val_split(X_tr, y_tr)

    # --- Feature selection: train initial model, prune low-importance features ---
    init_model, _ = _train_xgb(X_train_a, y_train_a, X_val_a, y_val_a, hp, sample_weight=sample_weight)
    selected_features = _select_features_by_importance(
        init_model, list(X_tr.columns), X_train_a, y_train_a, X_val_a, y_val_a,
        min_features=80
    )
    n_orig = len(X_tr.columns)
    n_sel = len(selected_features)
    if n_sel < n_orig:
        print(f"    Feature selection ({param}): {n_orig} → {n_sel} features")
        X_tr_sel = X_tr[selected_features]
        X_te_sel = X_te[selected_features]
    else:
        X_tr_sel = X_tr
        X_te_sel = X_te

    # --- Monotonic constraints ---
    # Enforce: a higher baseline in the model's target space implies a higher
    # direct prediction. For transformed targets (dew deficit / solar CSI),
    # ``ens_col`` is the transformed baseline; constraining the raw ensemble
    # would encode the wrong physical relationship.
    ens_mean_feature = ens_col
    sel_feature_list = list(X_tr_sel.columns) if hasattr(X_tr_sel, 'columns') else selected_features
    if ens_mean_feature in sel_feature_list:
        mono_idx = sel_feature_list.index(ens_mean_feature)
        constraints = [0] * len(sel_feature_list)
        constraints[mono_idx] = 1
        hp['monotone_constraints'] = tuple(constraints)

    X_train_a, y_train_a, X_val_a, y_val_a = _make_val_split(X_tr_sel, y_tr)

    # --- Direct model (MAE loss) ---
    direct_model, _ = _train_xgb(X_train_a, y_train_a, X_val_a, y_val_a, hp, sample_weight=sample_weight)
    direct_pred = direct_model.predict(X_te_sel)

    # --- Residual model (Huber loss) ---
    y_resid_tr = y_tr - ens_tr.values
    X_train_b, y_train_b, X_val_b, y_val_b = _make_val_split(X_tr_sel, y_resid_tr)
    hp_resid = hp.copy()
    hp_resid['objective'] = 'reg:pseudohubererror'
    # The residual target is y - baseline. It must be free to decrease as the
    # baseline rises (regression to the mean), so a direct-model monotonicity
    # constraint is invalid here.
    hp_resid.pop('monotone_constraints', None)
    resid_model, _ = _train_xgb(X_train_b, y_train_b, X_val_b, y_val_b, hp_resid, sample_weight=sample_weight)
    resid_correction = resid_model.predict(X_te_sel)
    resid_pred = ens_te.values + resid_correction

    # --- Multi-objective stacking (MSE model) ---
    # Based on Frontiers paper: training with different loss functions and blending
    hp_mse = hp.copy()
    hp_mse['objective'] = 'reg:squarederror'
    X_train_c, y_train_c, X_val_c, y_val_c = _make_val_split(X_tr_sel, y_tr)
    mse_model, _ = _train_xgb(X_train_c, y_train_c, X_val_c, y_val_c, hp_mse, sample_weight=sample_weight)
    mse_pred = mse_model.predict(X_te_sel)

    # --- CatBoost base learner ---
    cb_predict_task = 'GPU' if USING_GPU else 'CPU'
    try:
        if not RUN_AUX_DIAGNOSTICS:
            raise _AuxDiagnosticsDisabled
        cb_hp = {
            'iterations': 500,
            'depth': hp.get('max_depth', 6),
            'learning_rate': hp.get('learning_rate', 0.03),
            'l2_leaf_reg': hp.get('reg_lambda', 1.0),
            'subsample': hp.get('subsample', 0.8),
            # GPU CatBoost does not allow subsample with its default Bayesian
            # bootstrap. Bernoulli supports subsampling on both CPU and GPU.
            'bootstrap_type': 'Bernoulli',
            'loss_function': 'MAE',
            'random_seed': 42, 'verbose': 0,
            'early_stopping_rounds': 30,
        }
        cb_pool_tr = cb.Pool(X_train_a, y_train_a, weight=sample_weight[:len(X_train_a)] if sample_weight is not None else None)
        cb_pool_val = cb.Pool(X_val_a, y_val_a)
        cb_model = _new_catboost_regressor(**cb_hp)
        cb_model.fit(cb_pool_tr, eval_set=cb_pool_val)
        cb_pred = _catboost_predict(cb_model, X_te_sel)
        has_catboost = True
    except _AuxDiagnosticsDisabled:
        cb_pred = direct_pred.copy()
        cb_model = None
        has_catboost = False
    except Exception as e:
        if _DEVICE_REQUEST == 'cuda':
            raise RuntimeError(f'CatBoost GPU trening nije uspio: {e}') from e
        if _DEVICE_REQUEST == 'auto' and USING_GPU:
            try:
                print(f"    CatBoost GPU failed ({e}); retry na CPU")
                cb_model = _new_catboost_cpu_regressor(**cb_hp)
                cb_model.fit(cb_pool_tr, eval_set=cb_pool_val)
                cb_predict_task = 'CPU'
                cb_pred = cb_model.predict(X_te_sel, task_type='CPU')
                has_catboost = True
            except Exception as cpu_error:
                print(f"    CatBoost CPU retry failed ({cpu_error}), skipping")
                cb_pred = direct_pred.copy()
                cb_model = None
                has_catboost = False
        else:
            print(f"    CatBoost failed ({e}), skipping")
            cb_pred = direct_pred.copy()
            cb_model = None
            has_catboost = False

    # --- LightGBM base learner ---
    try:
        if not RUN_AUX_DIAGNOSTICS:
            raise _AuxDiagnosticsDisabled
        lgb_hp = {
            'n_estimators': 500,
            'max_depth': hp.get('max_depth', 6),
            'learning_rate': hp.get('learning_rate', 0.03),
            'subsample': hp.get('subsample', 0.8),
            'colsample_bytree': hp.get('colsample_bytree', 0.6),
            'reg_alpha': hp.get('reg_alpha', 0.05),
            'reg_lambda': hp.get('reg_lambda', 1.0),
            'min_child_weight': hp.get('min_child_weight', 5),
            # LightGBM recommends smaller histograms for its OpenCL GPU path.
            'max_bin': 63 if USING_GPU else 255,
            'objective': 'mae',
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        lgb_model = _new_lgbm_regressor(**lgb_hp)
        lgb_model.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)],
                       callbacks=[lgb.early_stopping(30, verbose=False)],
                       sample_weight=sample_weight[:len(X_train_a)] if sample_weight is not None else None)
        lgb_pred = lgb_model.predict(X_te_sel)
        has_lightgbm = True
    except _AuxDiagnosticsDisabled:
        lgb_pred = direct_pred.copy()
        lgb_model = None
        has_lightgbm = False
    except Exception as e:
        if _DEVICE_REQUEST == 'cuda':
            raise RuntimeError(f'LightGBM GPU trening nije uspio: {e}') from e
        if _DEVICE_REQUEST == 'auto' and USING_GPU:
            try:
                print(f"    LightGBM GPU failed ({e}); retry na CPU")
                lgb_model = _new_lgbm_cpu_regressor(**lgb_hp)
                lgb_model.fit(
                    X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                    sample_weight=(sample_weight[:len(X_train_a)]
                                   if sample_weight is not None else None),
                )
                lgb_pred = lgb_model.predict(X_te_sel)
                has_lightgbm = True
            except Exception as cpu_error:
                print(f"    LightGBM CPU retry failed ({cpu_error}), skipping")
                lgb_pred = direct_pred.copy()
                lgb_model = None
                has_lightgbm = False
        else:
            print(f"    LightGBM failed ({e}), skipping")
            lgb_pred = direct_pred.copy()
            lgb_model = None
            has_lightgbm = False

    # --- RidgeCV meta-learner ---
    # Stack predictions from all base learners using RidgeCV for optimal linear combination.
    # Use out-of-fold predictions on train set to avoid overfitting the meta-learner.
    base_preds_te = [direct_pred, resid_pred, mse_pred]
    base_names = ['xgb_direct', 'xgb_resid', 'xgb_mse']
    if has_catboost:
        base_preds_te.append(cb_pred)
        base_names.append('catboost')
    if has_lightgbm:
        base_preds_te.append(lgb_pred)
        base_names.append('lightgbm')

    base_mat_te = np.column_stack(base_preds_te)

    # Build meta-train features using train/val split predictions
    base_mat_train = np.column_stack([
        direct_model.predict(X_train_a),
        ens_tr.values[:len(X_train_a)] + resid_model.predict(X_train_a) if len(ens_tr) >= len(X_train_a) else direct_model.predict(X_train_a),
        mse_model.predict(X_train_a),
    ] + ([cb_model.predict(X_train_a, task_type=cb_predict_task)] if has_catboost else [])
      + ([lgb_model.predict(X_train_a)] if has_lightgbm else []))

    # --- regime x model interactions ("mixture-of-experts za
    # siromašne"). Each base prediction also enters multiplied by the NWP-proxy
    # regime flags, letting Ridge learn per-regime weights. Layout MUST be
    # mirrored exactly at inference (apply_correction) and reload.
    RIDGE_REGIME_FLAGS = ['is_summer', 'regime_ne', 'regime_wet']
    regime_cols = [c for c in RIDGE_REGIME_FLAGS
                   if c in df_v_tr.columns and c in df_v_te.columns]

    def _meta_with_regimes(base_mat, frame, n_rows=None):
        parts = [base_mat]
        for rc in regime_cols:
            flags = pd.to_numeric(frame[rc], errors='coerce').fillna(0).values
            if n_rows is not None:
                flags = flags[:n_rows]
            parts.append(base_mat * flags[:, None])
        return np.column_stack(parts)

    meta_X_train = _meta_with_regimes(base_mat_train, df_v_tr, n_rows=len(X_train_a))
    meta_X_te = _meta_with_regimes(base_mat_te, df_v_te)

    meta_y_train = y_train_a.values if hasattr(y_train_a, 'values') else y_train_a

    ridge_meta = None
    mae_ridge = float('nan')
    if RUN_AUX_DIAGNOSTICS:
        ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
        ridge_meta.fit(meta_X_train, meta_y_train)
        ridge_pred = ridge_meta.predict(meta_X_te)
        mae_ridge = mean_absolute_error(y_te, ridge_pred)
        print(f"    RidgeCV meta-learner: MAE={mae_ridge:.3f}, "
              f"alpha={ridge_meta.alpha_:.4f}, regimes={regime_cols}, "
              f"coefs=[{', '.join(f'{n}={c:.3f}' for n, c in zip(base_names, ridge_meta.coef_[:len(base_names)]))}]")

    # --- Stack predictions: find optimal mix of MAE, Huber-residual, MSE models ---
    best_stack_mae = float('inf')
    best_stack_weights = (1.0, 0.0, 0.0)
    best_stack_pred = direct_pred.copy()
    for w_direct in np.arange(0.3, 1.01, 0.1):
        for w_resid in np.arange(0.0, 1.01 - w_direct, 0.1):
            w_mse = 1.0 - w_direct - w_resid
            if w_mse < -0.01:
                continue
            stacked = w_direct * direct_pred + w_resid * resid_pred + w_mse * mse_pred
            sm = mean_absolute_error(y_te, stacked)
            if sm < best_stack_mae:
                best_stack_mae = sm
                best_stack_weights = (w_direct, w_resid, w_mse)
                best_stack_pred = stacked.copy()

    # --- Ensemble blend (stack + raw ensemble) ---
    best_alpha, best_blend_mae = 1.0, float('inf')
    for alpha in np.arange(0.5, 1.01, 0.025):
        blend = alpha * best_stack_pred + (1 - alpha) * ens_te.values
        bm = mean_absolute_error(y_te, blend)
        if bm < best_blend_mae:
            best_blend_mae, best_alpha = bm, alpha
    blend_pred = best_alpha * best_stack_pred + (1 - best_alpha) * ens_te.values

    mae_direct = mean_absolute_error(y_te, direct_pred)
    mae_resid = mean_absolute_error(y_te, resid_pred)
    mae_stack = best_stack_mae
    mae_blend = best_blend_mae

    # Ridge remains diagnostic-only until its base predictions are generated by
    # embargoed temporal OOF folds. Its current in-sample meta-fit is optimistic
    # and must not be eligible for production selection.
    methods = {'direct': (mae_direct, direct_pred, direct_model, False),
               'residual': (mae_resid, resid_pred, resid_model, True),
               'stacked': (mae_stack, best_stack_pred, direct_model, False),
               'blend': (mae_blend, blend_pred, direct_model, False)}

    best_name = min(methods, key=lambda k: methods[k][0])
    best_mae, best_pred, best_model, is_residual = methods[best_name]
    best_rmse = np.sqrt(mean_squared_error(y_te, best_pred))

    w_d, w_r, w_m = best_stack_weights
    ridge_info = (f"ridge_diagnostic={mae_ridge:.3f} (not eligible)"
                  if RUN_AUX_DIAGNOSTICS else "ridge_diagnostic=disabled")
    info_str = (f"direct={mae_direct:.3f}, residual={mae_resid:.3f}, "
                f"stacked({w_d:.1f}/{w_r:.1f}/{w_m:.1f})={mae_stack:.3f}, "
                f"blend({best_alpha:.2f})={mae_blend:.3f}, "
                f"{ridge_info} → {best_name}")

    return {
        'model': best_model, 'direct_model': direct_model, 'resid_model': resid_model,
        'mse_model': mse_model,
        'cb_model': cb_model, 'lgb_model': lgb_model, 'ridge_meta': ridge_meta,
        'has_catboost': has_catboost, 'has_lightgbm': has_lightgbm,
        'method': best_name, 'is_residual': is_residual,
        'ridge_meta_regime': regime_cols,
        'blend_alpha': best_alpha if best_name == 'blend' else None,
        # blend is built from the optimized three-model stack too, so its
        # weights must be persisted for report/reload/live parity.
        'stack_weights': best_stack_weights if best_name in ('stacked', 'blend') else None,
        'selected_features': selected_features if n_sel < n_orig else None,
        'tuned_hp': hp,  # Optuna-tuned hyperparameters for production retrain
        'decay_half_life': tuned_hl,
        'direct_n_estimators': direct_model.get_params()['n_estimators'],
        'resid_n_estimators': resid_model.get_params()['n_estimators'],
        'mse_n_estimators': mse_model.get_params()['n_estimators'],
        'mae': best_mae, 'rmse': best_rmse,
        'method_maes': {k: float(v[0]) for k, v in methods.items()},
        'info_str': info_str,
    }


def _predict_nonprecip_bundle(bundle, X, frame, ens_col):
    """Predict one non-precipitation model bundle in its model target space.

    This is the single dispatcher for untouched-report evaluation and live
    inference. Keeping the method reconstruction here prevents stored metrics
    from accidentally scoring the direct model when the shipped method is a
    residual, stack, blend, or Ridge meta-model.
    """
    method = bundle.get('method', 'direct')
    direct_model = bundle.get('direct_model') or bundle.get('model')
    direct = direct_model.predict(X)
    ens = (pd.to_numeric(frame[ens_col], errors='coerce').fillna(0).values
           if ens_col in frame.columns else np.zeros(len(X)))

    if method == 'direct':
        return direct

    resid_model = bundle.get('resid_model')
    mse_model = bundle.get('mse_model')
    needs_resid = method in ('residual', 'stacked', 'blend', 'ridge_meta') or bundle.get('is_residual')
    needs_mse = method in ('stacked', 'blend', 'ridge_meta')
    if needs_resid and resid_model is None:
        raise ValueError(f'{method} model nema ucitan resid_model artifact')
    if needs_mse and mse_model is None:
        raise ValueError(f'{method} model nema ucitan mse_model artifact')

    resid = ens + resid_model.predict(X) if resid_model is not None else direct
    mse = mse_model.predict(X) if mse_model is not None else direct

    if method == 'residual' or bundle.get('is_residual'):
        return resid

    if method in ('stacked', 'blend'):
        weights = bundle.get('stack_weights')
        if weights is None:
            raise ValueError(f'{method} model nema sacuvane stack_weights')
        w_direct, w_resid, w_mse = weights
        pred = w_direct * direct + w_resid * resid + w_mse * mse
        if method == 'blend':
            alpha = bundle.get('blend_alpha')
            if alpha is None:
                raise ValueError('blend model nema sacuvan blend_alpha')
            pred = alpha * pred + (1.0 - alpha) * ens
        return pred

    if method == 'ridge_meta':
        ridge = bundle.get('ridge_meta')
        if ridge is None:
            raise ValueError('ridge_meta model nije ucitan')
        base_preds = [direct, resid, mse]
        if bundle.get('has_catboost') and bundle.get('cb_model') is not None:
            base_preds.append(_catboost_predict(bundle['cb_model'], X))
        if bundle.get('has_lightgbm') and bundle.get('lgb_model') is not None:
            base_preds.append(bundle['lgb_model'].predict(X))
        base_mat = np.column_stack(base_preds)
        meta_parts = [base_mat]
        for regime_col in (bundle.get('ridge_meta_regime') or []):
            flags = pd.to_numeric(
                frame.get(regime_col, pd.Series(0, index=frame.index)),
                errors='coerce',
            ).fillna(0).values
            meta_parts.append(base_mat * flags[:, None])
        return ridge.predict(np.column_stack(meta_parts))

    # Forward compatibility for a simple estimator selected by a future method.
    return bundle['model'].predict(X)


def _postprocess_cloud_prediction(prediction, frame, station_target_version=2):
    """Cloud constraints shared by untouched-report and live inference."""
    pred = np.clip(np.asarray(prediction, dtype=float).copy(), 0, 100)
    ens_col = 'cloud_cover_ens_mean'
    if ens_col not in frame.columns:
        return pred
    ensemble = pd.to_numeric(frame[ens_col], errors='coerce').fillna(0).values
    low = ensemble < 10
    high = ensemble > 90
    pred[low] = np.minimum(pred[low], ensemble[low] + 30)
    pred[high] = np.maximum(pred[high], ensemble[high] - 30)

    # IBUDVA5's western terrain/sensor obstruction becomes too strong after
    # 16h in the warm season.  Those 17-18h cloud labels are intentionally not
    # learned; use the direct cloud ensemble there.  The calibrated station
    # proxy remains useful through 10-12h and the rest of the reliable window.
    dt = pd.to_datetime(frame['datetime'])
    hours = dt.dt.hour
    warm = dt.dt.month.isin([4, 5, 6, 7, 8, 9])
    solar_high_enough = compute_clear_sky(dt) >= 75
    station_unshaded = (
        (warm & hours.between(8, 16)) |
        (~warm & hours.between(9, 15))
    )
    in_window = (solar_high_enough & station_unshaded).values
    pred[~in_window] = ensemble[~in_window]

    # Existing artifacts were trained on the uncalibrated station-radiation
    # proxy.  Until a future scheduled retrain produces target version 2, do
    # not let that known bias drive the exact hours reported by the user.
    # This compatibility guard is deliberately absent from v2 evaluation and
    # future v2 artifacts, where the corrected labels are used directly.
    if int(station_target_version or 1) < 2:
        legacy_bad_hours = hours.isin([10, 11, 12, 17, 18]).values
        pred[legacy_bad_hours] = ensemble[legacy_bad_hours]
    return np.clip(pred, 0, 100)


def _pop_blend_inputs(frame, cls_proba, model_cols):
    """input matrix for the calibrated PoP blend — per-model wet
    flags + log1p amounts + agreement features + the XGB classifier proba.
    Must build identically at train and inference (column spec is persisted)."""
    feats = []
    idx = frame.index
    for c in model_cols:
        v = pd.to_numeric(frame.get(c, pd.Series(np.nan, index=idx)), errors='coerce')
        feats.append((v >= TRUSTED_RAIN_THRESHOLD).astype(float).fillna(0.0).values)
        feats.append(np.log1p(v.clip(lower=0)).fillna(0.0).values)
    for extra in ('rain_agreement', 'lam_frac_wet'):
        v = pd.to_numeric(frame.get(extra, pd.Series(0, index=idx)), errors='coerce')
        feats.append(v.fillna(0.0).values)
    feats.append(np.clip(np.asarray(cls_proba, dtype=float), 0.0, 1.0))
    return np.nan_to_num(np.column_stack(feats), nan=0.0)


def _clamp_precip_prediction(prediction, features):
    """Deterministic model-level precipitation clamp used in select/report."""
    pred = np.clip(np.asarray(prediction, dtype=float).copy(), 0, None)
    pred[pred < CORRECTED_RAIN_THRESHOLD_MM] = 0.0
    if 'ens_all_dry' in features.columns:
        pred[pd.to_numeric(features['ens_all_dry'], errors='coerce').values > 0.5] = 0.0
    if 'precip_ens_max_single' in features.columns:
        max_single = pd.to_numeric(
            features['precip_ens_max_single'], errors='coerce'
        ).values
        restore = (pred < CORRECTED_RAIN_THRESHOLD_MM) & (
            max_single >= CORRECTED_RAIN_THRESHOLD_MM
        )
        pred[restore] = max_single[restore]
    return pred


def _train_precipitation_twostage(X_tr, y_tr, X_te, y_te, X_val, y_val,
                                  X_cal, y_cal, X_gate, y_gate,
                                  feature_cols):
    """Enhanced two-stage precipitation: Optuna-tuned classifier + regressor.
    precision-first pipeline:
      * focal loss
      * asymmetric seasonal sample weights
      * hyperparam ranges: lower LR, more trees, higher min_child_weight, reg_alpha > 0
      * isotonic calibration
      * precision@recall threshold tuning
      * full meteorological scorecard incl. Brier + reliability
      * sanity-check baselines vs ICON-2I alone / ensemble / climatology / always-dry
    """
    RAIN_THRESH = CORRECTED_RAIN_THRESHOLD_MM

    y_cls_tr = (y_tr >= RAIN_THRESH).astype(int)
    y_cls_val = (y_val >= RAIN_THRESH).astype(int)
    y_cls_cal = (y_cal >= RAIN_THRESH).astype(int)
    y_cls_gate = (y_gate >= RAIN_THRESH).astype(int)
    rain_ratio = float(y_cls_tr.mean())
    # NB: we DO NOT use scale_pos_weight together with focal loss. Sample weights handle imbalance instead.

    # asymmetric seasonal sample weights conditioned on month.
    if 'month' in X_tr.columns:
        sw_tr = seasonal_sample_weights(X_tr['month'].values, y_cls_tr.values)
        sw_val = seasonal_sample_weights(X_val['month'].values, y_cls_val.values)
    else:
        sw_tr = np.ones(len(y_cls_tr))
        sw_val = np.ones(len(y_cls_val))

    # DMatrices once (xgb.train is the only way to use a custom objective reliably
    # across XGBoost versions).
    dtrain = xgb.DMatrix(X_tr, label=y_cls_tr, weight=sw_tr,
                         feature_names=list(X_tr.columns), missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_cls_val, weight=sw_val,
                       feature_names=list(X_val.columns), missing=np.nan)
    dcal = xgb.DMatrix(X_cal, label=y_cls_cal,
                       feature_names=list(X_cal.columns), missing=np.nan)
    dgate = xgb.DMatrix(X_gate, label=y_cls_gate,
                        feature_names=list(X_gate.columns), missing=np.nan)
    # --- Optuna joint tuning: focal-loss hyperparams + tree hyperparams ---
    def cls_objective(trial):
        # recommended ranges
        gamma_focal = trial.suggest_float('focal_gamma', 1.0, 3.0)
        alpha_focal = trial.suggest_float('focal_alpha', 0.20, 0.45)
        # tightened ranges
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 50, 200),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'tree_method': 'hist',
            'seed': 42,
        }
        booster = _train_xgb_booster(
            params, dtrain,
            num_boost_round=trial.suggest_int('n_estimators', 800, 1500, step=100),
            obj=focal_loss_xgb_objective(gamma_focal, alpha_focal),
            custom_metric=focal_loss_xgb_feval(gamma_focal, alpha_focal),
            evals=[(dval, 'val')],
            early_stopping_rounds=80,
            verbose_eval=False,
            maximize=False,
        )
        margins = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        proba = 1.0 / (1.0 + np.exp(-margins))
        try:
            prec, rec, _thr = precision_recall_curve(y_cls_val, proba)
            feasible = rec[:-1] >= 0.50
            best_prec_inner = float(prec[:-1][feasible].max()) if feasible.any() else 0.0
        except Exception:
            best_prec_inner = 0.0
        return -best_prec_inner

    study_cls = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_cls.optimize(cls_objective, n_trials=max(15, N_TRIALS // 2), show_progress_bar=False)
    best = study_cls.best_params
    focal_gamma = float(best.pop('focal_gamma'))
    focal_alpha = float(best.pop('focal_alpha'))
    n_estimators = int(best.pop('n_estimators'))
    cls_hp = {**best, 'tree_method': 'hist', 'seed': 42}
    print(f"    Optuna (precip_cls): best precision@recall>=0.5 = {-study_cls.best_value:.4f} "
          f"(focal_gamma={focal_gamma:.2f}, focal_alpha={focal_alpha:.2f}, "
          f"depth={cls_hp['max_depth']}, lr={cls_hp['learning_rate']:.4f}, "
          f"min_child_weight={cls_hp['min_child_weight']})")

    # --- Retrain on train+val to get final model, get best iteration via early stopping ---
    cls_val_booster = _train_xgb_booster(
        cls_hp, dtrain,
        num_boost_round=n_estimators,
        obj=focal_loss_xgb_objective(focal_gamma, focal_alpha),
        custom_metric=focal_loss_xgb_feval(focal_gamma, focal_alpha),
        evals=[(dval, 'val')],
        early_stopping_rounds=80,
        verbose_eval=False,
        maximize=False,
    )
    cls_best_n = max(int(cls_val_booster.best_iteration) + 1, 1)

    # Isotonic calibration gets its own chronological block. The later gate
    # block selects thresholds / trusted-vs-PoP mode and never fits the map.
    margins_cal = cls_val_booster.predict(dcal, iteration_range=(0, cls_best_n))
    proba_cal_raw = 1.0 / (1.0 + np.exp(-margins_cal))
    if len(proba_cal_raw) >= 500 and y_cls_cal.nunique() >= 2:
        iso_cal = IsotonicRegression(out_of_bounds='clip')
        iso_cal.fit(proba_cal_raw, y_cls_cal.astype(float))
        print(f"    Precip cls: isotonic calibration fit on "
              f"{len(proba_cal_raw)} calibration points")
    else:
        iso_cal = None

    margins_gate = cls_val_booster.predict(dgate, iteration_range=(0, cls_best_n))
    proba_gate_raw = 1.0 / (1.0 + np.exp(-margins_gate))
    proba_gate = (
        iso_cal.transform(proba_gate_raw) if iso_cal is not None
        else proba_gate_raw
    )

    # optimize the decision threshold directly on SEDI (robust to the
    # base-rate problem at 0.1-0.2mm), bounded so FAR doesn't blow past the old
    # precision@recall criterion by more than 5pp.
    thresh_pr = threshold_for_precision_at_recall(
        y_cls_gate.values, proba_gate, min_recall=0.50
    )
    far_at_pr = pf.far_score(
        y_cls_gate.values, (proba_gate >= thresh_pr).astype(int)
    )
    thresh_sedi, sedi_val = pf.threshold_for_max_sedi(
        y_cls_gate.values, proba_gate,
        far_cap=min(far_at_pr + 0.05, 0.95)
    )
    mets_pr = meteorological_metrics(
        y_cls_gate.values, (proba_gate >= thresh_pr).astype(int),
        p_proba=proba_gate,
    )
    mets_sedi = meteorological_metrics(
        y_cls_gate.values, (proba_gate >= thresh_sedi).astype(int),
        p_proba=proba_gate,
    )
    print(f"    Threshold P@R>=.5: t={thresh_pr:.3f} POD={mets_pr['pod']:.3f} "
          f"FAR={mets_pr['far']:.3f} CSI={mets_pr['csi']:.3f} SEDI={mets_pr['sedi']:.3f}")
    print(f"    Threshold maxSEDI:  t={thresh_sedi:.3f} POD={mets_sedi['pod']:.3f} "
          f"FAR={mets_sedi['far']:.3f} CSI={mets_sedi['csi']:.3f} SEDI={mets_sedi['sedi']:.3f}")
    best_thresh = thresh_sedi if mets_sedi['sedi'] >= mets_pr['sedi'] else thresh_pr
    pred_at_thresh = (proba_gate >= best_thresh).astype(int)
    # Full scorecard incl. Brier + reliability
    mets = meteorological_metrics(
        y_cls_gate.values, pred_at_thresh, p_proba=proba_gate
    )
    # CORP decomposition of the calibrated PoP
    _corp = pf.corp_reliability(y_cls_gate.values.astype(float), proba_gate)
    print(f"    CORP: Brier={_corp['brier']:.4f} MCB={_corp['mcb']:.4f} "
          f"DSC={_corp['dsc']:.4f} UNC={_corp['unc']:.4f}")
    print(f"    Precip cls @ thresh={best_thresh:.3f}: "
          f"POD={mets['pod']:.3f}, FAR={mets['far']:.3f}, "
          f"CSI={mets['csi']:.3f}, HSS={mets['hss']:.3f}, SEDI={mets['sedi']:.3f}, "
          f"Brier={mets['brier']:.4f}, BSS={mets['brier_skill_score']:.3f}, "
          f"RelRMSE={mets['reliability_rmse']:.4f}")

    # Sanity-check baselines on the independent gate-selection block.
    # (a) ICON-2I alone (>= RAIN_THRESH)
    # (b) Ensemble mean >= 0.1 mm
    # (c) Climatology (always predict base rate, threshold 0.5 -> always-dry)
    # (d) Always-dry
    baselines_info = {}
    try:
        icon_col = 'ITALIAMETEO_ICON2I_precipitation_model'
        if icon_col in X_gate.columns:
            pred_icon = (pd.to_numeric(X_gate[icon_col], errors='coerce').fillna(0) >= RAIN_THRESH).astype(int).values
            baselines_info['icon2i_alone'] = meteorological_metrics(y_cls_gate.values, pred_icon)
        ens_col = 'precipitation_ens_mean'
        if ens_col in X_gate.columns:
            pred_ens = (pd.to_numeric(X_gate[ens_col], errors='coerce').fillna(0) >= RAIN_THRESH).astype(int).values
            baselines_info['ensemble_mean'] = meteorological_metrics(y_cls_gate.values, pred_ens)
        baselines_info['always_dry'] = meteorological_metrics(
            y_cls_gate.values, np.zeros_like(y_cls_gate.values))
        baselines_info['climatology'] = meteorological_metrics(
            y_cls_gate.values,
            (np.full(len(y_cls_gate), rain_ratio) >= 0.5).astype(int),
            p_proba=np.full(len(y_cls_gate), rain_ratio))
        print(f"    Baselines on gate block (POD / FAR / CSI):")
        for name, b in baselines_info.items():
            print(f"      {name:18s} POD={b['pod']:.3f}  FAR={b['far']:.3f}  CSI={b['csi']:.3f}")
        # acceptance test: we must beat ICON-2I-alone + ensemble on FAR
        # while staying within ~10% POD.
        for ref in ('icon2i_alone', 'ensemble_mean'):
            if ref in baselines_info:
                b = baselines_info[ref]
                far_drop = (b['far'] - mets['far']) / max(b['far'], 1e-6) * 100
                pod_loss = (b['pod'] - mets['pod']) * 100
                print(f"      vs {ref}: FAR drop {far_drop:+.1f}%, POD loss {pod_loss:+.1f}pp")
    except Exception as _e:
        print(f"    Baseline computation skipped: {_e}")

    # --- calibrated PoP blend vs the single-LAM trusted gate ---
    # The hard ICON-2I veto throws away the other 9 models' information. Train
    # a logistic blend over per-model PoP inputs on the TRAIN fold, then play
    # both deciders on the independent gate fold, split by regime. The winner is persisted
    # as rain_gate_mode and applied at inference.
    pop_blend_info = None
    try:
        from sklearn.linear_model import LogisticRegression
        pop_model_cols = [f"{m}_precipitation_model" for m in MODELS
                          if f"{m}_precipitation_model" in X_gate.columns]
        margins_tr_pb = cls_val_booster.predict(dtrain, iteration_range=(0, cls_best_n))
        proba_tr_pb = 1.0 / (1.0 + np.exp(-margins_tr_pb))
        if iso_cal is not None:
            proba_tr_pb = iso_cal.transform(proba_tr_pb)
        M_tr = _pop_blend_inputs(X_tr, proba_tr_pb, pop_model_cols)
        lr_pop = LogisticRegression(max_iter=2000, C=1.0)
        lr_pop.fit(M_tr, y_cls_tr.values)
        M_gate = _pop_blend_inputs(X_gate, proba_gate, pop_model_cols)
        pop_val = lr_pop.predict_proba(M_gate)[:, 1]
        tau_pop, _ = pf.threshold_for_max_sedi(y_cls_gate.values, pop_val)
        pop_dec = (pop_val >= tau_pop).astype(int)

        icon_col = f"{TRUSTED_RAIN_MODEL}_precipitation_model"
        gate_dec = None
        if icon_col in X_gate.columns:
            gate_dec = (pd.to_numeric(X_gate[icon_col], errors='coerce').fillna(0)
                        >= TRUSTED_RAIN_THRESHOLD).astype(int).values

        mode = 'trusted'
        regime_table = {}
        if gate_dec is not None:
            yv = y_cls_gate.values
            regimes = {'overall': np.ones(len(yv), dtype=bool)}
            if 'month' in X_gate.columns:
                regimes['summer'] = X_gate['month'].isin([6, 7, 8, 9]).values
                regimes['non_summer'] = ~regimes['summer']
            if 'regime_ne' in X_gate.columns:
                regimes['ne'] = (pd.to_numeric(X_gate['regime_ne'], errors='coerce')
                                 .fillna(0) > 0.5).values
            for rname, rmask in regimes.items():
                if rmask.sum() < 100 or yv[rmask].sum() < 10:
                    continue
                regime_table[rname] = {
                    'gate': {'sedi': pf.sedi_score(yv[rmask], gate_dec[rmask]),
                             'csi': pf.csi_score(yv[rmask], gate_dec[rmask]),
                             'far': pf.far_score(yv[rmask], gate_dec[rmask])},
                    'pop': {'sedi': pf.sedi_score(yv[rmask], pop_dec[rmask]),
                            'csi': pf.csi_score(yv[rmask], pop_dec[rmask]),
                            'far': pf.far_score(yv[rmask], pop_dec[rmask])},
                    'n': int(rmask.sum()),
                }
            print("    PoP-blend vs trusted gate (gate block, SEDI/CSI/FAR):")
            for rname, t in regime_table.items():
                print(f"      {rname:11s} gate {t['gate']['sedi']:.3f}/{t['gate']['csi']:.3f}/"
                      f"{t['gate']['far']:.3f} | pop {t['pop']['sedi']:.3f}/"
                      f"{t['pop']['csi']:.3f}/{t['pop']['far']:.3f} (n={t['n']})")
            ov = regime_table.get('overall')
            if ov and (ov['pop']['sedi'] > ov['gate']['sedi']
                       and ov['pop']['far'] <= ov['gate']['far'] + 0.02):
                mode = 'pop_blend'
        print(f"    rain_gate_mode = {mode} (tau={tau_pop:.3f})")
        pop_blend_info = {'lr': lr_pop, 'cols': pop_model_cols,
                          'tau': float(tau_pop), 'mode': mode,
                          'regime_table': regime_table}
    except Exception as _e:
        print(f"    PoP blend preskočen ({_e}) — ostaje trusted gate")

    best_f1 = f1_score(y_cls_gate, pred_at_thresh, zero_division=0)

    # Keep the classifier that produced the held-out calibration probabilities.
    # Refitting it on the calibration fold would make the isotonic map, decision
    # threshold, and PoP blend refer to a different score distribution.
    cls_booster = cls_val_booster

    # Wrap booster in a thin adapter so the rest of the pipeline (which expects
    # an XGBClassifier-like object with predict_proba and save_model) works.
    class _BoosterProbaAdapter:
        """Wraps a booster trained with custom focal-loss objective so it
        looks like an XGBClassifier for the rest of the pipeline."""
        def __init__(self, booster, feature_names, focal_gamma, focal_alpha,
                     n_rounds):
            self._b = booster
            self._fn = list(feature_names)
            self._gamma = float(focal_gamma)
            self._alpha = float(focal_alpha)
            self._n_rounds = int(n_rounds)
        def _to_dmatrix(self, X):
            if isinstance(X, xgb.DMatrix):
                return X
            cols = list(X.columns) if hasattr(X, 'columns') else self._fn
            return xgb.DMatrix(X, feature_names=cols, missing=np.nan)
        def predict_proba(self, X):
            margins = self._b.predict(
                self._to_dmatrix(X), iteration_range=(0, self._n_rounds)
            )
            p = 1.0 / (1.0 + np.exp(-margins))
            return np.stack([1 - p, p], axis=1)
        def predict(self, X):
            p = self.predict_proba(X)[:, 1]
            return (p >= 0.5).astype(int)
        def save_model(self, path):
            self._b.save_model(path)
        # Persist hyperparams so production reload can rebuild predictions identically
        def get_params(self, deep=True):
            return {'focal_gamma': self._gamma, 'focal_alpha': self._alpha,
                    'best_iteration': self._n_rounds - 1,
                    'feature_names': self._fn}

    cls_model = _BoosterProbaAdapter(cls_booster, X_tr.columns,
                                      focal_gamma, focal_alpha, cls_best_n)

    # Test-set predictions for downstream blending logic
    cls_proba_te_raw = cls_model.predict_proba(X_te)[:, 1]
    cls_proba_te = iso_cal.transform(cls_proba_te_raw) if iso_cal is not None else cls_proba_te_raw

    # cls_hp_final is needed by downstream code that retrains XGBClassifier(**cls_hp_final);
    # provide a compatible dict (only used by old reload path which now branches to focal loader).
    cls_hp_final = {**cls_hp, 'n_estimators': cls_best_n,
                    'focal_gamma': focal_gamma, 'focal_alpha': focal_alpha}

    rain_mask_tr = y_tr >= RAIN_THRESH
    rain_mask_val = y_val >= RAIN_THRESH

    # --- Optuna tuning for precipitation regressor ---
    def reg_objective(trial):
        hp = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'objective': 'reg:absoluteerror',
            'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 40,
        }
        if rain_mask_tr.sum() >= 100 and rain_mask_val.sum() >= 20:
            model = _new_xgb_regressor(**hp)
            model.fit(X_tr[rain_mask_tr], np.sqrt(y_tr[rain_mask_tr]),
                      eval_set=[(X_val[rain_mask_val], np.sqrt(y_val[rain_mask_val]))],
                      verbose=False)
            pred_sqrt = model.predict(X_val)
            pred = np.square(np.clip(pred_sqrt, 0, None))
        else:
            model = _new_xgb_regressor(**hp)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = np.clip(model.predict(X_val), 0, None)
        return mean_absolute_error(y_val, pred)

    study_reg = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(reg_objective, n_trials=max(12, N_TRIALS // 3), show_progress_bar=False)
    reg_hp = study_reg.best_params
    reg_hp['objective'] = 'reg:absoluteerror'
    reg_hp['random_state'] = 42
    reg_hp['n_jobs'] = -1
    reg_hp['early_stopping_rounds'] = 30
    print(f"    Optuna (precip_reg): best MAE={study_reg.best_value:.4f} "
          f"(depth={reg_hp['max_depth']}, lr={reg_hp['learning_rate']:.4f})")

    if rain_mask_tr.sum() >= 100 and rain_mask_val.sum() >= 20:
        reg_val_model = _new_xgb_regressor(**reg_hp)
        y_rain_tr_sqrt = np.sqrt(y_tr[rain_mask_tr])
        y_rain_val_sqrt = np.sqrt(y_val[rain_mask_val])
        reg_val_model.fit(X_tr[rain_mask_tr], y_rain_tr_sqrt,
                          eval_set=[(X_val[rain_mask_val], y_rain_val_sqrt)], verbose=False)
        reg_best_n = max(int(reg_val_model.best_iteration) + 1, 1)

        reg_hp_final = {k: v for k, v in reg_hp.items() if k != 'early_stopping_rounds'}
        reg_hp_final['n_estimators'] = reg_best_n
        y_full = pd.concat([y_tr, y_val], axis=0)
        X_full = pd.concat([X_tr, X_val], axis=0)
        rain_mask_full = y_full >= RAIN_THRESH
        reg_model = _new_xgb_regressor(**reg_hp_final)
        reg_model.fit(X_full[rain_mask_full], np.sqrt(y_full[rain_mask_full]), verbose=False)
        reg_pred_te_sqrt = reg_model.predict(X_te)
        reg_pred_te = np.square(np.clip(reg_pred_te_sqrt, 0, None))
    else:
        reg_val_model = _new_xgb_regressor(**reg_hp)
        reg_val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        reg_best_n = max(int(reg_val_model.best_iteration) + 1, 1)
        reg_hp_final = {k: v for k, v in reg_hp.items() if k != 'early_stopping_rounds'}
        reg_hp_final['n_estimators'] = reg_best_n
        X_full = pd.concat([X_tr, X_val], axis=0)
        y_full = pd.concat([y_tr, y_val], axis=0)
        reg_model = _new_xgb_regressor(**reg_hp_final)
        reg_model.fit(X_full, y_full, verbose=False)
        reg_pred_te = np.clip(reg_model.predict(X_te), 0, None)

    single_hp = dict(
        n_estimators=1000, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=3.0,
        min_child_weight=15, gamma=0.2,
        objective='reg:absoluteerror', random_state=42, n_jobs=-1,
        early_stopping_rounds=30
    )
    single_val_model = _new_xgb_regressor(**single_hp)
    single_val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    single_best_n = max(int(single_val_model.best_iteration) + 1, 1)
    single_hp_final = {k: v for k, v in single_hp.items() if k != 'early_stopping_rounds'}
    single_hp_final['n_estimators'] = single_best_n
    X_full_s = pd.concat([X_tr, X_val], axis=0)
    y_full_s = pd.concat([y_tr, y_val], axis=0)
    single_model = _new_xgb_regressor(**single_hp_final)
    single_model.fit(X_full_s, y_full_s, verbose=False)
    single_pred = np.clip(single_model.predict(X_te), 0, None)
    single_pred[single_pred < RAIN_THRESH] = 0.0

    hard_pred = np.where(cls_proba_te >= best_thresh, reg_pred_te, 0.0)
    soft_pred = cls_proba_te * reg_pred_te
    sharp_pred = np.where(
        cls_proba_te >= best_thresh,
        0.7 * reg_pred_te + 0.3 * single_pred,
        single_pred * cls_proba_te
    )
    confidence = np.abs(cls_proba_te - 0.5) * 2
    adaptive_pred = np.where(
        cls_proba_te >= best_thresh,
        confidence * reg_pred_te + (1 - confidence) * single_pred,
        (1 - confidence) * single_pred * 0.5
    )

    # --- Tweedie model: unified zero-inflation + continuous positive density ---
    # Tweedie with p∈(1,2) handles point mass at zero naturally via log-link.
    # Replaces classifier+regressor with a single model, eliminating threshold sensitivity.
    # Tighter search space: shallower trees + stronger regularization to reduce false alarms.
    tscv_tw = TimeSeriesSplit(n_splits=3)
    def tweedie_objective(trial):
        tw_hp = {
            'n_estimators': 1000,
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.06, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.6),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 40),
            'gamma': trial.suggest_float('gamma', 0.1, 1.0),
            'objective': 'reg:tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.3, 1.8),
            'tree_method': 'hist',
            'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 30,
        }
        X_full_tw = pd.concat([X_tr, X_val], axis=0)
        y_full_tw = pd.concat([y_tr, y_val], axis=0).clip(lower=0)
        scores = []
        for ti, vi in tscv_tw.split(X_full_tw):
            X_t, X_v = X_full_tw.iloc[ti], X_full_tw.iloc[vi]
            y_t, y_v = y_full_tw.iloc[ti], y_full_tw.iloc[vi]
            m = _new_xgb_regressor(**tw_hp)
            m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            p = np.clip(m.predict(X_v), 0, None)
            p[p < RAIN_THRESH] = 0.0  # evaluate with production amount threshold
            scores.append(mean_absolute_error(y_v, p))
        return np.mean(scores)

    study_tw = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, warn_independent_sampling=False))
    study_tw.optimize(tweedie_objective, n_trials=max(20, N_TRIALS // 2), show_progress_bar=False)
    tw_hp = study_tw.best_params
    tw_hp['objective'] = 'reg:tweedie'
    tw_hp['tree_method'] = 'hist'
    tw_hp['n_estimators'] = 1000
    tw_hp['random_state'] = 42
    tw_hp['n_jobs'] = -1
    tw_hp['early_stopping_rounds'] = 30

    # Train Tweedie model on train, validate on val
    tw_val_model = _new_xgb_regressor(**tw_hp)
    X_full_tw = pd.concat([X_tr, X_val], axis=0)
    y_full_tw = pd.concat([y_tr, y_val], axis=0).clip(lower=0)
    tw_val_model.fit(X_tr, y_tr.clip(lower=0), eval_set=[(X_val, y_val.clip(lower=0))], verbose=False)
    tw_best_n = max(int(tw_val_model.best_iteration) + 1, 1)

    tw_hp_final = {k: v for k, v in tw_hp.items() if k != 'early_stopping_rounds'}
    tw_hp_final['n_estimators'] = tw_best_n
    tweedie_model = _new_xgb_regressor(**tw_hp_final)
    tweedie_model.fit(X_full_tw, y_full_tw, verbose=False)
    tweedie_pred = np.clip(tweedie_model.predict(X_te), 0, None)
    print(f"    Tweedie: p={tw_hp.get('tweedie_variance_power', 1.5):.2f}, "
          f"MAE={mean_absolute_error(y_te, tweedie_pred):.4f}")

    methods = {
        'single': (np.clip(single_pred, 0, None), single_model),
        'hard': (np.clip(hard_pred, 0, None), None),
        'soft': (np.clip(soft_pred, 0, None), None),
        'sharp': (np.clip(sharp_pred, 0, None), None),
        'adaptive': (np.clip(adaptive_pred, 0, None), None),
        'tweedie': (np.clip(tweedie_pred, 0, None), tweedie_model),
    }

    # Select the algorithm using the deterministic clamp that will also be
    # applied on the untouched report split and in live output.
    method_maes = {}
    for name, (pred, _) in methods.items():
        clamped = _clamp_precip_prediction(pred, X_te)
        methods[name] = (clamped, methods[name][1])
        method_maes[name] = mean_absolute_error(y_te, clamped)

    best_method = min(method_maes, key=method_maes.get)
    best_pred = methods[best_method][0]
    mae = method_maes[best_method]
    rmse = np.sqrt(mean_squared_error(y_te, best_pred))

    print(f"\n  >> PADAVINE: Two-stage model (klasifikacija + regresija)")
    print(f"    Stage 1 (cls): rain_ratio={rain_ratio:.3f}, thresh={best_thresh:.2f}, F1={best_f1:.3f}")
    print(f"    Stage 2 (reg): train_rain={rain_mask_tr.sum()}, test_rain={(y_te >= RAIN_THRESH).sum()}")
    print(f"    Methods: " + ", ".join(f"{k}={v:.3f}" for k, v in method_maes.items()) + f" → BEST={best_method}")

    return {
        'cls_model': cls_model, 'reg_model': reg_model, 'single_model': single_model,
        'tweedie_model': tweedie_model,
        'best_method': best_method, 'threshold': best_thresh,
        'mae': mae, 'rmse': rmse, 'features': feature_cols,
        'use_sqrt': bool(rain_mask_tr.sum() >= 100 and rain_mask_val.sum() >= 20),
        'iso_calibrator': iso_cal,  # isotonic calibrator for cls proba
        # HP info for production retrain on all data
        'cls_hp_final': cls_hp_final,
        'reg_hp_final': reg_hp_final,
        'single_hp_final': single_hp_final,
        'tweedie_hp_final': tw_hp_final,
        'pop_blend': pop_blend_info,  # (None -> trusted gate)
        'gate_metrics_full': mets,
        'gate_corp': _corp,
        'baselines': baselines_info,
    }


# Rain-onset timing: discrete-time conditional-hazard model. The historical
# data is a continuous valid-time series (no archived per-run lead tables),
# so the discrete-time hazard is adapted to "dry spells":
#   onset = first hour with obs precip >= ONSET_THRESHOLD_MM after >=
#           ONSET_DRY_GAP_HOURS dry hours. We model h(t) = P(onset this hour |
#           dry through t-1) as a function of atmospheric state + dry_age (t).
# At inference we walk the live 48h forecast hour-by-hour, accumulate
#   S(t)=Π(1-h), F(t)=1-S(t), and extract earliest/likely/latest onset.
ONSET_THRESHOLD_MM = 0.2
ONSET_DRY_GAP_HOURS = 3
ONSET_MAX_DRY_AGE = 48          # cap the elapsed-dry feature
ONSET_P_LIKELY = 0.50           # F(t) crossing (kept for context/CDF reporting)
ONSET_P_EARLY = 0.25
ONSET_P_LATE = 0.75
# Precision-first declaration uses the INSTANTANEOUS per-hour hazard, not the
# accumulated CDF: over a long dry spell F(t) creeps to 0.5 from base rate alone
# (~1.8%/h) and would declare rain almost every day. We declare only when a
# specific hour's hazard is genuinely elevated. Gate-tune this on held-out
# ±Nh hit-rate vs false-early-alarm rate (report B5).
ONSET_HAZARD_DECLARE = 0.10     # per-hour hazard for a confident onset hour
ONSET_HAZARD_BAND = 0.05        # lower hazard bounding the earliest/latest window


def _score_onset_threshold(hazard, observed_onset, dry_age, datetimes,
                           threshold):
    """Score one onset threshold at event level, including false alarms."""
    pred, obs = _onset_event_hours(
        hazard, observed_onset, dry_age, datetimes, threshold=threshold,
    )
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    has_pred = np.isfinite(pred)
    has_obs = np.isfinite(obs)
    tp = int((has_pred & has_obs).sum())
    fp = int((has_pred & ~has_obs).sum())
    fn = int((~has_pred & has_obs).sum())
    metrics = onset_timing_metrics(pred, obs)
    metrics.update({
        'threshold': float(threshold),
        'csi': float(tp / max(tp + fp + fn, 1)),
        'tp': tp, 'fp': fp, 'fn': fn,
    })
    return metrics


def _select_onset_hazard_threshold(hazard, observed_onset, dry_age, datetimes,
                                   min_pod=0.50):
    """Choose the precision-first onset threshold on validation spells.

    The threshold is selected by event CSI, subject to retaining at least 50%
    onset POD. This makes the production decision an explicit result of onset
    scores (including false alarms), rather than a hand-written probability.
    The untouched test period is used only after this selection.
    """
    candidates = np.round(np.arange(0.03, 0.305, 0.005), 3)
    scored = [
        _score_onset_threshold(
            hazard, observed_onset, dry_age, datetimes, float(threshold),
        )
        for threshold in candidates
    ]
    feasible = [m for m in scored if m['onset_pod'] >= min_pod]
    pool = feasible or scored
    best = max(
        pool,
        key=lambda m: (
            m['csi'], -m['onset_far'], m.get('hit_within_3h', float('-inf')),
            m['threshold'],
        ),
    )
    return float(best['threshold']), best


def build_onset_person_period(df, feature_cols):
    """Build the dry-spell person-period table (report B1).
    Returns (X with dry_age column, y onset 0/1, datetimes) over period rows
    (every dry hour = target 0; the hour a >=3h-dry spell ends in rain = 1).
    Wet-but-continuation hours are excluded (we're inside an event, not a spell).
    """
    obs = pd.to_numeric(df.get('_derived_precip_obs', pd.Series(np.nan, index=df.index)),
                        errors='coerce').values
    n = len(df)
    valid = np.isfinite(obs)
    wet = valid & (obs >= ONSET_THRESHOLD_MM)
    dry_age = np.zeros(n)
    onset = np.zeros(n, dtype=bool)
    period = np.zeros(n, dtype=bool)
    prev_dry = 0
    for i in range(n):
        if not valid[i]:
            prev_dry = 0          # unknown obs breaks the streak; no period row
            continue
        if wet[i]:
            if prev_dry >= ONSET_DRY_GAP_HOURS:
                onset[i] = True
                dry_age[i] = min(prev_dry, ONSET_MAX_DRY_AGE)
                period[i] = True   # onset row (target 1)
            prev_dry = 0           # short-gap/continuation wet: not a period row
        else:
            prev_dry += 1
            dry_age[i] = min(prev_dry, ONSET_MAX_DRY_AGE)
            period[i] = True       # dry hazard row (target 0)
    pos = np.where(period)[0]
    feats = [c for c in feature_cols if c in df.columns and c != 'dry_age']
    X = df.iloc[pos][feats].copy()
    X['dry_age'] = dry_age[pos]
    y = pd.Series(onset[pos].astype(int), index=X.index)
    dt = df.iloc[pos]['datetime'].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y, dt


def _onset_cdf_from_hazard(hazard, dry_age):
    """Given per-row hazard within a chronological dry-spell sequence, reset the
    survival product whenever a new spell starts (dry_age drops) and return the
    onset CDF F(t)=1-S(t) per row."""
    h = np.clip(np.asarray(hazard, dtype=float), 0.0, 0.999)
    da = np.asarray(dry_age, dtype=float)
    F = np.zeros(len(h))
    S = 1.0
    prev_da = np.nan
    for i in range(len(h)):
        if not np.isfinite(da[i]) or not np.isfinite(h[i]):
            F[i] = np.nan
            S = 1.0
            prev_da = np.nan
            continue
        # Equal values are normal after dry_age reaches its cap and on the
        # positive onset row. Only a decrease marks a new spell.
        if np.isfinite(prev_da) and da[i] < prev_da:
            S = 1.0
        S *= (1.0 - h[i])
        F[i] = 1.0 - S
        prev_da = da[i]
    return F


def _onset_event_hours(hazard, observed_onset, dry_age, datetimes,
                       threshold=ONSET_HAZARD_DECLARE):
    """Return per-spell predicted/observed onset elapsed hours.

    Timing is derived from timestamps, not capped ``dry_age``. This prevents a
    long spell whose age is fixed at 48 from receiving a mechanically zero
    timing error. Gaps in the person-period table (unknown observations or
    excluded wet-continuation rows) start a new spell.
    """
    hz = np.asarray(hazard, dtype=float)
    onset = np.asarray(observed_onset, dtype=int)
    ages = np.asarray(dry_age, dtype=float)
    times = pd.to_datetime(pd.Series(datetimes)).reset_index(drop=True)
    if not (len(hz) == len(onset) == len(ages) == len(times)):
        raise ValueError('onset event arrays moraju imati istu duzinu')

    predicted, observed = [], []
    spell = None
    previous_age = np.nan
    previous_time = None
    for i in range(len(hz)):
        timestamp = pd.Timestamp(times.iloc[i])
        time_gap = (
            previous_time is not None and
            timestamp - previous_time > pd.Timedelta(hours=1)
        )
        new_spell = (
            spell is None or time_gap or
            (np.isfinite(previous_age) and ages[i] < previous_age)
        )
        if new_spell:
            if spell is not None:
                predicted.append(spell['pred'])
                observed.append(spell['obs'])
            spell = {'start': timestamp, 'pred': np.nan, 'obs': np.nan}

        elapsed = (timestamp - spell['start']).total_seconds() / 3600.0
        if np.isnan(spell['pred']) and np.isfinite(hz[i]) and hz[i] >= threshold:
            spell['pred'] = elapsed
        if onset[i] == 1:
            spell['obs'] = elapsed
        previous_age = ages[i]
        previous_time = timestamp

    if spell is not None:
        predicted.append(spell['pred'])
        observed.append(spell['obs'])
    return predicted, observed


def train_onset_model(df):
    """Train the discrete-time onset hazard (XGBoost) on the dry-spell table.
    Trains on < SPLIT_DATE, isotonic-calibrates on a tail val slice, verifies on
    >= SPLIT_DATE with hazard Brier/reliability + event-based onset metrics
    (report B4) against the first-wet-hour baseline. Saves model+calibrator."""
    print("\n  [Onset] Treniranje discrete-time hazard modela...")
    feature_cols = get_feature_columns(df)
    try:
        X, y, dt = build_onset_person_period(df, feature_cols)
    except Exception as e:
        print(f"  [Onset] build person-period failed ({e}); preskačem onset model.")
        _invalidate_onset_artifacts()
        return None
    if len(X) < 1000 or int(y.sum()) < 30:
        print(f"  [Onset] nedovoljno: rows={len(X)}, onsets={int(y.sum())}; preskačem.")
        _invalidate_onset_artifacts()
        return None

    tr = (dt < SPLIT_DATE).values
    te = (dt >= SPLIT_DATE).values
    if tr.sum() < 500 or int(y[tr].sum()) < 20 or te.sum() < 100:
        print(f"  [Onset] split premali (train onsets={int(y[tr].sum())}); preskačem.")
        _invalidate_onset_artifacts()
        return None

    feats = list(X.columns)
    # Monotone constraints where physics dictates (report A2/B): onset hazard
    # increases with model agreement-wet and humidity.
    mono_up = {'lam_frac_wet', 'rain_agreement', 'frac_high_res_wet',
               'relative_humidity_2m_ens_mean', 'humidity_above_90',
               'precip_ens_mean_rainy', 'lam_all_wet'}
    constraints = tuple(1 if f in mono_up else 0 for f in feats)

    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    dt_te = dt[te].reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)

    # tail val slice of train for calibration + early stopping
    vcut = int(len(X_tr) * 0.9)
    X_fit, y_fit = X_tr.iloc[:vcut], y_tr.iloc[:vcut]
    X_val, y_val = X_tr.iloc[vcut:], y_tr.iloc[vcut:]

    if 'month' in X_fit.columns:
        sw_fit = seasonal_sample_weights(X_fit['month'].values, y_fit.values)
    else:
        sw_fit = np.ones(len(y_fit))

    hp = dict(n_estimators=800, max_depth=5, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.6, reg_alpha=0.2, reg_lambda=2.0,
              min_child_weight=20, gamma=0.1, objective='binary:logistic',
              eval_metric='logloss', tree_method='hist', max_bin=512,
              monotone_constraints=constraints, random_state=42, n_jobs=-1,
              early_stopping_rounds=40)
    clf = _new_xgb_classifier(**hp)
    clf.fit(X_fit, y_fit, sample_weight=sw_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n = max(int(getattr(clf, 'best_iteration', 0)) + 1, 1)

    # isotonic calibration of the hazard on val
    proba_val = clf.predict_proba(X_val)[:, 1]
    iso = None
    if len(proba_val) >= 300 and y_val.sum() >= 5:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(proba_val, y_val.astype(float))

    calibrated_val = iso.transform(proba_val) if iso is not None else proba_val
    dt_val = dt[tr].iloc[vcut:].reset_index(drop=True)
    declare_threshold, threshold_selection = _select_onset_hazard_threshold(
        calibrated_val, y_val.values, X_val['dry_age'].values, dt_val,
    )

    # Verify on the test period
    proba_te = clf.predict_proba(X_te)[:, 1]
    if iso is not None:
        proba_te = iso.transform(proba_te)
    brier = float(np.mean((proba_te - y_te.values) ** 2))

    # Event-based onset metrics: reconstruct per-spell predicted vs observed
    # onset. Predicted onset = first hour the INSTANTANEOUS hazard is elevated
    # (>= ONSET_HAZARD_DECLARE), mirroring the precision-first inference rule.
    da = X_te['dry_age'].values
    pred_onset, obs_onset = _onset_event_hours(
        proba_te, y_te.values, da, dt_te, declare_threshold
    )
    ometrics = onset_timing_metrics(pred_onset, obs_onset)
    test_threshold_metrics = _score_onset_threshold(
        proba_te, y_te.values, da, dt_te, declare_threshold,
    )

    # Baseline: fraction-of-models-wet crossing 0.5 as the onset anchor
    print(f"  [Onset] test Brier={brier:.4f} | onsets(test)={int(y_te.sum())} | "
          f"MAE={ometrics['mae_hours']:.2f}h | hit±1h={ometrics['hit_within_1h']:.2f} "
          f"hit±2h={ometrics['hit_within_2h']:.2f} hit±3h={ometrics['hit_within_3h']:.2f} | "
          f"onset POD={ometrics['onset_pod']:.2f} FAR={ometrics['onset_far']:.2f} | "
          f"threshold={declare_threshold:.3f} CSI={test_threshold_metrics['csi']:.2f}")

    # Keep the exact classifier whose validation scores were isotonic-calibrated.
    # Refitting across that calibration tail would invalidate the saved map.
    clf_prod = clf

    clf_prod.save_model(os.path.join(MODEL_DIR, 'onset_hazard.json'))
    _write_json_atomic(
        os.path.join(MODEL_DIR, 'onset_meta.json'),
        {'features': feats, 'best_iteration': best_n,
         'threshold_mm': ONSET_THRESHOLD_MM, 'dry_gap': ONSET_DRY_GAP_HOURS,
         'declare_threshold': declare_threshold,
         'band_threshold': min(ONSET_HAZARD_BAND, declare_threshold / 2.0),
         'threshold_selection': threshold_selection,
         'metrics': test_threshold_metrics, 'brier': brier},
        ensure_ascii=False, indent=2,
    )
    if iso is not None:
        import joblib
        joblib.dump(iso, os.path.join(MODEL_DIR, 'onset_iso.joblib'))
    else:
        _remove_if_exists(os.path.join(MODEL_DIR, 'onset_iso.joblib'))
    print(f"  [Onset] sačuvan kalibrisani bundle (features={len(feats)}, best_n={best_n}).")
    return {'model': clf_prod, 'calibrator': iso, 'features': feats,
            'metrics': test_threshold_metrics,
            'declare_threshold': declare_threshold,
            'band_threshold': min(ONSET_HAZARD_BAND, declare_threshold / 2.0),
            'threshold_source': 'validation_event_scores'}


def load_onset_model():
    """Reload the onset hazard bundle for --skip-training; None if absent."""
    mpath = os.path.join(MODEL_DIR, 'onset_hazard.json')
    metapath = os.path.join(MODEL_DIR, 'onset_meta.json')
    if not (os.path.exists(mpath) and os.path.exists(metapath)):
        return None
    try:
        clf = _new_xgb_classifier()
        clf.load_model(mpath)
        _restore_xgb_device(clf)
        with open(metapath, encoding='utf-8') as f:
            meta = json.load(f)
        iso = None
        ipath = os.path.join(MODEL_DIR, 'onset_iso.joblib')
        if os.path.exists(ipath):
            import joblib
            iso = joblib.load(ipath)
        return {'model': clf, 'calibrator': iso, 'features': meta.get('features', []),
                'metrics': meta.get('metrics', {}),
                'declare_threshold': float(meta.get(
                    'declare_threshold', ONSET_HAZARD_DECLARE)),
                'band_threshold': float(meta.get(
                    'band_threshold', ONSET_HAZARD_BAND)),
                'threshold_source': ('validation_event_scores'
                                     if 'declare_threshold' in meta
                                     else 'heldout_report_legacy')}
    except Exception as e:
        if _DEVICE_REQUEST == 'cuda':
            raise RuntimeError(f'Onset GPU reload failed: {e}') from e
        print(f"  [Onset] reload failed ({e}); bez onset bloka.")
        return None


def _forecast_onset_dry_age(precip):
    """Dry-age feature for live onset rows, matching training onset semantics."""
    values = np.asarray(precip, dtype=float)
    dry_age = np.zeros(len(values), dtype=float)
    previous_dry = 0
    for i, amount in enumerate(values):
        if not np.isfinite(amount):
            dry_age[i] = np.nan
            previous_dry = 0
        elif amount >= ONSET_THRESHOLD_MM:
            dry_age[i] = min(previous_dry, ONSET_MAX_DRY_AGE)
            previous_dry = 0
        else:
            previous_dry += 1
            dry_age[i] = min(previous_dry, ONSET_MAX_DRY_AGE)
    return dry_age


def _align_onset_features_to_display(fc_features):
    """Align the complete onset feature row to displayed rain intervals.

    Open-Meteo precipitation at raw timestamp T describes [T-1h, T], while
    the public output moves it to timestamp T-1h. The hazard model was trained
    on the complete raw row, so live inference must move every model feature
    with that precipitation row. Moving only corrected rain made dry age come
    from T+1 while humidity/agreement still came from T.
    """
    aligned = fc_features.copy()
    feature_cols = [c for c in aligned.columns if c != 'datetime']
    # Assign each shifted Series through ``frame[col]`` so pandas is allowed to
    # widen integer/bool columns when the final shifted row becomes NaN.
    # ``.loc[:, cols] = ...`` attempts an in-place dtype-preserving update in
    # pandas 3 and raises LossySetitemError for live integer-valued features.
    shifted = aligned[feature_cols].shift(-1)
    for col in feature_cols:
        aligned[col] = shifted[col]
    return aligned


def predict_onset_timing(fc, bundle):
    """Return a scored onset decision for every hour of the +48h horizon.

    The saved validation-selected hazard threshold drives all 48 hours. SKALA
    is intentionally not baked into that long-range threshold: its support is
    added later only to the short hours where radar nowcasting exists.
    Defensive: never raises into the main pipeline.
    """
    if not bundle:
        return None
    try:
        feats = bundle['features']
        model = bundle['model']
        iso = bundle.get('calibrator')
        declare_threshold = float(bundle.get(
            'declare_threshold', ONSET_HAZARD_DECLARE))
        band_threshold = float(bundle.get(
            'band_threshold', ONSET_HAZARD_BAND))
        threshold_source = bundle.get(
            'threshold_source', 'heldout_report_legacy')

        # Prefer the same gated/corrected precipitation that is displayed to
        # users. Fall back to the raw ensemble only for older callers.
        state_precip = fc.get(
            '_onset_state_precip',
            fc.get('precipitation_ens_mean', pd.Series(np.nan, index=fc.index)),
        )
        ens_precip = pd.to_numeric(state_precip, errors='coerce').values
        dry_age = _forecast_onset_dry_age(ens_precip)

        X = pd.DataFrame(index=fc.index)
        for c in feats:
            if c == 'dry_age':
                continue
            X[c] = pd.to_numeric(fc[c], errors='coerce') if c in fc.columns else np.nan
        X['dry_age'] = dry_age
        X = X[feats]
        hazard = model.predict_proba(X)[:, 1]
        if iso is not None:
            hazard = iso.transform(hazard)

        now = local_now().floor('h')
        dts = pd.to_datetime(fc['datetime'])
        fut = (dts >= now).values
        idx = np.where(fut)[0]
        if len(idx) == 0:
            return None
        idx = idx[:48]  # 48h horizon

        already_raining = bool(
            np.isfinite(ens_precip[idx[0]])
            and ens_precip[idx[0]] >= ONSET_THRESHOLD_MM
        )
        S = 1.0
        prob_by_hour = []
        scored_hazard = np.full(len(idx), np.nan)
        cumulative = np.zeros(len(idx), dtype=float)
        for j, i in enumerate(idx):
            amount = ens_precip[i]
            eligible = bool(
                np.isfinite(amount) and np.isfinite(dry_age[i])
                and dry_age[i] >= ONSET_DRY_GAP_HOURS
            )
            if not eligible:
                # A wet/unknown hour or the start of a new dry spell cannot be
                # an onset-after-3h-dry decision. Reset the spell CDF.
                S = 1.0
                h = None
                p_by_then = 0.0
            else:
                h = float(np.clip(hazard[i], 0, 0.999))
                scored_hazard[j] = h
                S *= (1.0 - h)
                p_by_then = float(1.0 - S)
            cumulative[j] = p_by_then
            ts = dts.iloc[i]
            signal = bool(h is not None and h >= declare_threshold)
            prob_by_hour.append({
                'datetime': ts.isoformat(), 'hour': int(ts.hour),
                'lead_h': j + 1, 'eligible': eligible,
                'hazard': round(h, 3) if h is not None else None,
                'p_by_then': round(p_by_then, 3),
                'threshold': round(declare_threshold, 3),
                'signal': signal,
            })
        finite_hazard = np.where(np.isfinite(scored_hazard))[0]
        peak = (float(np.nanmax(scored_hazard)) if finite_hazard.size else 0.0)
        max_prob = round(float(cumulative.max()), 3)
        signal_hours = np.where(
            np.isfinite(scored_hazard) & (scored_hazard >= declare_threshold)
        )[0]

        # Precision-first: declare only when a specific hour's hazard is elevated
        # (avoids declaring from base-rate CDF accumulation over long dry spells).
        if signal_hours.size == 0:
            return {'already_raining': already_raining,
                    'declared': already_raining,
                    'likely_datetime': (dts.iloc[idx[0]].isoformat()
                                        if already_raining else None),
                    'max_prob': max_prob, 'peak_hazard': round(peak, 3),
                    'declare_threshold': round(declare_threshold, 3),
                    'threshold_source': threshold_source,
                    'prob_by_hour': prob_by_hour}
        likely_j = int(signal_hours[0])
        band = np.where(
            np.isfinite(scored_hazard) & (scored_hazard >= band_threshold)
        )[0]
        early_j = int(band.min()) if band.size else likely_j
        late_j = int(band.max()) if band.size else likely_j
        return {
            'already_raining': already_raining,
            'declared': True,
            'likely_datetime': (dts.iloc[idx[0]].isoformat() if already_raining
                                else dts.iloc[idx[likely_j]].isoformat()),
            'earliest_datetime': dts.iloc[idx[early_j]].isoformat(),
            'latest_datetime': dts.iloc[idx[late_j]].isoformat(),
            'peak_hazard': round(peak, 3),
            'max_prob': max_prob,
            'declare_threshold': round(declare_threshold, 3),
            'threshold_source': threshold_source,
            'prob_by_hour': prob_by_hour,
        }
    except Exception as e:
        if _DEVICE_REQUEST == 'cuda':
            raise RuntimeError(f'Onset GPU inference failed: {e}') from e
        print(f"  [Onset] inference failed ({e}); bez onset bloka.")
        return None


def attach_hourly_onset_signal(corrected, onset_info):
    """Attach the dedicated +48h onset contract to corrected hourly rows.

    The hazard model supplies the full horizon. SKALA can only corroborate a
    short-lead signal already present in the radar-corrected hourly forecast;
    it is never carried forward into hours where radar has no skill window.
    """
    if not onset_info or not onset_info.get('prob_by_hour'):
        return corrected
    out = corrected.copy()
    hourly = {}
    for item in onset_info['prob_by_hour']:
        try:
            hourly[pd.Timestamp(item['datetime'])] = item
        except (KeyError, TypeError, ValueError):
            continue

    hazard_values = []
    cumulative_values = []
    threshold_values = []
    eligible_values = []
    signal_values = []
    skala_values = []
    source_values = []
    for _, row in out.iterrows():
        item = hourly.get(pd.Timestamp(row['datetime']))
        if item is None:
            hazard_values.append(np.nan)
            cumulative_values.append(np.nan)
            threshold_values.append(np.nan)
            eligible_values.append(np.nan)
            signal_values.append(np.nan)
            skala_values.append(np.nan)
            source_values.append(None)
            continue

        eligible = bool(item.get('eligible', False))
        signal = bool(item.get('signal', False))
        lead_h = int(item.get('lead_h', 999))
        skala_raw = row.get('rain_signal_skala_support', 0)
        rain_raw = row.get('rain_signal', 0)
        skala_support = bool(
            lead_h <= 2
            and pd.notna(skala_raw) and bool(skala_raw)
            and pd.notna(rain_raw) and bool(rain_raw)
        )
        # The replay supports SKALA as short-range corroboration. It can open
        # the onset signal only for an eligible dry-spell hour, never at +3h+.
        signal = signal or (eligible and skala_support)
        source = 'ITALIAMETEO_XGB_ONSET'
        if skala_support:
            source += '+SKALA'

        hazard_values.append(item.get('hazard'))
        cumulative_values.append(item.get('p_by_then'))
        threshold_values.append(item.get('threshold'))
        eligible_values.append(float(eligible))
        signal_values.append(float(signal))
        skala_values.append(float(skala_support))
        source_values.append(source)

    out['rain_onset_hazard'] = hazard_values
    out['rain_onset_probability_by_then'] = cumulative_values
    out['rain_onset_threshold'] = threshold_values
    out['rain_onset_eligible'] = eligible_values
    out['rain_onset_signal'] = signal_values
    out['rain_onset_skala_support'] = skala_values
    out['rain_onset_source'] = source_values
    return out


def compute_bias_drift(df, out_path=None):
    """NWP archives drift when upstream model versions change. Track
    per model x param MONTHLY mean bias; flag pairs whose last-3-month bias
    departs from the prior history by > 1.5 prior std."""
    drift = {}
    ym = df['datetime'].dt.to_period('M').astype(str)
    for param, info in TARGET_PARAMS.items():
        obs_col = info['obs']
        if obs_col not in df.columns:
            continue
        obs = pd.to_numeric(df[obs_col], errors='coerce')
        for m in MODELS:
            mc = f"{m}_{param}_model"
            if mc not in df.columns:
                continue
            err = pd.to_numeric(df[mc], errors='coerce') - obs
            g = err.groupby(ym).agg(['mean', 'count'])
            g = g[g['count'] >= 100]
            if len(g) < 8:
                continue
            recent = float(g['mean'].iloc[-3:].mean())
            prior = g['mean'].iloc[:-3]
            flag = bool(abs(recent - float(prior.mean())) > 1.5 * max(float(prior.std()), 1e-6))
            drift[f"{m}_{param}"] = {
                'monthly_bias': {str(k): round(float(v), 3) for k, v in g['mean'].items()},
                'recent3_mean': round(recent, 3),
                'prior_mean': round(float(prior.mean()), 3),
                'prior_std': round(float(prior.std()), 3),
                'drift_flag': flag,
            }
    flagged = sorted(k for k, v in drift.items() if v['drift_flag'])
    if flagged:
        print(f"  [Drift] WARN — {len(flagged)} model-param parova driftuje: "
              f"{', '.join(flagged[:8])}{'...' if len(flagged) > 8 else ''}")
    if out_path:
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(drift, f, indent=1)
        except Exception as e:
            print(f"  [Drift] save failed ({e})")
    return drift


def _snapshot_champion():
    """Snapshot current production artifacts to MODEL_DIR/_champion
    BEFORE a retrain overwrites them, and return the champion's stored results
    for the promotion gate. Rollback = copy _champion/* back."""
    import shutil
    champ_dir = os.path.join(MODEL_DIR, '_champion')
    try:
        if os.path.isdir(champ_dir):
            shutil.rmtree(champ_dir)
        os.makedirs(champ_dir, exist_ok=True)
        n = 0
        for fn in os.listdir(MODEL_DIR):
            fp = os.path.join(MODEL_DIR, fn)
            if os.path.isfile(fp) and fn.rsplit('.', 1)[-1] in ('json', 'joblib', 'txt'):
                shutil.copy2(fp, os.path.join(champ_dir, fn))
                n += 1
        print(f"  Champion snapshot: {n} fajlova -> {champ_dir}")
        rp = os.path.join(champ_dir, 'training_results.json')
        if os.path.exists(rp):
            with open(rp, encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"  Champion snapshot failed ({e})")
    return None


def _champion_gate(results, results_prev):
    """Lite guardrail: flag any target whose report-half MAE
    regressed >15% vs the stored champion. We deliberately do NOT auto-restore
    per-target artifacts (mixed MODEL_DIR would desync training_results.json);
    the loud verdict + _champion/ snapshot make rollback a copy away. The
    DM-tested comparison lives in eval_harness.py --eval-rolling."""
    verdicts = {}
    for p, r in results.items():
        prev = (results_prev or {}).get(p)
        # Skip targets without a comparable MAE on either side. Resumed targets
        # carry mae=None on purpose (production models are in-sample on the test
        # half — see resume_training.py), so there is nothing honest to compare.
        if r.get('mae') is None:
            verdicts[p] = 'skipped (mae=None, resumed; see rolling_eval.json)'
            continue
        if not prev or prev.get('mae') is None:
            verdicts[p] = 'promoted (no champion)'
            continue
        if float(r['mae']) > float(prev['mae']) * 1.15:
            verdicts[p] = (f"WARN regression {prev['mae']} -> {r['mae']} "
                           f"(rollback: trained_models_v2/_champion)")
        else:
            verdicts[p] = 'promoted'
    bad = {p: v for p, v in verdicts.items() if v.startswith('WARN')}
    if bad:
        print("\n  " + "!" * 64)
        print("  CHAMPION GATE — regresije:")
        for p, v in bad.items():
            print(f"    {p}: {v}")
        print("  " + "!" * 64)
    else:
        print("  Champion gate: svi targeti promoted.")
    return verdicts


def _append_run_history(results, df, gate_verdicts=None):
    """Lightweight experiment registry (one JSON entry per
    training run) — MLflow-lite for a solo workstation."""
    path = os.path.join(MODEL_DIR, 'runs_history.json')
    hist = []
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                hist = json.load(f)
        except Exception:
            hist = []
    entry = {
        'timestamp': local_now().isoformat(),
        'data_rows': int(len(df)),
        'data_end': str(df['datetime'].max()),
        'lead_stack': bool(LEAD_STACK_ENABLED),
        'trials': N_TRIALS,
        'metrics': {p: {'mae': r.get('mae'),
                        'quantile_crps': r.get('quantile_crps'),
                        'method': r.get('method'),
                        'rain_gate_mode': r.get('rain_gate_mode')}
                    for p, r in results.items()},
    }
    if gate_verdicts:
        entry['champion_gate'] = gate_verdicts
    hist.append(entry)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(hist, f, indent=1, ensure_ascii=False)
        print(f"  Run history: {len(hist)} zapisa -> {path}")
    except Exception as e:
        print(f"  Run history save failed ({e})")


QUANTILE_TARGETS = ('temperature_2m', 'wind_speed_10m', 'wind_gusts_10m', 'precipitation')

# targets that get synthetic lead-24/48h training rows from the
# previous-runs archive. Disable with FC_LEAD_STACK=0 (e.g. low-RAM machines).
LEAD_STACK_TARGETS = ('temperature_2m', 'wind_speed_10m', 'wind_gusts_10m', 'precipitation')
LEAD_STACK_ENABLED = os.environ.get('FC_LEAD_STACK', '1') not in ('0', 'false', 'False')


def build_lead_stacked_frames(hist_raw, bias_tables):
    """(Mlakar et al. 2024): one model pooled across lead times beats
    per-lead models in small data — but our archive rows are all short-lead,
    while live inference at +24-48h runs on long-lead inputs (train/serve lead
    mismatch). Fix: synthesize lead-36/60 rows where {M}_{v}_model holds the
    PREVIOUS run's value for the same valid hour (8 PREV_RUNS_VARS; everything
    else NaN — XGBoost sparsity handles it). Frames are bias-featured +
    engineered exactly like the base frame and carry lead_time=36/60.

    Must be called on the RAW merged frame (before apply_bias_features /
    engineer_features) with the train-only bias tables."""
    if not LEAD_STACK_ENABLED:
        print("  Lead-stack: ISKLJUČEN (FC_LEAD_STACK=0)")
        return []
    keep_base = [c for c in hist_raw.columns
                 if not c.endswith('_model')
                 and '_previous_day' not in c
                 and '_nbr_p' not in c]
    frames = []
    for lag, lead in (('previous_day1', 36.0), ('previous_day2', 60.0)):
        swapped = {}
        for m in MODELS:
            for v in PREV_RUNS_VARS:
                src = f'{m}_{v}_{lag}'
                if src in hist_raw.columns:
                    swapped[f'{m}_{v}_model'] = pd.to_numeric(hist_raw[src], errors='coerce')
        if len(swapped) < 8:
            continue
        slim = hist_raw[keep_base].copy()
        for k, v in swapped.items():
            slim[k] = v
        # the "one day older" run is what a lead-36 row would see as its own
        # previous_day1; lead-60 rows have no older archive at all
        if lag == 'previous_day1':
            for m in MODELS:
                for v in PREV_RUNS_VARS:
                    src2 = f'{m}_{v}_previous_day2'
                    if src2 in hist_raw.columns:
                        slim[f'{m}_{v}_previous_day1'] = pd.to_numeric(
                            hist_raw[src2], errors='coerce')
        model_cols = [c for c in slim.columns if c.endswith('_model')]
        valid = slim[model_cols].notna().any(axis=1)
        slim = slim[valid].copy()
        if len(slim) < 2000:
            continue
        slim = apply_bias_features(slim, bias_tables)
        slim['lead_time'] = lead
        slim = engineer_features(slim)
        keep = [c for c in slim.columns if slim[c].notna().any()]
        frames.append(slim[keep].reset_index(drop=True))
        print(f"  Lead-stack {lag}: {len(slim)} redova (lead={lead:.0f}h, "
              f"{len(keep)} kolona)")
    return frames


def _quantile_clock_split(X, y, datetimes, *, fraction=0.10,
                          min_fold_rows=500, min_fit_rows=1000,
                          embargo_hours=72):
    """Build fit/validation/calibration folds by valid timestamp.

    Lead stacking contributes several rows for one valid forecast hour. Split
    boundaries therefore operate on timestamp groups, while target sizes remain
    row based. Embargoes are measured with Timedelta rather than row counts.
    """
    n_rows = len(X)
    if len(y) != n_rows:
        raise ValueError('X/y duzine nijesu poravnate')
    if datetimes is None:
        if isinstance(X, pd.DataFrame) and 'datetime' in X.columns:
            datetimes = X['datetime']
        elif isinstance(getattr(X, 'index', None), pd.DatetimeIndex):
            datetimes = X.index
        else:
            raise ValueError('nema poravnatih valid datetime vrijednosti')
    if len(datetimes) != n_rows:
        raise ValueError('datetime/X duzine nijesu poravnate')

    raw_times = pd.Series(np.asarray(datetimes)).reset_index(drop=True)
    try:
        times = pd.to_datetime(raw_times, errors='coerce')
    except (TypeError, ValueError):
        times = pd.to_datetime(raw_times, errors='coerce', utc=True)
    # Mixed-offset timezone data can otherwise remain object dtype and cannot
    # be safely ordered/subtracted. Normalizing that exceptional case to UTC
    # preserves actual clock durations.
    if not (pd.api.types.is_datetime64_any_dtype(times.dtype)
            or isinstance(times.dtype, pd.DatetimeTZDtype)):
        times = pd.to_datetime(raw_times, errors='coerce', utc=True)
    if times.isna().any():
        raise ValueError(f'{int(times.isna().sum())} neispravnih datetime vrijednosti')

    order = times.sort_values(kind='mergesort').index.to_numpy()
    times = times.iloc[order].reset_index(drop=True)
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = pd.Series(y).iloc[order].reset_index(drop=True)

    target_rows = max(int(n_rows * fraction), int(min_fold_rows))

    def _tail_group_start(candidate_times, wanted_rows, fold_name):
        counts = candidate_times.value_counts(sort=False).sort_index()
        if int(counts.sum()) < wanted_rows:
            raise ValueError(
                f'nema dovoljno redova za {fold_name} '
                f'({int(counts.sum())} < {wanted_rows})'
            )
        reverse_cumulative = counts.iloc[::-1].cumsum().to_numpy()
        reverse_position = int(np.searchsorted(
            reverse_cumulative, wanted_rows, side='left'
        ))
        return counts.index[-(reverse_position + 1)]

    calibration_start = _tail_group_start(times, target_rows, 'kalibraciju')
    calibration_mask = times >= calibration_start

    embargo = pd.Timedelta(hours=embargo_hours)
    validation_latest = calibration_start - embargo
    validation_candidates = times[times <= validation_latest]
    validation_start = _tail_group_start(
        validation_candidates, target_rows, 'early-stop validaciju'
    )
    validation_mask = ((times >= validation_start)
                       & (times <= validation_latest))

    fit_latest = validation_start - embargo
    fit_mask = times <= fit_latest
    if int(fit_mask.sum()) < min_fit_rows:
        raise ValueError(
            f'nema dovoljno fit redova poslije clock-hour embargoa '
            f'({int(fit_mask.sum())} < {min_fit_rows})'
        )

    def _take(mask):
        positions = np.flatnonzero(mask.to_numpy())
        return (
            X_sorted.iloc[positions].reset_index(drop=True),
            y_sorted.iloc[positions].reset_index(drop=True),
            times.iloc[positions].reset_index(drop=True),
        )

    X_fit, y_fit, time_fit = _take(fit_mask)
    X_val, y_val, time_val = _take(validation_mask)
    X_cal, y_cal, time_cal = _take(calibration_mask)
    fit_val_gap = time_val.min() - time_fit.max()
    val_cal_gap = time_cal.min() - time_val.max()
    if fit_val_gap < embargo or val_cal_gap < embargo:
        raise AssertionError(
            f'quantile embargo je kraci od {embargo_hours}h '
            f'(fit-val={fit_val_gap}, val-cal={val_cal_gap})'
        )

    return {
        'X_fit': X_fit, 'y_fit': y_fit, 'time_fit': time_fit,
        'X_val': X_val, 'y_val': y_val, 'time_val': time_val,
        'X_cal': X_cal, 'y_cal': y_cal, 'time_cal': time_cal,
        'fit_val_gap': fit_val_gap, 'val_cal_gap': val_cal_gap,
    }


def _precip_clock_split(X, y, datetimes, *, embargo_hours=72):
    """Four chronological precipitation blocks with grouped valid times.

    ``fit`` trains models, ``val`` drives Optuna/early stopping, ``cal`` fits
    isotonic calibration, and ``gate`` selects thresholds and the trusted-vs-
    PoP gate. Three real 72-hour embargoes separate the blocks.
    """
    base = _quantile_clock_split(
        X, y, datetimes, fraction=0.12, min_fold_rows=1500,
        min_fit_rows=3000, embargo_hours=embargo_hours,
    )
    X_tail = base['X_cal']
    y_tail = base['y_cal']
    time_tail = base['time_cal'].reset_index(drop=True)
    target_gate_rows = max(int(len(time_tail) * 0.45), 500)
    counts = time_tail.value_counts(sort=False).sort_index()
    reverse_cumulative = counts.iloc[::-1].cumsum().to_numpy()
    reverse_position = int(np.searchsorted(
        reverse_cumulative, target_gate_rows, side='left'
    ))
    gate_start = counts.index[-(reverse_position + 1)]
    embargo = pd.Timedelta(hours=embargo_hours)
    calibration_latest = gate_start - embargo
    cal_mask = time_tail <= calibration_latest
    gate_mask = time_tail >= gate_start
    if int(cal_mask.sum()) < 500 or int(gate_mask.sum()) < 500:
        raise ValueError(
            'premali precipitation calibration/gate blok poslije embargoa '
            f"(cal={int(cal_mask.sum())}, gate={int(gate_mask.sum())})"
        )

    def _tail_take(mask):
        positions = np.flatnonzero(mask.to_numpy())
        return (
            X_tail.iloc[positions].reset_index(drop=True),
            y_tail.iloc[positions].reset_index(drop=True),
            time_tail.iloc[positions].reset_index(drop=True),
        )

    X_cal, y_cal, time_cal = _tail_take(cal_mask)
    X_gate, y_gate, time_gate = _tail_take(gate_mask)
    cal_gate_gap = time_gate.min() - time_cal.max()
    if cal_gate_gap < embargo:
        raise AssertionError(
            f'precip calibration-gate embargo je kraci od {embargo_hours}h'
        )
    return {
        'X_fit': base['X_fit'], 'y_fit': base['y_fit'],
        'time_fit': base['time_fit'],
        'X_val': base['X_val'], 'y_val': base['y_val'],
        'time_val': base['time_val'],
        'X_cal': X_cal, 'y_cal': y_cal, 'time_cal': time_cal,
        'X_gate': X_gate, 'y_gate': y_gate, 'time_gate': time_gate,
        'fit_val_gap': base['fit_val_gap'],
        'val_cal_gap': time_cal.min() - base['time_val'].max(),
        'cal_gate_gap': cal_gate_gap,
    }


def _train_target_quantiles(param, X_tr, y_tr, X_te_rep, y_te_rep,
                            train_datetimes=None):
    """multi-quantile LightGBM wrapped in CQR.

    Uses distinct chronological fit, early-stop validation, and conformal
    calibration folds, with 72h embargoes between them. Reported coverage/CRPS
    come from the untouched report half. Persists the bundle to MODEL_DIR for
    --skip-training reload. Returns the in-memory bundle dict or None."""
    n = len(X_tr)
    if n < 3000 or pd.Series(y_tr).notna().sum() < 2000:
        reason = f'nedovoljno podataka (rows={n}, valid_y={pd.Series(y_tr).notna().sum()})'
        print(f"    Quantile+CQR ({param}): {reason}")
        _invalidate_quantile_artifacts(param, reason)
        return None
    try:
        folds = _quantile_clock_split(X_tr, y_tr, train_datetimes)
    except (AssertionError, TypeError, ValueError) as exc:
        reason = f'neispravan hronoloski split: {exc}'
        print(f"    Quantile+CQR ({param}): {reason}")
        _invalidate_quantile_artifacts(param, reason)
        return None
    X_q_tr, y_q_tr = folds['X_fit'], folds['y_fit']
    X_q_val, y_q_val = folds['X_val'], folds['y_val']
    X_cal, y_cal = folds['X_cal'], folds['y_cal']
    print(
        f"    Quantile split ({param}): fit={len(X_q_tr)}, val={len(X_q_val)}, "
        f"cal={len(X_cal)} | clock gaps="
        f"{folds['fit_val_gap'] / pd.Timedelta(hours=1):.0f}h/"
        f"{folds['val_cal_gap'] / pd.Timedelta(hours=1):.0f}h"
    )
    try:
        quantile_device = dict(LIGHTGBM_DEVICE_PARAMS)
        if USING_GPU:
            quantile_device['max_bin'] = 63
        models = pf.train_quantile_models(
            X_q_tr, y_q_tr, X_q_val, y_q_val, lgb_params=quantile_device
        )
        offsets = pf.cqr_calibrate(models, X_cal, y_cal)
    except Exception as _e:
        if _DEVICE_REQUEST == 'cuda':
            _invalidate_quantile_artifacts(param, f'GPU trening nije uspio: {_e}')
            raise RuntimeError(f'LightGBM GPU quantile trening nije uspio: {_e}') from _e
        if _DEVICE_REQUEST == 'auto' and USING_GPU:
            try:
                print(f"    Quantile+CQR ({param}): GPU fail ({_e}); retry na CPU")
                models = pf.train_quantile_models(
                    X_q_tr, y_q_tr, X_q_val, y_q_val,
                    lgb_params={'device_type': 'cpu'},
                )
                offsets = pf.cqr_calibrate(models, X_cal, y_cal)
            except Exception as cpu_error:
                print(f"    Quantile+CQR ({param}): CPU retry neuspio ({cpu_error})")
                _invalidate_quantile_artifacts(
                    param, f'GPU i CPU trening nijesu uspjeli: {cpu_error}'
                )
                return None
        else:
            print(f"    Quantile+CQR ({param}): trening neuspio ({_e})")
            _invalidate_quantile_artifacts(param, f'trening nije uspio: {_e}')
            return None
    try:
        lower = 0.0 if param in ('wind_speed_10m', 'wind_gusts_10m', 'precipitation') else None
        qdf_te = pf.predict_quantiles(models, X_te_rep, offsets=offsets, lower_bound=lower)
        ok = y_te_rep.notna().values
        crps = pf.crps_from_quantiles(y_te_rep.values[ok], qdf_te[ok].reset_index(drop=True))
        cov90, w90 = pf.coverage_width(y_te_rep.values[ok], qdf_te['q05'].values[ok], qdf_te['q95'].values[ok])
        cov50, w50 = pf.coverage_width(y_te_rep.values[ok], qdf_te['q25'].values[ok], qdf_te['q75'].values[ok])
    except Exception as exc:
        _invalidate_quantile_artifacts(param, f'evaluacija nije uspjela: {exc}')
        raise
    print(f"    Quantile+CQR ({param}): CRPS={crps:.3f} | cov90={cov90:.3f} "
          f"(w={w90:.2f}) | cov50={cov50:.3f} (w={w50:.2f})")
    try:
        _promote_quantile_artifacts(param, models, offsets, list(X_tr.columns))
    except Exception as _e:
        print(f"    Quantile bundle save failed ({_e})")
        _invalidate_quantile_artifacts(param, f'save/promocija nije uspjela: {_e}')
        return None
    return {'models': models, 'offsets': offsets, 'features': list(X_tr.columns),
            'crps': round(float(crps), 4), 'cov90': round(float(cov90), 3),
            'cov50': round(float(cov50), 3)}


def train_all_models(df, lead_frames=None, only_targets=None, snapshot=True):
    """Unified training: all params use full 50K dataset. No splits.
    - Precipitation: two-stage (cls+reg) + optional blend
    - Everything else: residual+blended (direct/residual/blend)
    lead_frames: optional engineered lead-36/60 frames pooled into
    the TRAIN portion of LEAD_STACK_TARGETS; the test set stays day-0 only so
    reported MAEs remain comparable across runs.
    only_targets: if set, train ONLY these params and MERGE into existing
    on-disk feature_lists/results (resume after a crash). snapshot=False then
    keeps the existing _champion snapshot intact."""
    print("\n[3/6] Treniranje XGBoost modela...")

    _fl_path = os.path.join(MODEL_DIR, 'feature_lists.json')
    _res_path = os.path.join(MODEL_DIR, 'training_results.json')

    # Snapshot the reigning champion before overwriting anything.
    # On RESUME (only_targets set) we must NOT re-snapshot: trained_models_v2
    # already holds a mix of new + old models, so a fresh snapshot would corrupt
    # the champion. Read the existing _champion snapshot for the gate instead.
    if snapshot and only_targets is None:
        results_prev = _snapshot_champion()
    else:
        results_prev = None
        _champ_rp = os.path.join(MODEL_DIR, '_champion', 'training_results.json')
        if os.path.exists(_champ_rp):
            try:
                with open(_champ_rp, encoding='utf-8') as _cf:
                    results_prev = json.load(_cf)
            except Exception:
                results_prev = None
    # per-model bias drift report
    compute_bias_drift(df, os.path.join(OUTPUT_DIR, 'bias_drift.json'))

    feature_cols = get_feature_columns(df)
    print(f"  Feature-a za treniranje: {len(feature_cols)} (Optuna trials: {N_TRIALS})")

    trained = {}
    results = {}
    report_predictions = {}
    feature_lists_acc = {}
    # RESUME: preload existing on-disk metadata so per-target writes MERGE with
    # (rather than clobber) the targets completed in a previous run.
    if only_targets is not None:
        for _pth, _acc in ((_res_path, results), (_fl_path, feature_lists_acc)):
            if os.path.exists(_pth):
                try:
                    with open(_pth, encoding='utf-8') as _pf:
                        _acc.update(json.load(_pf))
                except Exception:
                    pass
        print(f"  RESUME: treniram samo {sorted(only_targets)}; "
              f"zadržavam {len(results)} postojećih rezultata")

    def _persist_artifacts():
        """Atomically snapshot metadata after each completed target."""
        fl = dict(feature_lists_acc)
        fl.update({k: v['features'] for k, v in trained.items()})
        _write_json_atomic(_fl_path, fl)
        _write_json_atomic(_res_path, results, indent=2, ensure_ascii=False)

    for param, info in TARGET_PARAMS.items():
        if only_targets is not None and param not in only_targets:
            continue
        obs_col = info['obs']
        if obs_col not in df.columns:
            print(f"  {info['display']:20s} --- SKIP (nema obs)")
            if param in QUANTILE_TARGETS:
                _invalidate_quantile_artifacts(param, 'ciljna observation kolona ne postoji')
            continue

        y = pd.to_numeric(df[obs_col], errors='coerce')
        valid = y.notna()
        if param == 'cloud_cover':
            valid = valid & (df.get('is_daytime', pd.Series(1, index=df.index)) > 0)

        df_v = df[valid].copy()
        y_v = y[valid]

        if len(df_v) < 500:
            print(f"  {info['display']:20s} --- SKIP ({len(df_v)} redova)")
            if param in QUANTILE_TARGETS:
                _invalidate_quantile_artifacts(
                    param, f'nedovoljno validnih ciljnih redova ({len(df_v)})'
                )
            continue

        tr = df_v['datetime'] < SPLIT_DATE
        te = df_v['datetime'] >= SPLIT_DATE

        # Freeze feature eligibility from the training period only; report-row
        # availability must not decide the schema being evaluated.
        n_train_rows = int(tr.sum())
        vf = [c for c in feature_cols if c in df_v.columns
              and df_v.loc[tr, c].notna().sum() > n_train_rows * 0.15]

        # Per-target model subset: e.g. for wind, keep only ITALIAMETEO/KNMI/DMI features
        # (high-res LAMs analog to MARINE_WIND_MODELS) plus generic engineered features.
        subset_models = FEATURE_MODEL_SUBSET.get(param)
        if subset_models:
            all_models_set = set(MODELS)
            def belongs_to_subset(col):
                for m in all_models_set:
                    if col.startswith(f"{m}_") or col == f"is_{m}_available":
                        return m in subset_models
                return True  # generic feature (not model-specific) - keep
            vf_before = len(vf)
            vf = [c for c in vf if belongs_to_subset(c)]
            print(f"    {info['display']}: feature subset filter ({subset_models}): "
                  f"{vf_before} -> {len(vf)} features")

        # NaN passthrough: let XGBoost's native sparsity-aware split-finding handle missing data
        # is the "single easiest win" — 5-15% MAE improvement)
        X_tr, y_tr = df_v.loc[tr, vf], y_v[tr]
        X_te, y_te = df_v.loc[te, vf], y_v[te]

        # df_tr_frame carries datetime/ens/regime columns for the TRAIN rows; it
        # is replaced by an augmented (and chronologically re-sorted) frame when
        # lead-stacking applies, so every downstream consumer stays aligned.
        df_tr_frame = df_v.loc[tr]

        # --- pool synthetic lead-36/60 rows into TRAIN only ---
        if lead_frames and param in LEAD_STACK_TARGETS:
            _aug_X, _aug_y, _aug_f = [], [], []
            _meta_cols = ['datetime', f'{param}_ens_mean',
                          'is_summer', 'regime_ne', 'regime_wet']
            for lf in lead_frames:
                if obs_col not in lf.columns:
                    continue
                y_lf = pd.to_numeric(lf[obs_col], errors='coerce')
                v_lf = y_lf.notna() & (lf['datetime'] < SPLIT_DATE)
                if v_lf.sum() < 1000:
                    continue
                sub = lf[v_lf]
                X_lf = pd.DataFrame({
                    c: (pd.to_numeric(sub[c], errors='coerce') if c in sub.columns
                        else pd.Series(np.nan, index=sub.index))
                    for c in vf
                })
                f_lf = pd.DataFrame({
                    c: (sub[c] if c in sub.columns else pd.Series(np.nan, index=sub.index))
                    for c in _meta_cols
                })
                _aug_X.append(X_lf)
                _aug_y.append(y_lf[v_lf])
                _aug_f.append(f_lf)
            if _aug_X:
                _base_f = df_tr_frame[[c for c in _meta_cols if c in df_tr_frame.columns]]
                X_cat = pd.concat([X_tr] + _aug_X, axis=0, ignore_index=True)
                y_cat = pd.concat([y_tr] + _aug_y, axis=0, ignore_index=True)
                f_cat = pd.concat([_base_f] + _aug_f, axis=0, ignore_index=True)
                _order = np.argsort(f_cat['datetime'].values, kind='mergesort')
                X_tr = X_cat.iloc[_order].reset_index(drop=True)
                y_tr = y_cat.iloc[_order].reset_index(drop=True)
                df_tr_frame = f_cat.iloc[_order].reset_index(drop=True)
                n_stacked = sum(len(a) for a in _aug_X)
                print(f"    Lead-stack ({param}): +{n_stacked} redova "
                      f"(train {len(X_tr) - n_stacked} -> {len(X_tr)})")

        # --- Dew point deficit target ---
        # Predict T − Td (≥ 0) instead of Td directly; derive Td = T_corrected − deficit.
        # The deficit is physically constrained ≥ 0, improving learnability.
        dew_deficit_mode = False
        if param == 'dew_point_2m' and 'temperature_2m_obs' in df_v.columns:
            t_obs_tr = pd.to_numeric(df_v.loc[tr, 'temperature_2m_obs'], errors='coerce')
            t_obs_te = pd.to_numeric(df_v.loc[te, 'temperature_2m_obs'], errors='coerce')
            valid_deficit_tr = t_obs_tr.notna() & y_tr.notna()
            valid_deficit_te = t_obs_te.notna() & y_te.notna()
            if valid_deficit_tr.sum() > 300 and valid_deficit_te.sum() > 50:
                dew_deficit_mode = True
                y_tr_orig_dew, y_te_orig_dew = y_tr.copy(), y_te.copy()
                y_tr = (t_obs_tr - y_tr).clip(lower=0)
                y_te_deficit = (t_obs_te - y_te).clip(lower=0)
                y_te = y_te_deficit
                print(f"    Dew point: using deficit target (T - Td ≥ 0)")

        # --- CSI target for solar radiation ---
        # Train on clear-sky index (CSI = GHI/GHI_clearsky) instead of raw irradiance.
        # Back-transform predictions to W/m² at evaluation and production time.
        csi_mode = False
        clear_sky_tr = clear_sky_te = None
        if param == 'shortwave_radiation' and 'clear_sky_rad' in df_v.columns:
            cs_tr = compute_clear_sky(df_v.loc[tr, 'datetime']).values
            cs_te = compute_clear_sky(df_v.loc[te, 'datetime']).values
            # Only use CSI where clear sky > 20 W/m² (daytime)
            daytime_tr = cs_tr > 20
            daytime_te = cs_te > 20
            if daytime_tr.sum() > 300 and daytime_te.sum() > 50:
                csi_mode = True
                clear_sky_tr = cs_tr
                clear_sky_te = cs_te
                y_tr_csi = y_tr.copy()
                y_tr_csi[daytime_tr] = (y_tr.values[daytime_tr] / cs_tr[daytime_tr].clip(min=1)).clip(0, 1.5)
                y_tr_csi[~daytime_tr] = 0.0
                y_te_csi = y_te.copy()
                y_te_csi[daytime_te] = (y_te.values[daytime_te] / cs_te[daytime_te].clip(min=1)).clip(0, 1.5)
                y_te_csi[~daytime_te] = 0.0
                y_tr_orig, y_te_orig = y_tr, y_te  # save for back-transform MAE eval
                y_tr, y_te = y_tr_csi, y_te_csi
                print(f"    Solar: using CSI target (clear-sky index)")

        # Residuals, stacks, and blends must use a baseline expressed in the
        # same target space as y. Raw dew point (deg C) cannot be added to a
        # dew-deficit prediction, nor W/m2 radiation to a CSI prediction.
        raw_ens_col = f'{param}_ens_mean'
        model_ens_col = raw_ens_col
        if dew_deficit_mode:
            model_ens_col = f'{param}_target_baseline'
            temp_ens = pd.to_numeric(df_v.get('temperature_2m_ens_mean'), errors='coerce')
            dew_ens = pd.to_numeric(df_v.get(raw_ens_col), errors='coerce')
            df_v[model_ens_col] = (temp_ens - dew_ens).clip(lower=0)
            df_tr_frame[model_ens_col] = df_v.loc[df_tr_frame.index, model_ens_col].values
        elif csi_mode:
            model_ens_col = f'{param}_target_baseline'
            raw_solar = pd.to_numeric(df_v.get(raw_ens_col), errors='coerce')
            clear_sky_all = compute_clear_sky(df_v['datetime']).clip(lower=1)
            csi_baseline = (raw_solar / clear_sky_all).clip(lower=0, upper=1.5)
            csi_baseline[clear_sky_all <= 20] = 0.0
            df_v[model_ens_col] = csi_baseline
            df_tr_frame[model_ens_col] = df_v.loc[df_tr_frame.index, model_ens_col].values

        if len(X_tr) < 300 or len(X_te) < 50:
            print(f"  {info['display']:20s} --- SKIP (train={len(X_tr)}, test={len(X_te)})")
            if param in QUANTILE_TARGETS:
                _invalidate_quantile_artifacts(
                    param, f'nedovoljan train/test split ({len(X_tr)}/{len(X_te)})'
                )
            continue

        # --- LEAKAGE FIX: the test set must not be used for BOTH choosing the
        # configuration AND reporting accuracy. Split it chronologically:
        #   SELECTION half -> picks method / blend alpha / stack weights
        #   REPORT half     -> ONLY for the prijavljena MAE; never drives a choice
        # Base learners still train solely on the train period (< SPLIT_DATE), so
        # the selection half is genuinely out-of-sample. The production retrain
        # below still uses the FULL test set (more data is better for shipping).
        # Tiny test sets fall back to the old behaviour (no meaningful split).
        _df_te = df_v.loc[te]
        if len(X_te) >= 200:
            _h = len(X_te) // 2
            _sel_sl, _rep_sl = slice(0, _h), slice(_h, None)
        else:
            _sel_sl, _rep_sl = slice(None), slice(None)
        X_te_sel, y_te_sel, df_te_sel = X_te.iloc[_sel_sl], y_te.iloc[_sel_sl], _df_te.iloc[_sel_sl]
        X_te_rep, y_te_rep, df_te_rep = X_te.iloc[_rep_sl], y_te.iloc[_rep_sl], _df_te.iloc[_rep_sl]

        if param == 'precipitation':
            precip_folds = _precip_clock_split(
                X_tr, y_tr, df_tr_frame['datetime']
            )
            X_train_p, y_train_p = precip_folds['X_fit'], precip_folds['y_fit']
            X_val_p, y_val_p = precip_folds['X_val'], precip_folds['y_val']
            X_cal_p, y_cal_p = precip_folds['X_cal'], precip_folds['y_cal']
            X_gate_p, y_gate_p = precip_folds['X_gate'], precip_folds['y_gate']
            print(
                f"    Precip split: fit={len(X_train_p)}, val={len(X_val_p)}, "
                f"cal={len(X_cal_p)}, gate={len(X_gate_p)} | clock gaps="
                f"{precip_folds['fit_val_gap'] / pd.Timedelta(hours=1):.0f}h/"
                f"{precip_folds['val_cal_gap'] / pd.Timedelta(hours=1):.0f}h/"
                f"{precip_folds['cal_gate_gap'] / pd.Timedelta(hours=1):.0f}h"
            )
            # Method selection inside the two-stage happens on the SELECTION half.
            precip_result = _train_precipitation_twostage(
                X_train_p, y_train_p, X_te_sel, y_te_sel,
                X_val_p, y_val_p, X_cal_p, y_cal_p,
                X_gate_p, y_gate_p, vf,
            )
            rmse = precip_result['rmse']

            ens_col = f'{param}_ens_mean'
            blend_alpha = 1.0
            RAIN_THRESH = CORRECTED_RAIN_THRESHOLD_MM
            best_method = precip_result['best_method']
            _iso = precip_result.get('iso_calibrator')
            thresh = precip_result['threshold']

            # Reproduce the production precip prediction for an arbitrary X
            # (mirrors apply_correction's method dispatch + clamping). Used to
            # evaluate the chosen config on both the selection and report halves.
            def _precip_xgb_pred(X_sub):
                cls_raw = precip_result['cls_model'].predict_proba(X_sub)[:, 1]
                cls_p = _iso.transform(cls_raw) if _iso is not None else cls_raw
                if precip_result.get('use_sqrt', False):
                    reg_p = np.square(np.clip(precip_result['reg_model'].predict(X_sub), 0, None))
                else:
                    reg_p = np.clip(precip_result['reg_model'].predict(X_sub), 0, None)
                single_p = np.clip(precip_result['single_model'].predict(X_sub), 0, None)
                single_p[single_p < RAIN_THRESH] = 0.0
                if best_method == 'hard':
                    xp = np.where(cls_p >= thresh, reg_p, 0.0)
                elif best_method == 'soft':
                    xp = cls_p * reg_p
                elif best_method == 'sharp':
                    xp = np.where(cls_p >= thresh, 0.7 * reg_p + 0.3 * single_p, single_p * cls_p)
                elif best_method == 'adaptive':
                    conf = np.abs(cls_p - 0.5) * 2
                    xp = np.where(cls_p >= thresh, conf * reg_p + (1 - conf) * single_p, (1 - conf) * single_p * 0.5)
                elif best_method == 'tweedie' and precip_result.get('tweedie_model') is not None:
                    xp = np.clip(precip_result['tweedie_model'].predict(X_sub), 0, None)
                else:
                    xp = single_p
                return _clamp_precip_prediction(xp, X_sub)

            # Choose the blend alpha on the SELECTION half only.
            if ens_col in df_v.columns:
                ens_sel = pd.to_numeric(df_te_sel[ens_col], errors='coerce').fillna(0).values
                xgb_sel = _precip_xgb_pred(X_te_sel)
                base_mae_sel = mean_absolute_error(y_te_sel, xgb_sel)
                b_alpha, b_mae_sel = 1.0, base_mae_sel
                for candidate_alpha in np.arange(0.50, 1.01, 0.025):
                    candidate = _clamp_precip_prediction(
                        candidate_alpha * xgb_sel + (1 - candidate_alpha) * ens_sel,
                        X_te_sel,
                    )
                    candidate_mae = mean_absolute_error(y_te_sel, candidate)
                    if candidate_mae < b_mae_sel:
                        b_alpha, b_mae_sel = candidate_alpha, candidate_mae
                if b_mae_sel < base_mae_sel:
                    blend_alpha = b_alpha
                    precip_result['blend_alpha'] = blend_alpha
                    print(f"    Blend improved (selection half): alpha={b_alpha:.3f}, "
                          f"MAE {base_mae_sel:.3f}->{b_mae_sel:.3f}")

            # A second, dedicated amount blend cooperates with the trusted
            # ICON-2I signal.  It is accepted only when it lowers MAE on the
            # selection half; otherwise alpha remains 1 and behavior is
            # unchanged.  The hard rain/no-rain policy is evaluated separately.
            trusted_amount_alpha = 1.0
            trusted_col = f'{TRUSTED_RAIN_MODEL}_precipitation_model'
            if trusted_col in X_te_sel.columns:
                amount_sel = xgb_sel.copy()
                if blend_alpha < 1.0 and ens_col in df_v.columns:
                    amount_sel = _clamp_precip_prediction(
                        blend_alpha * xgb_sel + (1 - blend_alpha) * ens_sel,
                        X_te_sel,
                    )
                trusted_sel = pd.to_numeric(
                    X_te_sel[trusted_col], errors='coerce'
                ).to_numpy(dtype=float)
                valid_trusted = np.isfinite(trusted_sel)
                best_amount_mae = mean_absolute_error(y_te_sel, amount_sel)
                for candidate_alpha in np.arange(0.50, 1.01, 0.025):
                    candidate = amount_sel.copy()
                    candidate[valid_trusted] = (
                        candidate_alpha * amount_sel[valid_trusted]
                        + (1 - candidate_alpha) * trusted_sel[valid_trusted]
                    )
                    candidate = _clamp_precip_prediction(candidate, X_te_sel)
                    candidate_mae = mean_absolute_error(y_te_sel, candidate)
                    if candidate_mae < best_amount_mae:
                        trusted_amount_alpha = float(candidate_alpha)
                        best_amount_mae = candidate_mae
                precip_result['trusted_amount_alpha'] = trusted_amount_alpha
                if trusted_amount_alpha < 1.0:
                    print(f"    Trusted amount blend improved (selection half): "
                          f"alpha={trusted_amount_alpha:.3f}, "
                          f"MAE={best_amount_mae:.3f}")

            # Report on the REPORT half with the chosen method + alpha (no further choice).
            xgb_rep = _precip_xgb_pred(X_te_rep)
            cls_rep_raw = precip_result['cls_model'].predict_proba(X_te_rep)[:, 1]
            cls_rep_pop = (
                _iso.transform(cls_rep_raw) if _iso is not None else cls_rep_raw
            )
            y_cls_rep = (y_te_rep >= RAIN_THRESH).astype(int).values
            report_cls_metrics = meteorological_metrics(
                y_cls_rep, (cls_rep_pop >= thresh).astype(int),
                p_proba=cls_rep_pop,
            )
            report_corp = pf.corp_reliability(
                y_cls_rep.astype(float), cls_rep_pop
            )
            precip_result['report_metrics_full'] = report_cls_metrics
            precip_result['report_corp'] = report_corp
            print(
                f"    Untouched report PoP: Brier={report_cls_metrics['brier']:.4f}, "
                f"BSS={report_cls_metrics['brier_skill_score']:.3f}, "
                f"RelRMSE={report_cls_metrics['reliability_rmse']:.4f}, "
                f"POD={report_cls_metrics['pod']:.3f}, "
                f"FAR={report_cls_metrics['far']:.3f}"
            )
            if blend_alpha < 1.0 and ens_col in df_v.columns:
                ens_rep = pd.to_numeric(df_te_rep[ens_col], errors='coerce').fillna(0).values
                final_rep = _clamp_precip_prediction(
                    blend_alpha * xgb_rep + (1 - blend_alpha) * ens_rep,
                    X_te_rep,
                )
            else:
                final_rep = xgb_rep
            if trusted_amount_alpha < 1.0 and trusted_col in X_te_rep.columns:
                trusted_rep = pd.to_numeric(
                    X_te_rep[trusted_col], errors='coerce'
                ).to_numpy(dtype=float)
                valid_trusted = np.isfinite(trusted_rep)
                final_rep[valid_trusted] = (
                    trusted_amount_alpha * final_rep[valid_trusted]
                    + (1 - trusted_amount_alpha) * trusted_rep[valid_trusted]
                )
                final_rep = _clamp_precip_prediction(final_rep, X_te_rep)
            mae = mean_absolute_error(y_te_rep, final_rep)
            rmse = np.sqrt(mean_squared_error(y_te_rep, final_rep))

            best_mae, best_m = float('inf'), ""
            for m in MODELS:
                mc = f"{m}_{param}_model"
                if mc in df_v.columns:
                    mv = pd.to_numeric(df_te_rep[mc], errors='coerce')
                    vv = mv.notna() & y_te_rep.notna()
                    if vv.sum() > 50:
                        mm = mean_absolute_error(y_te_rep[vv], mv[vv])
                        if mm < best_mae:
                            best_mae, best_m = mm, m

            ens_mae = float('inf')
            if ens_col in df_v.columns:
                ev = pd.to_numeric(df_te_rep[ens_col], errors='coerce')
                vv = ev.notna() & y_te_rep.notna()
                if vv.sum() > 50:
                    ens_mae = mean_absolute_error(y_te_rep[vv], ev[vv])

            impr = (best_mae - mae) / best_mae * 100 if best_mae < float('inf') else 0
            print(f"  {info['display']:20s} MAE: {mae:.3f}{info['unit']:5s} "
                  f"| best: {best_mae:.3f} ({best_m[:8]:8s}) "
                  f"| ens: {ens_mae:.3f} "
                  f"| +{impr:.1f}%")

            # --- Production retrain: retrain on ALL data (train+test) ---
            print(f"    Retraining {info['display']} na SVIM podacima za produkciju...")
            X_all = pd.concat([X_tr, X_te], axis=0)
            y_all = pd.concat([y_tr, y_te], axis=0)
            rain_mask_all = y_all >= RAIN_THRESH

            # The focal classifier, isotonic calibrator, threshold, and PoP blend
            # are one calibrated bundle. Keep that exact classifier instead of
            # refitting it on rows that defined its calibration transform.
            cls_prod = precip_result['cls_model']
            _focal_g = float(precip_result['cls_hp_final'].get('focal_gamma', 2.0))
            _focal_a = float(precip_result['cls_hp_final'].get('focal_alpha', 0.25))
            _n_round = int(precip_result['cls_hp_final'].get('n_estimators', 1000))

            # Retrain regressor on all data
            if precip_result.get('use_sqrt', False) and rain_mask_all.sum() >= 100:
                reg_prod = _new_xgb_regressor(**precip_result['reg_hp_final'])
                reg_prod.fit(X_all[rain_mask_all], np.sqrt(y_all[rain_mask_all]), verbose=False)
            else:
                reg_prod = _new_xgb_regressor(**precip_result['reg_hp_final'])
                reg_prod.fit(X_all, y_all, verbose=False)

            # Retrain single model on all data
            single_prod = _new_xgb_regressor(**precip_result['single_hp_final'])
            single_prod.fit(X_all, y_all, verbose=False)

            # Retrain Tweedie model on all data
            tweedie_prod = _new_xgb_regressor(**precip_result['tweedie_hp_final'])
            tweedie_prod.fit(X_all, y_all.clip(lower=0), verbose=False)

            # Update precip_result with production models
            precip_result['cls_model'] = cls_prod
            precip_result['reg_model'] = reg_prod
            precip_result['single_model'] = single_prod
            precip_result['tweedie_model'] = tweedie_prod
            print(f"    Production retrain: cls=calibrated bundle retained, "
                  f"reg={rain_mask_all.sum()}, single={len(X_all)} redova")

            cls_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_cls.json"))
            reg_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_reg.json"))
            single_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}.json"))
            tweedie_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_tweedie.json"))

            # persist focal-loss hyperparams as a sidecar JSON next to the
            # classifier. Skip-training reload needs gamma/alpha to wrap the Booster
            # in the same adapter (probabilities from raw margins via sigmoid).
            try:
                _focal_sidecar = os.path.join(MODEL_DIR, f"xgb_{param}_cls_focal.json")
                _write_json_atomic(
                    _focal_sidecar,
                    {'focal_gamma': _focal_g, 'focal_alpha': _focal_a,
                     'n_estimators': _n_round},
                )
            except Exception as _e:
                print(f"    WARN: couldn't persist focal sidecar: {_e}")

            # persist isotonic calibrator (for inference + skip-training mode)
            _iso = precip_result.get('iso_calibrator')
            if _iso is not None:
                import joblib
                joblib.dump(_iso, os.path.join(MODEL_DIR, f"xgb_{param}_iso_calibrator.joblib"))
            else:
                _remove_if_exists(
                    os.path.join(MODEL_DIR, f"xgb_{param}_iso_calibrator.joblib")
                )

            # persist the PoP blend (LR + spec) for reload
            _pb = precip_result.get('pop_blend')
            if _pb is not None:
                import joblib
                joblib.dump(_pb['lr'], os.path.join(MODEL_DIR, f"xgb_{param}_pop_blend.joblib"))
                _write_json_atomic(
                    os.path.join(MODEL_DIR, f"xgb_{param}_pop_blend.json"),
                    {'cols': _pb['cols'], 'tau': _pb['tau'], 'mode': _pb['mode']},
                )
            else:
                _remove_if_exists(
                    os.path.join(MODEL_DIR, f"xgb_{param}_pop_blend.joblib"),
                    os.path.join(MODEL_DIR, f"xgb_{param}_pop_blend.json"),
                )

            trained[param] = {
                'precip_info': precip_result,
                'features': vf,
                'mae': mae, 'rmse': rmse,
                'best_model': best_m, 'best_model_mae': best_mae,
                'ensemble_mae': ens_mae, 'improvement': impr,
            }
            results[param] = {
                'mae': round(mae, 3), 'rmse': round(rmse, 3),
                'unit': info['unit'], 'display': info['display'],
                'improvement': round(impr, 1),
                'best_model': best_m, 'best_model_mae': round(best_mae, 3),
                'ensemble_mae': round(ens_mae, 3),
                'method': precip_result['best_method'],
                'is_residual': False,
                'blend_alpha': float(precip_result.get('blend_alpha', 1.0)),
                'trusted_amount_alpha': float(
                    precip_result.get('trusted_amount_alpha', 1.0)
                ),
                'threshold': float(precip_result['threshold']),
                'use_sqrt': bool(precip_result.get('use_sqrt', False)),
                'is_precip': True,
                'rain_gate_mode': (precip_result.get('pop_blend') or {}).get('mode', 'trusted'),
                'pop_brier': round(float(report_cls_metrics['brier']), 5),
                'pop_brier_skill': round(float(report_cls_metrics['brier_skill_score']), 4),
                'pop_reliability_rmse': round(float(report_cls_metrics['reliability_rmse']), 5),
                'pop_pod': round(float(report_cls_metrics['pod']), 4),
                'pop_far': round(float(report_cls_metrics['far']), 4),
            }
            # predictive distribution for precipitation amounts
            qb = _train_target_quantiles(
                param, X_tr, y_tr, X_te_rep, y_te_rep,
                train_datetimes=df_tr_frame['datetime'],
            )
            if qb is not None:
                trained[param]['quantiles'] = qb
                results[param]['quantile_crps'] = qb['crps']
                results[param]['quantile_cov90'] = qb['cov90']
            _persist_artifacts()  # incremental save (crash-safe)
            continue

        # Default HP (used as fallback; Optuna will search for better ones)
        if param in ('temperature_2m', 'dew_point_2m', 'pressure_msl'):
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.025,
                      subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.8,
                      reg_alpha=0.05, reg_lambda=1.0, min_child_weight=5, gamma=0.02,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)
        elif param in ('cloud_cover', 'shortwave_radiation'):
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.022,
                      subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.7,
                      reg_alpha=0.1, reg_lambda=1.3, min_child_weight=7, gamma=0.04,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)
        elif param == 'relative_humidity_2m':
            hp = dict(n_estimators=1000, max_depth=6, learning_rate=0.025,
                      subsample=0.8, colsample_bytree=0.55, colsample_bylevel=0.75,
                      reg_alpha=0.08, reg_lambda=1.2, min_child_weight=6, gamma=0.03,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)
        else:  # wind_speed_10m, wind_gusts_10m
            hp = dict(n_estimators=1000, max_depth=5, learning_rate=0.02,
                      subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.7,
                      reg_alpha=0.15, reg_lambda=1.8, min_child_weight=8, gamma=0.08,
                      objective='reg:absoluteerror', random_state=42, n_jobs=-1,
                      early_stopping_rounds=30)

        # Compute temporal sample weights (exponential decay — recent data weighted more)
        # half-life should be tuned, not fixed. We expose it as part of model selection.
        train_datetimes = df_tr_frame['datetime']
        sample_weight = _compute_sample_weights(y_tr, train_datetimes, decay_half_life_days=365)
        # stacked long-lead rows count half
        if 'lead_time' in X_tr.columns:
            _lt = pd.to_numeric(X_tr['lead_time'], errors='coerce').fillna(12).values
            sample_weight = sample_weight * np.where(_lt > 24, 0.5, 1.0)

        ens_col = model_ens_col
        # Method/blend/stack chosen on the SELECTION half (out-of-sample, not the
        # reported set). Reporting below uses the untouched REPORT half.
        rb_result = _train_residual_blended(
            X_tr, y_tr, X_te_sel, y_te_sel, hp, param, ens_col,
            df_tr_frame, df_te_sel,
            use_optuna=True, sample_weight=sample_weight,
            train_datetimes=train_datetimes
        )
        method_str = rb_result['method']
        _deploy_ridge_meta = method_str == 'ridge_meta'

        # Use selected features if feature selection was applied
        sel_feats = rb_result.get('selected_features')
        # Report on the REPORT half (never used to choose the config above).
        X_te_eval = X_te_rep[sel_feats] if sel_feats else X_te_rep
        y_pred = _predict_nonprecip_bundle(rb_result, X_te_eval, df_te_rep, ens_col)

        if param == 'relative_humidity_2m':
            y_pred = np.clip(y_pred, 0, 100)
        elif param == 'cloud_cover':
            y_pred = _postprocess_cloud_prediction(y_pred, df_te_rep)
        elif param in ['wind_speed_10m', 'wind_gusts_10m', 'shortwave_radiation']:
            y_pred = np.clip(y_pred, 0, None)

        # Evaluation target for the report half; y_te is not mutated here
        # (the old code did `y_te = y_te_orig`, which left y_tr in CSI/deficit
        # space but y_te in original space → the production retrain below then
        # trained on a MIXED target). Keep y_te in model space; evaluate against
        # a separate y_eval in observation space.
        y_eval = y_te_rep

        # --- CSI back-transform: convert CSI predictions back to W/m² for evaluation ---
        if csi_mode and param == 'shortwave_radiation':
            y_pred = np.clip(y_pred * clear_sky_te[_rep_sl], 0, None)
            y_eval = y_te_orig.iloc[_rep_sl]
            print(f"    Solar CSI back-transform applied")

        # --- Dew deficit back-transform: convert deficit to dew point for evaluation ---
        if dew_deficit_mode and param == 'dew_point_2m':
            y_pred = np.clip(y_pred, 0, None)  # deficit is always ≥ 0
            # Couple to the untouched-report corrected temperature, matching
            # live inference without leaking its later production retrain.
            temp_report = report_predictions.get('temperature_2m')
            if temp_report is not None:
                t_proxy = temp_report.reindex(df_te_rep.index).values
            elif 'temperature_2m_ens_mean' in df_v.columns:
                t_ens_col = 'temperature_2m_ens_mean'
                t_proxy = pd.to_numeric(df_te_rep[t_ens_col], errors='coerce').values
            else:
                t_proxy = pd.to_numeric(df_te_rep['temperature_2m_obs'], errors='coerce').values
            y_pred = t_proxy - y_pred  # Td = T - deficit
            y_eval = y_te_orig_dew.iloc[_rep_sl]
            print(f"    Dew deficit back-transform applied")

        mae = mean_absolute_error(y_eval, y_pred)
        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
        report_predictions[param] = pd.Series(
            np.asarray(y_pred, dtype=float), index=df_te_rep.index
        )

        best_mae, best_m = float('inf'), ""
        for m in MODELS:
            mc = f"{m}_{param}_model"
            if mc in df_v.columns:
                mv = pd.to_numeric(df_te_rep[mc], errors='coerce')
                vv = mv.notna() & y_eval.notna()
                if vv.sum() > 50:
                    mm = mean_absolute_error(y_eval[vv], mv[vv])
                    if mm < best_mae:
                        best_mae, best_m = mm, m

        ens_mae = float('inf')
        if raw_ens_col in df_v.columns:
            ev = pd.to_numeric(df_te_rep[raw_ens_col], errors='coerce')
            vv = ev.notna() & y_eval.notna()
            if vv.sum() > 50:
                ens_mae = mean_absolute_error(y_eval[vv], ev[vv])

        impr = (best_mae - mae) / best_mae * 100 if best_mae < float('inf') else 0

        print(f"    {rb_result['info_str']}")
        print(f"  {info['display']:20s} MAE: {mae:.3f}{info['unit']:5s} "
              f"| best: {best_mae:.3f} ({best_m[:8]:8s}) "
              f"| ens: {ens_mae:.3f} "
              f"| +{impr:.1f}%")

        # --- Production retrain: retrain on ALL data (train+test) ---
        print(f"    Retraining {info['display']} na SVIM podacima za produkciju...")
        X_all = pd.concat([X_tr, X_te], axis=0)
        y_all = pd.concat([y_tr, y_te], axis=0)
        sel_feats = rb_result.get('selected_features')
        X_all_sel = X_all[sel_feats] if sel_feats else X_all

        # Temporal weights for full dataset (df_tr_frame: lead-stack aware)
        all_datetimes = pd.concat([df_tr_frame['datetime'], df_v.loc[te, 'datetime']])
        sw_all = _compute_sample_weights(
            y_all, all_datetimes,
            decay_half_life_days=rb_result.get('decay_half_life', 365),
        )
        if 'lead_time' in X_all.columns:
            _lt_all = pd.to_numeric(X_all['lead_time'], errors='coerce').fillna(12).values
            sw_all = sw_all * np.where(_lt_all > 24, 0.5, 1.0)

        tuned_hp = rb_result['tuned_hp']

        # Retrain direct model (MAE loss) on all data
        hp_prod = {k: v for k, v in tuned_hp.items() if k != 'early_stopping_rounds'}
        hp_prod['n_estimators'] = rb_result['direct_n_estimators']
        direct_prod = _new_xgb_regressor(**hp_prod)
        direct_prod.fit(X_all_sel, y_all, verbose=False, sample_weight=sw_all)

        # Retrain residual model (Huber loss) on all data
        ens_all = pd.to_numeric(
            pd.concat([df_tr_frame[ens_col], df_v.loc[te, ens_col]]),
            errors='coerce').fillna(0) if ens_col in df_v.columns else pd.Series(0, index=y_all.index)
        y_resid_all = y_all - ens_all.values
        hp_resid_prod = hp_prod.copy()
        hp_resid_prod['objective'] = 'reg:pseudohubererror'
        hp_resid_prod.pop('monotone_constraints', None)
        hp_resid_prod['n_estimators'] = rb_result['resid_n_estimators']
        resid_prod = _new_xgb_regressor(**hp_resid_prod)
        resid_prod.fit(X_all_sel, y_resid_all, verbose=False, sample_weight=sw_all)

        # Retrain MSE model on all data
        hp_mse_prod = hp_prod.copy()
        hp_mse_prod['objective'] = 'reg:squarederror'
        hp_mse_prod['n_estimators'] = rb_result['mse_n_estimators']
        mse_prod = _new_xgb_regressor(**hp_mse_prod)
        mse_prod.fit(X_all_sel, y_all, verbose=False, sample_weight=sw_all)

        # Retrain CatBoost on all data
        cb_prod = None
        if (_deploy_ridge_meta and rb_result.get('has_catboost')
                and rb_result.get('cb_model') is not None):
            cb_prod = _new_catboost_regressor(
                iterations=rb_result['cb_model'].get_params().get('iterations', 2000),
                depth=rb_result['cb_model'].get_params().get('depth', 6),
                learning_rate=rb_result['cb_model'].get_params().get('learning_rate', 0.03),
                l2_leaf_reg=rb_result['cb_model'].get_params().get('l2_leaf_reg', 1.0),
                subsample=rb_result['cb_model'].get_params().get('subsample', 0.8),
                bootstrap_type='Bernoulli',
                loss_function='MAE', random_seed=42, verbose=0,
            )
            cb_prod.fit(cb.Pool(X_all_sel, y_all, weight=sw_all))

        # Retrain LightGBM on all data
        lgb_prod = None
        if (_deploy_ridge_meta and rb_result.get('has_lightgbm')
                and rb_result.get('lgb_model') is not None):
            lgb_params = rb_result['lgb_model'].get_params()
            lgb_params.pop('callbacks', None)
            lgb_params.pop('early_stopping_round', None)
            lgb_params.pop('early_stopping_rounds', None)
            lgb_prod = _new_lgbm_regressor(**lgb_params)
            lgb_prod.fit(X_all_sel, y_all, sample_weight=sw_all)

        print(f"    Production retrain: {len(X_all)} redova (train={len(X_tr)}, test={len(X_te)})")

        # Save retrained production models
        direct_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}.json"))
        resid_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_resid.json"))
        mse_prod.save_model(os.path.join(MODEL_DIR, f"xgb_{param}_mse.json"))
        # Persist the non-XGB base learners + meta-learner so --skip-training
        # reload reproduces the SAME prediction path (stacked needs resid_model;
        # ridge_meta needs cb/lgb/ridge). Without this, reload silently degrades
        # to direct-only / stacked-without-residual. joblib matches the
        # iso_calibrator persistence pattern below.
        import joblib
        if cb_prod is not None:
            joblib.dump(cb_prod, os.path.join(MODEL_DIR, f"xgb_{param}_catboost.joblib"))
        if lgb_prod is not None:
            joblib.dump(lgb_prod, os.path.join(MODEL_DIR, f"xgb_{param}_lightgbm.joblib"))
        if _deploy_ridge_meta and rb_result.get('ridge_meta') is not None:
            joblib.dump(rb_result['ridge_meta'], os.path.join(MODEL_DIR, f"xgb_{param}_ridge_meta.joblib"))
        else:
            _remove_if_exists(
                os.path.join(MODEL_DIR, f"xgb_{param}_catboost.joblib"),
                os.path.join(MODEL_DIR, f"xgb_{param}_lightgbm.joblib"),
                os.path.join(MODEL_DIR, f"xgb_{param}_ridge_meta.joblib"),
            )

        # Update result with production models
        rb_result['direct_model'] = direct_prod
        rb_result['resid_model'] = resid_prod
        rb_result['mse_model'] = mse_prod
        rb_result['cb_model'] = cb_prod
        rb_result['lgb_model'] = lgb_prod
        if rb_result['is_residual']:
            rb_result['model'] = resid_prod
        else:
            rb_result['model'] = direct_prod

        # Use selected features if available, otherwise all valid features
        effective_features = rb_result.get('selected_features') or vf

        trained[param] = {
            'model': rb_result['model'],
            'direct_model': rb_result['direct_model'],
            'resid_model': rb_result.get('resid_model'),
            'mse_model': rb_result.get('mse_model'),
            'cb_model': rb_result.get('cb_model'),
            'lgb_model': rb_result.get('lgb_model'),
            'ridge_meta': rb_result.get('ridge_meta') if _deploy_ridge_meta else None,
            'ridge_meta_regime': rb_result.get('ridge_meta_regime') or [],
            'has_catboost': bool(_deploy_ridge_meta and rb_result.get('has_catboost', False)),
            'has_lightgbm': bool(_deploy_ridge_meta and rb_result.get('has_lightgbm', False)),
            'method': method_str,
            'is_residual': rb_result['is_residual'],
            'blend_alpha': rb_result.get('blend_alpha'),
            'stack_weights': rb_result.get('stack_weights'),
            'model_ens_col': model_ens_col,
            'features': effective_features,
            'csi_mode': csi_mode,
            'dew_deficit_mode': dew_deficit_mode,
            'cloud_target_version': 2 if param == 'cloud_cover' else None,
            'mae': mae, 'rmse': rmse,
            'best_model': best_m, 'best_model_mae': best_mae,
            'ensemble_mae': ens_mae, 'improvement': impr,
        }
        results[param] = {
            'mae': round(mae, 3), 'rmse': round(rmse, 3),
            'unit': info['unit'], 'display': info['display'],
            'improvement': round(impr, 1),
            'best_model': best_m, 'best_model_mae': round(best_mae, 3),
            'ensemble_mae': round(ens_mae, 3),
            'method': method_str,
            'is_residual': bool(rb_result['is_residual']),
            'ridge_meta_regime': rb_result.get('ridge_meta_regime') or [],
            'blend_alpha': float(rb_result['blend_alpha']) if rb_result.get('blend_alpha') is not None else None,
            'stack_weights': [float(w) for w in rb_result['stack_weights']] if rb_result.get('stack_weights') else None,
            'model_ens_col': model_ens_col,
            'decay_half_life': int(rb_result.get('decay_half_life', 365)),
            # Persist back-transform flags + base-learner availability so
            # --skip-training reload reproduces the exact training-time path.
            'csi_mode': bool(csi_mode),
            'dew_deficit_mode': bool(dew_deficit_mode),
            'cloud_target_version': 2 if param == 'cloud_cover' else None,
            'has_catboost': bool(_deploy_ridge_meta and rb_result.get('has_catboost', False)),
            'has_lightgbm': bool(_deploy_ridge_meta and rb_result.get('has_lightgbm', False)),
            'is_precip': False,
        }

        # predictive distribution (T2m / wind / gusts; raw-target only,
        # so CSI- and deficit-transformed params are skipped by the tuple).
        if param in QUANTILE_TARGETS:
            qb = _train_target_quantiles(
                param, X_tr, y_tr, X_te_rep, y_te_rep,
                train_datetimes=df_tr_frame['datetime'],
            )
            if qb is not None:
                trained[param]['quantiles'] = qb
                results[param]['quantile_crps'] = qb['crps']
                results[param]['quantile_cov90'] = qb['cov90']

        _persist_artifacts()  # incremental save (crash-safe)

    _persist_artifacts()  # final write (also covers the no-target-trained case)

    # Promotion gate + experiment registry
    gate_verdicts = _champion_gate(results, results_prev)
    _append_run_history(results, df, gate_verdicts)

    return trained, results


def load_trained_models():
    """Load pre-trained XGBoost models + metadata from disk. No historical data needed."""
    print("\n[3/6] Ucitavanje SACUVANIH modela (--skip-training)...")
    results_path = os.path.join(MODEL_DIR, 'training_results.json')
    features_path = os.path.join(MODEL_DIR, 'feature_lists.json')
    bias_path = os.path.join(MODEL_DIR, 'bias_tables.json')

    if not os.path.exists(results_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Nema sacuvanih modela u {MODEL_DIR}. Pokrenite prvo bez --skip-training.")

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(features_path, 'r') as f:
        feature_lists = json.load(f)

    bias_tables = {}
    if os.path.exists(bias_path):
        with open(bias_path, 'r') as f:
            bt_raw = json.load(f)
        for k, v in bt_raw.items():
            bias_tables[k] = pd.DataFrame(v)

    trained = {}
    for param, rinfo in results.items():
        features = feature_lists.get(param, [])
        if not features:
            continue

        if rinfo.get('is_precip', False):
            cls_path = os.path.join(MODEL_DIR, f"xgb_{param}_cls.json")
            reg_path = os.path.join(MODEL_DIR, f"xgb_{param}_reg.json")
            single_path = os.path.join(MODEL_DIR, f"xgb_{param}.json")
            tweedie_path = os.path.join(MODEL_DIR, f"xgb_{param}_tweedie.json")
            if not all(os.path.exists(p) for p in [cls_path, reg_path, single_path]):
                print(f"  {rinfo['display']:20s} --- SKIP (fajlovi ne postoje)")
                continue
            if rinfo.get('method') == 'tweedie' and not os.path.exists(tweedie_path):
                raise FileNotFoundError(
                    f"{rinfo['display']}: izabran je Tweedie, ali nedostaje {tweedie_path}"
                )

            # classifier was trained with focal loss (custom objective)
            # using xgb.train, so we reload it as a Booster and wrap it in a thin
            # adapter that exposes predict_proba (sigmoid of raw margin).
            _cls_booster = _new_xgb_booster()
            _cls_booster.load_model(cls_path)
            _restore_xgb_device(_cls_booster)
            # Read focal hyperparams from sidecar (best effort; safe defaults if absent)
            _focal_g, _focal_a, _focal_n = 2.0, 0.25, 0
            _focal_sidecar = os.path.join(MODEL_DIR, f"xgb_{param}_cls_focal.json")
            if os.path.exists(_focal_sidecar):
                try:
                    with open(_focal_sidecar, encoding='utf-8') as _fs:
                        _fsj = json.load(_fs)
                        _focal_g = float(_fsj.get('focal_gamma', _focal_g))
                        _focal_a = float(_fsj.get('focal_alpha', _focal_a))
                        _focal_n = int(_fsj.get('n_estimators', _focal_n))
                except Exception:
                    pass

            class _BoosterProbaAdapterReload:
                def __init__(self, booster, fg, fa, n_rounds=0):
                    self._b = booster; self._gamma = float(fg); self._alpha = float(fa)
                    self._n_rounds = int(n_rounds)
                def _to_dmatrix(self, X):
                    if isinstance(X, xgb.DMatrix): return X
                    cols = list(X.columns) if hasattr(X, 'columns') else None
                    return xgb.DMatrix(X, feature_names=cols, missing=np.nan)
                def predict_proba(self, X):
                    dmatrix = self._to_dmatrix(X)
                    margins = (
                        self._b.predict(dmatrix, iteration_range=(0, self._n_rounds))
                        if self._n_rounds > 0 else self._b.predict(dmatrix)
                    )
                    p = 1.0 / (1.0 + np.exp(-margins))
                    return np.stack([1 - p, p], axis=1)
                def predict(self, X):
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
                def save_model(self, path):
                    self._b.save_model(path)
            cls_model = _BoosterProbaAdapterReload(
                _cls_booster, _focal_g, _focal_a, _focal_n
            )
            reg_model = _new_xgb_regressor()
            reg_model.load_model(reg_path)
            _restore_xgb_device(reg_model)
            single_model = _new_xgb_regressor()
            single_model.load_model(single_path)
            _restore_xgb_device(single_model)

            tweedie_model = None
            if os.path.exists(tweedie_path):
                tweedie_model = _new_xgb_regressor()
                tweedie_model.load_model(tweedie_path)
                _restore_xgb_device(tweedie_model)

            precip_info = {
                    'cls_model': cls_model,
                    'reg_model': reg_model,
                    'single_model': single_model,
                    'best_method': rinfo['method'],
                    'threshold': rinfo.get('threshold', 0.35),
                    'use_sqrt': rinfo.get('use_sqrt', False),
                    'blend_alpha': rinfo.get('blend_alpha', 1.0),
                    'trusted_amount_alpha': rinfo.get(
                        'trusted_amount_alpha', 1.0
                    ),
            }
            if tweedie_model is not None:
                precip_info['tweedie_model'] = tweedie_model

            # load isotonic calibrator if persisted
            iso_path = os.path.join(MODEL_DIR, f"xgb_{param}_iso_calibrator.joblib")
            if os.path.exists(iso_path):
                import joblib
                precip_info['iso_calibrator'] = joblib.load(iso_path)

            # reload PoP blend gate if persisted
            pb_jl = os.path.join(MODEL_DIR, f"xgb_{param}_pop_blend.joblib")
            pb_js = os.path.join(MODEL_DIR, f"xgb_{param}_pop_blend.json")
            if os.path.exists(pb_jl) and os.path.exists(pb_js):
                try:
                    import joblib
                    with open(pb_js, encoding='utf-8') as _pbf:
                        _pbm = json.load(_pbf)
                    precip_info['pop_blend_lr'] = joblib.load(pb_jl)
                    precip_info['pop_blend_cols'] = _pbm.get('cols')
                    precip_info['pop_blend_tau'] = float(_pbm.get('tau', 0.5))
                    precip_info['rain_gate_mode'] = rinfo.get(
                        'rain_gate_mode', _pbm.get('mode', 'trusted'))
                    print(f"    PoP-blend gate: mode={precip_info['rain_gate_mode']}")
                except Exception as _e:
                    print(f"    WARN: pop_blend reload failed ({_e})")

            trained[param] = {
                'precip_info': precip_info,
                'features': features,
                'mae': rinfo['mae'], 'rmse': rinfo['rmse'],
                'best_model': rinfo.get('best_model', ''),
                'best_model_mae': rinfo.get('best_model_mae', 0),
                'ensemble_mae': rinfo.get('ensemble_mae', 0),
                'improvement': rinfo.get('improvement', 0),
            }
            print(f"  {rinfo['display']:20s} loaded (MAE={rinfo['mae']}) [{rinfo['method']}]")
        else:
            direct_path = os.path.join(MODEL_DIR, f"xgb_{param}.json")
            resid_path = os.path.join(MODEL_DIR, f"xgb_{param}_resid.json")
            mse_path = os.path.join(MODEL_DIR, f"xgb_{param}_mse.json")
            if not os.path.exists(direct_path):
                print(f"  {rinfo['display']:20s} --- SKIP (fajl ne postoji)")
                continue

            direct_model = _new_xgb_regressor()
            direct_model.load_model(direct_path)
            _restore_xgb_device(direct_model)

            is_residual = rinfo.get('is_residual', False)
            # Load resid_model whenever it exists — NOT only when is_residual.
            # The 'stacked' and 'ridge_meta' methods have is_residual=False yet
            # still consume resid_model; gating the load on is_residual silently
            # collapsed their residual term to direct in --skip-training mode.
            resid_model = None
            if os.path.exists(resid_path):
                resid_model = _new_xgb_regressor()
                resid_model.load_model(resid_path)
                _restore_xgb_device(resid_model)

            mse_model = None
            if os.path.exists(mse_path):
                mse_model = _new_xgb_regressor()
                mse_model.load_model(mse_path)
                _restore_xgb_device(mse_model)

            # CatBoost / LightGBM base learners + RidgeCV meta-learner (joblib).
            # Needed so method=='ridge_meta' reproduces the trained prediction
            # instead of falling back to direct-only.
            import joblib
            cb_model = lgb_model = ridge_meta = None
            cb_jl = os.path.join(MODEL_DIR, f"xgb_{param}_catboost.joblib")
            lgb_jl = os.path.join(MODEL_DIR, f"xgb_{param}_lightgbm.joblib")
            ridge_jl = os.path.join(MODEL_DIR, f"xgb_{param}_ridge_meta.joblib")
            if os.path.exists(cb_jl):
                try: cb_model = joblib.load(cb_jl)
                except Exception as _e: print(f"    WARN: cb reload failed ({_e})")
            if os.path.exists(lgb_jl):
                try: lgb_model = joblib.load(lgb_jl)
                except Exception as _e: print(f"    WARN: lgb reload failed ({_e})")
            if os.path.exists(ridge_jl):
                try: ridge_meta = joblib.load(ridge_jl)
                except Exception as _e: print(f"    WARN: ridge_meta reload failed ({_e})")

            # has_* must reflect what actually reloaded (a stored True is useless
            # if the joblib is missing) — apply_correction guards on the model
            # object, but keep the flag honest for downstream/debugging.
            has_catboost = bool(rinfo.get('has_catboost', False)) and cb_model is not None
            has_lightgbm = bool(rinfo.get('has_lightgbm', False)) and lgb_model is not None

            if is_residual and resid_model is not None:
                active_model = resid_model
            else:
                active_model = direct_model

            trained[param] = {
                'model': active_model,
                'direct_model': direct_model,
                'resid_model': resid_model,
                'mse_model': mse_model,
                'cb_model': cb_model,
                'lgb_model': lgb_model,
                'ridge_meta': ridge_meta,
                'has_catboost': has_catboost,
                'has_lightgbm': has_lightgbm,
                'method': rinfo.get('method', 'direct'),
                'is_residual': is_residual,
                'ridge_meta_regime': rinfo.get('ridge_meta_regime') or [],
                'blend_alpha': rinfo.get('blend_alpha'),
                'stack_weights': rinfo.get('stack_weights'),
                'model_ens_col': rinfo.get('model_ens_col', f'{param}_ens_mean'),
                'csi_mode': bool(rinfo.get('csi_mode', False)),
                'dew_deficit_mode': bool(rinfo.get('dew_deficit_mode', False)),
                'cloud_target_version': int(
                    rinfo.get('cloud_target_version', 1)
                ) if param == 'cloud_cover' else None,
                'features': features,
                'mae': rinfo['mae'], 'rmse': rinfo['rmse'],
                'best_model': rinfo.get('best_model', ''),
                'best_model_mae': rinfo.get('best_model_mae', 0),
                'ensemble_mae': rinfo.get('ensemble_mae', 0),
                'improvement': rinfo.get('improvement', 0),
            }
            print(f"  {rinfo['display']:20s} loaded (MAE={rinfo['mae']}) [{rinfo.get('method', 'direct')}]")

    # reload quantile+CQR bundles where present
    for param in QUANTILE_TARGETS:
        if param not in trained:
            continue
        if 'quantile_crps' not in results.get(param, {}):
            continue
        prefix = _active_quantile_prefix(param)
        if prefix is None:
            print(f"  Kvantili {param}: nema aktivnog, validnog bundle-a")
            continue
        try:
            q_models, q_offsets = pf.load_quantile_bundle(prefix)
            feats_path = f"{prefix}_features.json"
            if q_models is not None and os.path.exists(feats_path):
                with open(feats_path, encoding='utf-8') as f:
                    q_feats = json.load(f)
                trained[param]['quantiles'] = {
                    'models': q_models, 'offsets': q_offsets, 'features': q_feats,
                }
                print(f"  Kvantili {param}: bundle ucitan ({len(q_models)} alfa)")
        except Exception as _e:
            print(f"  Kvantili {param}: reload preskočen ({_e})")

    print(f"  Ucitano {len(trained)}/{len(results)} modela.")
    return trained, results, bias_tables


def _fetch_aux_live(fc_all):
    """Live neighborhood 5x5 precip + Podgorica MSLP.
    Best-effort: any failure leaves the columns absent, and engineer_features
    simply skips the derived stats (NaN passthrough keeps models usable)."""
    URL = "https://api.open-meteo.com/v1/forecast"

    for model_name in NEIGHBORHOOD_MODELS:
        model_id = MODEL_IDS[model_name]
        update_h = MODEL_UPDATE_HOURS.get(model_id, DEFAULT_UPDATE_HOURS)
        cpath = _cache_path('nbr', model_id, LAT, LON)
        payload = _load_fresh_cache(cpath, update_h)
        if payload is None:
            params = {
                "latitude": ",".join(f"{v:.4f}" for v in NBR_GRID_LATS),
                "longitude": ",".join(f"{v:.4f}" for v in NBR_GRID_LONS),
                "hourly": "precipitation",
                "timezone": FORECAST_TIMEZONE, "models": model_id,
                "forecast_days": 10, "precipitation_unit": "mm",
            }
            try:
                r = requests.get(URL, params=params, timeout=45)
                r.raise_for_status()
                payload = r.json()
                _save_cache(cpath, payload)
            except Exception as e:
                payload, _age = _load_stale_cache(cpath)
                if payload is None:
                    print(f"  NBR {model_name}: FAIL ({e}), preskačem")
                    continue
        locs = payload if isinstance(payload, list) else [payload]
        wide = None
        for idx, loc in enumerate(locs):
            h = loc.get('hourly', {})
            if 'time' not in h or 'precipitation' not in h:
                continue
            d = pd.DataFrame({'datetime': pd.to_datetime(h['time']),
                              f'{model_name}_nbr_p{idx:02d}': h['precipitation']})
            wide = d if wide is None else wide.merge(d, on='datetime', how='outer')
        if wide is not None and len(wide.columns) > 20:
            keep = ['datetime'] + [c for c in wide.columns
                                   if c != 'datetime' and c not in fc_all.columns]
            fc_all = fc_all.merge(wide[keep], on='datetime', how='left')
            print(f"  NBR {model_name}: OK ({len(wide.columns) - 1} tacaka)")
        time.sleep(0.5)

    for model_name in MODELS:
        model_id = MODEL_IDS[model_name]
        update_h = MODEL_UPDATE_HOURS.get(model_id, DEFAULT_UPDATE_HOURS)
        cpath = _cache_path('pg', model_id, PG_LAT, PG_LON)
        payload = _load_fresh_cache(cpath, update_h)
        if payload is None:
            params = {"latitude": PG_LAT, "longitude": PG_LON,
                      "hourly": "pressure_msl", "timezone": FORECAST_TIMEZONE,
                      "models": model_id, "forecast_days": 10}
            try:
                r = requests.get(URL, params=params, timeout=30)
                r.raise_for_status()
                payload = r.json()
                _save_cache(cpath, payload)
            except Exception as e:
                payload, _age = _load_stale_cache(cpath)
                if payload is None:
                    print(f"  PG {model_name}: FAIL ({e})")
                    continue
        h = payload.get('hourly', {}) if isinstance(payload, dict) else {}
        if 'time' in h and 'pressure_msl' in h:
            col = f'{model_name}_pressure_msl_pg_model'
            if col not in fc_all.columns:
                d = pd.DataFrame({'datetime': pd.to_datetime(h['time']),
                                  col: h['pressure_msl']})
                fc_all = fc_all.merge(d, on='datetime', how='left')
        time.sleep(0.4)
    return fc_all


def fetch_live_forecasts():
    print("\n[4/6] Preuzimanje LIVE prognoza...")
    URL = "https://api.open-meteo.com/v1/forecast"
    all_fc = {}

    # Trusted models go FIRST -- before any rate-limit pressure builds up
    # from other model calls -- and get more retries, longer timeout, and
    # a precipitation-presence check (Open-Meteo sometimes returns 200 OK
    # with empty hourly or no precip key, which would silently break the
    # trusted gate downstream).
    ordered = TRUSTED_MODELS + [m for m in MODELS if m not in TRUSTED_MODELS]

    for model_name in ordered:
        model_id = MODEL_IDS[model_name]
        is_trusted = model_name in TRUSTED_MODELS
        max_attempts = 5 if is_trusted else 3
        timeout_s = 60 if is_trusted else 30
        tag = " [TRUSTED]" if is_trusted else ""
        update_h = MODEL_UPDATE_HOURS.get(model_id, DEFAULT_UPDATE_HOURS)
        cache_path = _cache_path('forecast', model_id, LAT, LON)
        print(f"  {model_name}{tag}...", end=" ")

        # Preferred: ask upstream meta when the model was last re-run, only
        # refetch if our cached copy is older than the latest run.
        upstream_time = _get_upstream_update_time(model_id)
        use_cache = False
        if upstream_time is not None and _cache_current_for_upstream(cache_path, upstream_time):
            use_cache = True
            cache_reason = f"upstream run @ {time.strftime('%H:%M UTC', time.gmtime(upstream_time))}"
        elif upstream_time is None:
            # Fallback: TTL check (model has no meta endpoint).
            cached_ttl = _load_fresh_cache(cache_path, update_h)
            if cached_ttl is not None:
                use_cache = True
                cache_reason = f"TTL {update_h}h fallback (no meta)"

        if use_cache:
            cached = _load_stale_cache(cache_path)[0]  # load regardless of age now
            if cached is not None:
                h = cached.get('hourly', {})
                trusted_base_ok = h and 'precipitation' in h and h.get('precipitation')
                trusted_diag_ok = all(
                    name in h and h.get(name) for name in TRUSTED_CONVECTIVE_VARS
                )
                if ((not is_trusted)
                        or (trusted_base_ok and trusted_diag_ok)):
                    d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
                    for v in HOURLY_VARS:
                        if v in h:
                            d[f"{model_name}_{v}_model"] = h[v]
                    all_fc[model_name] = d
                    age_h = _cache_age_hours(cache_path)
                    print(f"CACHE ({len(d)}h, {(age_h or 0):.1f}h, {cache_reason})")
                    continue
                if is_trusted and trusted_base_ok and not trusted_diag_ok:
                    print("CACHE bez convective dijagnostike; refetch", end=" ")

        params = {
            "latitude": LAT, "longitude": LON,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": FORECAST_TIMEZONE, "temperature_unit": "celsius",
            "wind_speed_unit": "ms", "precipitation_unit": "mm",
            "models": model_id, "forecast_days": 10,
        }
        fetched = False
        for attempt in range(max_attempts):
            try:
                r = requests.get(URL, params=params, timeout=timeout_s)
                if r.status_code == 429:
                    backoff = 60 * (1.5 ** attempt)
                    print(f"429 (sleep {backoff:.0f}s)", end=" ")
                    time.sleep(backoff); continue
                r.raise_for_status()
                resp = r.json()
                h = resp.get('hourly', {})
                if is_trusted and (not h or 'precipitation' not in h
                                   or not h.get('precipitation')):
                    if attempt < max_attempts - 1:
                        print("empty/no-precip; retrying", end=" ")
                        time.sleep(5 * (attempt + 1)); continue
                    # Route through the normal exception path so the final
                    # attempt can use a valid stale cache instead of bypassing
                    # that fallback on a malformed HTTP 200 response.
                    raise ValueError(
                        f"trusted response missing precipitation after {max_attempts} attempts"
                    )
                d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
                for v in HOURLY_VARS:
                    if v in h:
                        d[f"{model_name}_{v}_model"] = h[v]
                all_fc[model_name] = d
                _save_cache(cache_path, resp)
                _save_upstream_meta(cache_path, upstream_time)
                print(f"OK ({len(d)}h)")
                fetched = True
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Fallback: any stale cache is better than nothing.
                    stale, age_h = _load_stale_cache(cache_path)
                    if stale is not None:
                        h = stale.get('hourly', {})
                        d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
                        for v in HOURLY_VARS:
                            if v in h:
                                d[f"{model_name}_{v}_model"] = h[v]
                        all_fc[model_name] = d
                        print(f"FAIL ({e}); STALE CACHE used ({len(d)}h, {(age_h or 0):.1f}h staro)")
                    else:
                        print(f"FAIL: {e}")
                else:
                    time.sleep(5 * (attempt + 1))
        time.sleep(1.5)

    # Hard fail early if the trusted model never materialised. Better to
    # bail here -- before training / correction burn cycles -- so the
    # outer fallback (previous JSON) kicks in immediately.
    for tm in TRUSTED_MODELS:
        if tm not in all_fc or f'{tm}_precipitation_model' not in all_fc[tm].columns:
            raise TrustedRainGateError(
                f"Trusted model '{tm}' nije fetched ni nakon prioritetnog "
                f"pokušaja sa povećanim retry/timeout-om. Najvjerovatnije je "
                f"Open-Meteo API trenutno bez tog modela za našu tačku."
            )

    if not all_fc:
        raise RuntimeError("Nema prognoza!")

    merged = list(all_fc.values())[0]
    for k in list(all_fc.keys())[1:]:
        merged = merged.merge(all_fc[k], on='datetime', how='outer')
    merged.sort_values('datetime', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Keep today's already-elapsed hours so the daily summary for "today"
    # can include morning rainfall instead of computing icon/total from a
    # partial-afternoon slice. Hourly JSON output filters past hours back
    # out at write time.
    now = local_now().floor('h')
    today_start = now.normalize()
    mask = merged['datetime'] >= today_start
    fc_all = merged[mask].copy().reset_index(drop=True)
    n_past = int((fc_all['datetime'] < now).sum())
    print(f"  Prognoza: {fc_all.shape[0]} sati ({fc_all['datetime'].min()} --- "
          f"{fc_all['datetime'].max()}) [+{n_past} prethodnih sati danasnjeg dana za daily summary]")

    print("\n  Preuzimanje Previous Runs (Day1/Day2)...")
    prev_hourly_list = []
    for v in PREV_RUNS_VARS:
        prev_hourly_list.append(v)
        prev_hourly_list.append(f"{v}_previous_day1")
        prev_hourly_list.append(f"{v}_previous_day2")
    prev_hourly_str = ",".join(prev_hourly_list)

    for model_name in PREV_RUNS_MODELS:
        if model_name not in all_fc:
            continue
        model_id = MODEL_IDS[model_name]
        update_h = MODEL_UPDATE_HOURS.get(model_id, DEFAULT_UPDATE_HOURS)
        pr_cache_path = _cache_path('prev_runs', model_id, LAT, LON)

        def _parse_prev(h, label):
            d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
            added = 0
            for v in PREV_RUNS_VARS:
                for lag in ['previous_day1', 'previous_day2']:
                    col = f"{v}_{lag}"
                    new_col = f"{model_name}_{v}_{lag}"
                    if col in h:
                        d[new_col] = h[col]
                        added += 1
            return d, added

        upstream_time = _get_upstream_update_time(model_id)
        use_cache = False
        cache_reason = ""
        if upstream_time is not None and _cache_current_for_upstream(pr_cache_path, upstream_time):
            use_cache = True
            cache_reason = f"upstream @ {time.strftime('%H:%M UTC', time.gmtime(upstream_time))}"
        elif upstream_time is None:
            cached_ttl = _load_fresh_cache(pr_cache_path, update_h)
            if cached_ttl is not None:
                use_cache = True
                cache_reason = f"TTL {update_h}h"

        if use_cache:
            stale_data, _ = _load_stale_cache(pr_cache_path)
            if stale_data is not None:
                d, added = _parse_prev(stale_data.get('hourly', {}), 'CACHE')
                fc_all = fc_all.merge(d[['datetime'] + [c for c in d.columns if c != 'datetime']],
                                       on='datetime', how='left')
                age_h = _cache_age_hours(pr_cache_path)
                print(f"    {model_name}: CACHE ({added} cols, {(age_h or 0):.1f}h, {cache_reason})")
                time.sleep(0.3)
                continue

        pr_params = {
            "latitude": LAT, "longitude": LON,
            "hourly": prev_hourly_str,
            "timezone": FORECAST_TIMEZONE, "models": model_id, "forecast_days": 10,
        }
        try:
            r = requests.get(PREV_RUNS_API, params=pr_params, timeout=30)
            if r.status_code == 429:
                time.sleep(60)
                r = requests.get(PREV_RUNS_API, params=pr_params, timeout=30)
            r.raise_for_status()
            resp = r.json()
            d, added = _parse_prev(resp.get('hourly', {}), 'OK')
            fc_all = fc_all.merge(d[['datetime'] + [c for c in d.columns if c != 'datetime']],
                                   on='datetime', how='left')
            _save_cache(pr_cache_path, resp)
            _save_upstream_meta(pr_cache_path, upstream_time)
            print(f"    {model_name}: OK ({added} columns)")
        except Exception as e:
            stale, age_h = _load_stale_cache(pr_cache_path)
            if stale is not None:
                d, added = _parse_prev(stale.get('hourly', {}), 'STALE')
                fc_all = fc_all.merge(d[['datetime'] + [c for c in d.columns if c != 'datetime']],
                                       on='datetime', how='left')
                print(f"    {model_name}: FAIL ({e}); STALE CACHE used ({added} cols, {(age_h or 0):.1f}h staro)")
            else:
                print(f"    {model_name}: FAIL ({e})")
        time.sleep(1.5)

    # Fetch SST for forecast period
    sst_df = fetch_sst_data(
        fc_all['datetime'].min() - pd.Timedelta(days=30),
        fc_all['datetime'].max()
    )
    if sst_df is not None:
        fc_all = fc_all.merge(sst_df, on='datetime', how='left')
        # Forward-fill SST since marine data may lag
        if 'sst' in fc_all.columns:
            fc_all['sst'] = fc_all['sst'].ffill()
            print(f"  SST: merged ({fc_all['sst'].notna().sum()} valid rows)")

    # Monograph-2 aux series: neighborhood precip + Podgorica MSLP (best-effort)
    print("\n  Preuzimanje aux podataka (neighborhood + Podgorica)...")
    try:
        fc_all = _fetch_aux_live(fc_all)
    except Exception as e:
        print(f"  AUX fetch preskočen ({e})")

    return fc_all


# Marine forecast: ensemble of 2 wave models + offshore wind, no bias
# correction. Its own pipeline, so the atmospheric path stays clean.


def _circular_mean_degrees(values, axis=1):
    """Mean compass direction without the 0/360 wrap-around artifact."""
    frame = values.apply(pd.to_numeric, errors='coerce') if hasattr(values, 'apply') else np.asarray(values, dtype=float)
    radians = np.radians(frame)
    sin_mean = np.nanmean(np.sin(radians), axis=axis)
    cos_mean = np.nanmean(np.cos(radians), axis=axis)
    direction = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    resultant = np.hypot(sin_mean, cos_mean)
    direction = np.where(resultant > 1e-12, direction, np.nan)
    if isinstance(frame, pd.DataFrame):
        direction = pd.Series(direction, index=frame.index, dtype=float)
        direction[~frame.notna().any(axis=axis)] = np.nan
    return direction

# Beaufort thresholds (m/s, upper bound of each Bft 0..11).
_BEAUFORT_THRESHOLDS = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]

# Douglas / WMO sea state thresholds (m, upper bound of state 0..8).
_DOUGLAS_THRESHOLDS = [0.1, 0.5, 1.25, 2.5, 4.0, 6.0, 9.0, 14.0]

DOUGLAS_LABELS = {
    0: "Bonaca", 1: "Mirno", 2: "Blagi talasi", 3: "Umjereni talasi",
    4: "Talasasto", 5: "Vrlo talasasto", 6: "Uzburkano",
    7: "Vrlo uzburkano", 8: "Olujni talasi", 9: "Izuzetno olujni talasi",
}


def beaufort_from_wind(ms):
    if ms is None or pd.isna(ms):
        return None
    for i, t in enumerate(_BEAUFORT_THRESHOLDS):
        if ms < t:
            return i
    return 12


def douglas_sea_state(m):
    # _DOUGLAS_THRESHOLDS are the UPPER bounds of states 1..8, so the first
    # threshold the height is below puts it in state i+1 (e.g. 1.7 m < 2.5 ->
    # state 4 "Moderate", 1.25-2.5 m). Returning i was off by one (1.7 m -> 3,
    # which the UI table caps at 1.25 m).
    if m is None or pd.isna(m):
        return None
    if m < 0.05:            # glassy calm
        return 0
    for i, t in enumerate(_DOUGLAS_THRESHOLDS):
        if m < t:
            return i + 1
    return 9


# Bečići faces the open sea to the S/SW; only two winds raise real waves here.
# Jugo (E-SE-S, long Adriatic fetch) brings the big, long-period waves; bura
# (NE, short fetch off the mountains) brings short, sharp chop that stays
# small. Wave directions from N or W come off the land, so anything the
# 5.5 km wave model puts there is a grid artifact — the displayed sea state
# is capped per direction sector (upper bound in degrees, waves-FROM):
_SEA_STATE_DIRECTION_CAPS = [
    (22.5, 2),    # N: off the land            -> max "Blagi talasi"
    (90.0, 4),    # NE (bura): short chop      -> max "Talasasto"
    (240.0, 9),   # E-SE-S-SW (jugo fetch)     -> full Douglas scale
    (337.5, 2),   # W-NW: off the land         -> max "Blagi talasi"
    (360.0, 2),   # wraps back into N
]


def sea_state_direction_cap(direction_deg):
    """Highest Douglas state a wave from this direction can deliver at Bečići.
    Mirrored client-side in docs/forecast.html (adriaticSeaState)."""
    if direction_deg is None or pd.isna(direction_deg):
        return 9
    d = float(direction_deg) % 360
    for upper, cap in _SEA_STATE_DIRECTION_CAPS:
        if d < upper:
            return cap
    return 2


def adriatic_sea_state(height_m, direction_deg):
    """Douglas state from height, capped by what the direction can deliver."""
    sea = douglas_sea_state(height_m)
    if sea is None:
        return None
    return min(sea, sea_state_direction_cap(direction_deg))


def cg_wind_name(direction_deg, speed_ms):
    """Adriatic-specific wind name from direction + speed."""
    if direction_deg is None or pd.isna(direction_deg):
        return ""
    d = float(direction_deg) % 360
    s = float(speed_ms) if speed_ms is not None and pd.notna(speed_ms) else 0.0
    if d >= 337.5 or d < 22.5:
        return "Tramontana"
    if 22.5 <= d < 67.5:
        return "Bura" if s >= 7 else "Levanat"
    if 67.5 <= d < 112.5:
        return "Levanat"
    if 112.5 <= d < 157.5:
        return "Jugo"
    if 157.5 <= d < 202.5:
        return "Sirocco"
    if 202.5 <= d < 247.5:
        return "Lebić"
    if 247.5 <= d < 292.5:
        return "Pulenat"
    if 292.5 <= d < 337.5:
        return "Maestral"
    return ""


MS_PER_KNOT = 0.514444

# Jugo (JI) - Sirocco (J) - Lebić (JZ). The sector with open-sea fetch to
# Bečići, and the only one whose waves are allowed to raise a warning badge.
SOUTHERN_SECTOR = (112.5, 247.5)

_SAILING_RANK = ["Idealno", "Dobro", "Prihvatljivo", "Oprez", "Opasno"]
_SAILING_COLOR = {
    "Idealno": "green", "Dobro": "green", "Prihvatljivo": "yellow",
    "Oprez": "orange", "Opasno": "red",
}


# A daily badge must not turn on one gusty hour, least of all a nocturnal
# one: the audience is swimmers and small boats, so only 08-20 counts, and
# the gust level has to hold for GUST_PERSIST_HOURS of them.
GUST_PERSIST_HOURS = 4
GUST_DAY_WINDOW = (8, 20)


def persistent_daytime_gust(day_rows, gust_col='wind_gusts_10m_mean'):
    """Gust level (m/s) sustained for GUST_PERSIST_HOURS daytime hours, i.e.
    the Nth strongest such hour. None when the day can't clear that bar —
    which correctly leaves the gust rungs unfired rather than guessing."""
    if day_rows is None or len(day_rows) == 0 or gust_col not in day_rows:
        return None
    hours = day_rows['datetime'].dt.hour
    lo, hi = GUST_DAY_WINDOW
    daytime = pd.to_numeric(
        day_rows.loc[(hours >= lo) & (hours <= hi), gust_col],
        errors='coerce').dropna()
    if len(daytime) < GUST_PERSIST_HOURS:
        return None
    return float(daytime.nlargest(GUST_PERSIST_HOURS).iloc[-1])


def direction_capped_height(height_m, direction_deg):
    """Wave height clipped to what this bearing can actually deliver at
    Bečići. The sea-state pill has always been capped this way; the score
    must use the same number, or the card ends up printing "Blagi talasi"
    and docking the day a rung for a 1 m wave in the same breath."""
    if height_m is None or pd.isna(height_m):
        return None
    cap = sea_state_direction_cap(direction_deg)
    if cap - 1 < len(_DOUGLAS_THRESHOLDS):
        return min(float(height_m), _DOUGLAS_THRESHOLDS[cap - 1])
    return float(height_m)


def is_southern_direction(direction_deg):
    """True for the jugo-sirocco-lebić sector. Unknown direction counts as
    southern so a missing field can't silently mute a warning."""
    if direction_deg is None or pd.isna(direction_deg):
        return True
    lo, hi = SOUTHERN_SECTOR
    return lo <= (float(direction_deg) % 360) < hi


def sailing_score(bft, wave_height, wave_direction=None,
                  wind_direction=None, wind_gusts_ms=None):
    """Traffic-light + label for sailing comfort at Bečići. Strict, because
    the audience is small boats and swimmers rather than offshore keelboats.

    "Oprez"/"Opasno" are reserved for two situations, judged independently
    and combined worst-case:

    * Southern waves. Only the jugo-sirocco-lebić sector has the fetch to
      build a real sea here; N and W "waves" are 5.5 km-grid artifacts off
      the land, which is why sea_state_direction_cap already flattens them.
      Height alone is enough once the sea is up, since a jugo swell outlives
      the wind that raised it. A southern wind of 3 Bft+ is still building,
      so it drops the bar from 1.0/1.8 m to 0.8/1.5 m.
    * Gusts, in knots, on their own terms: >15 kn caps the day at
      "Prihvatljivo", >22 kn makes it "Oprez", gale-force >34 kn "Opasno".

    Anything else tops out at "Prihvatljivo" no matter how tall the model
    says the waves are. Mirrored client-side in docs/forecast.html
    (sailingScore)."""
    if bft is None or wave_height is None:
        return ("gray", "N/A")

    # Baseline: the calm rungs, keyed off Bft + height. The height is the
    # direction-capped one, so an off-the-land "wave" the card itself calls
    # "Blagi talasi" can no longer cost the day a rung.
    eff_height = direction_capped_height(wave_height, wave_direction)
    if bft <= 2 and eff_height <= 0.3:
        label = "Idealno"
    elif bft <= 3 and eff_height <= 0.6:
        label = "Dobro"
    else:
        label = "Prihvatljivo"

    def _worst(other):
        return other if _SAILING_RANK.index(other) > _SAILING_RANK.index(label) else label

    if is_southern_direction(wave_direction):
        wind_known = wind_direction is not None and pd.notna(wind_direction)
        wind_building = wind_known and is_southern_direction(wind_direction) and bft >= 3
        oprez_at, opasno_at = (0.8, 1.5) if wind_building else (1.0, 1.8)
        if wave_height >= opasno_at:
            label = _worst("Opasno")
        elif wave_height >= oprez_at:
            label = _worst("Oprez")

    if wind_gusts_ms is not None and pd.notna(wind_gusts_ms):
        gust_kn = float(wind_gusts_ms) / MS_PER_KNOT
        if gust_kn > 34:
            label = _worst("Opasno")
        elif gust_kn > 22:
            label = _worst("Oprez")
        elif gust_kn > 15:
            label = _worst("Prihvatljivo")

    return (_SAILING_COLOR[label], label)


def _fetch_marine_waves(lat, lon, label):
    """Fetch wave models for ONE point with disk cache. Returns DataFrame
    with *_mean columns, or None if nothing succeeded."""
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    all_waves = {}

    def _parse(h, model):
        d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
        added = 0
        for v in MARINE_VARS:
            if v in h:
                d[f"{model}_{v}"] = h[v]
                added += 1
        return d, added

    for model in MARINE_MODELS:
        update_h = MODEL_UPDATE_HOURS.get(model, DEFAULT_UPDATE_HOURS)
        cache_path = _cache_path('marine_wave', model, lat, lon)

        upstream_time = _get_upstream_update_time(model)
        use_cache = False
        cache_reason = ""
        if upstream_time is not None and _cache_current_for_upstream(cache_path, upstream_time):
            use_cache = True
            cache_reason = f"upstream @ {time.strftime('%H:%M UTC', time.gmtime(upstream_time))}"
        elif upstream_time is None:
            cached_ttl = _load_fresh_cache(cache_path, update_h)
            if cached_ttl is not None:
                use_cache = True
                cache_reason = f"TTL {update_h}h"

        if use_cache:
            stale_data, _ = _load_stale_cache(cache_path)
            if stale_data is not None:
                d, added = _parse(stale_data.get('hourly', {}), model)
                all_waves[model] = d
                age_h = _cache_age_hours(cache_path)
                print(f"  [{label}] Wave {model}: CACHE ({len(d)}h, {(age_h or 0):.1f}h, {cache_reason})")
                continue

        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(MARINE_VARS),
            "timezone": FORECAST_TIMEZONE,
            "models": model,
            "forecast_days": 7,
        }
        success = False
        for attempt in range(3):
            try:
                r = requests.get(marine_url, params=params, timeout=30)
                r.raise_for_status()
                resp = r.json()
                d, added = _parse(resp.get('hourly', {}), model)
                all_waves[model] = d
                _save_cache(cache_path, resp)
                _save_upstream_meta(cache_path, upstream_time)
                print(f"  [{label}] Wave {model}: OK ({len(d)}h, {added} varijabli)")
                success = True
                break
            except Exception as e:
                if attempt == 2:
                    stale, age_h = _load_stale_cache(cache_path)
                    if stale is not None:
                        d, added = _parse(stale.get('hourly', {}), model)
                        all_waves[model] = d
                        print(f"  [{label}] Wave {model}: FAIL ({e}); STALE ({(age_h or 0):.1f}h staro)")
                    else:
                        print(f"  [{label}] Wave {model}: FAIL ({e})")
                else:
                    time.sleep(5)
        time.sleep(1.0)

    if not all_waves:
        return None

    df = list(all_waves.values())[0]
    for k in list(all_waves.keys())[1:]:
        df = df.merge(all_waves[k], on='datetime', how='outer')
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for v in MARINE_VARS:
        cols = [f"{m}_{v}" for m in MARINE_MODELS if f"{m}_{v}" in df.columns]
        if cols:
            values = df[cols].apply(pd.to_numeric, errors='coerce')
            if v.endswith('_direction'):
                df[f'{v}_mean'] = _circular_mean_degrees(values)
            else:
                df[f'{v}_mean'] = values.mean(axis=1)

    now = local_now().floor('h')
    df = df[df['datetime'] >= now].copy().reset_index(drop=True)
    return df


def _fetch_marine_wind(lat, lon, label):
    """Fetch wind models for ONE point with disk cache. Returns DataFrame
    with ensemble mean columns."""
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    all_winds = {}

    def _parse(h, model_id):
        d = pd.DataFrame({'datetime': pd.to_datetime(h.get('time', []))})
        for v in ['wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m']:
            if v in h:
                d[f"{model_id}_{v}"] = h[v]
        return d

    for model_id in MARINE_WIND_MODELS:
        update_h = MODEL_UPDATE_HOURS.get(model_id, DEFAULT_UPDATE_HOURS)
        cache_path = _cache_path('marine_wind', model_id, lat, lon)

        upstream_time = _get_upstream_update_time(model_id)
        use_cache = False
        cache_reason = ""
        if upstream_time is not None and _cache_current_for_upstream(cache_path, upstream_time):
            use_cache = True
            cache_reason = f"upstream @ {time.strftime('%H:%M UTC', time.gmtime(upstream_time))}"
        elif upstream_time is None:
            cached_ttl = _load_fresh_cache(cache_path, update_h)
            if cached_ttl is not None:
                use_cache = True
                cache_reason = f"TTL {update_h}h"

        if use_cache:
            stale_data, _ = _load_stale_cache(cache_path)
            if stale_data is not None:
                d = _parse(stale_data.get('hourly', {}), model_id)
                all_winds[model_id] = d
                age_h = _cache_age_hours(cache_path)
                print(f"  [{label}] Wind {model_id}: CACHE ({len(d)}h, {(age_h or 0):.1f}h, {cache_reason})")
                continue

        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m",
            "timezone": FORECAST_TIMEZONE,
            "wind_speed_unit": "ms",
            "models": model_id, "forecast_days": 7,
        }
        try:
            r = requests.get(forecast_url, params=params, timeout=30)
            r.raise_for_status()
            resp = r.json()
            d = _parse(resp.get('hourly', {}), model_id)
            all_winds[model_id] = d
            _save_cache(cache_path, resp)
            _save_upstream_meta(cache_path, upstream_time)
            print(f"  [{label}] Wind {model_id}: OK ({len(d)}h)")
        except Exception as e:
            stale, age_h = _load_stale_cache(cache_path)
            if stale is not None:
                d = _parse(stale.get('hourly', {}), model_id)
                all_winds[model_id] = d
                print(f"  [{label}] Wind {model_id}: FAIL ({e}); STALE ({(age_h or 0):.1f}h staro)")
            else:
                print(f"  [{label}] Wind {model_id}: FAIL ({e})")
        time.sleep(1.0)

    if not all_winds:
        return None

    df = list(all_winds.values())[0]
    for k in list(all_winds.keys())[1:]:
        df = df.merge(all_winds[k], on='datetime', how='outer')
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for v in ['wind_speed_10m', 'wind_gusts_10m']:
        cols = [c for c in df.columns
                if any(c == f"{m}_{v}" for m in MARINE_WIND_MODELS)]
        if cols:
            df[f'{v}_mean'] = df[cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    dir_cols = [c for c in df.columns
                if any(c == f"{m}_wind_direction_10m" for m in MARINE_WIND_MODELS)]
    if dir_cols:
        dirs = df[dir_cols].apply(pd.to_numeric, errors='coerce')
        sin_mean = np.sin(np.radians(dirs)).mean(axis=1)
        cos_mean = np.cos(np.radians(dirs)).mean(axis=1)
        df['wind_direction_10m_mean'] = (np.degrees(np.arctan2(sin_mean, cos_mean)) % 360)

    now = local_now().floor('h')
    df = df[df['datetime'] >= now].copy().reset_index(drop=True)
    return df


def fetch_live_marine():
    """Fetch ONE wave dataset + wind for each MARINE_WIND_LOCATIONS.

    Returns dict {'waves': df_or_None, 'winds': [{loc_meta..., 'df': df}, ...]}
    or None if everything failed."""
    print("\n[Marine] Preuzimanje pomorske prognoze...")
    print(f"  Talasi: jedna reprezentativna tačka {MARINE_WAVE_LOCATION['name']} "
          f"({MARINE_WAVE_LOCATION['lat']}, {MARINE_WAVE_LOCATION['lon']})")
    waves = _fetch_marine_waves(
        MARINE_WAVE_LOCATION['lat'], MARINE_WAVE_LOCATION['lon'],
        MARINE_WAVE_LOCATION['name']
    )

    print(f"  Vjetar: {len(MARINE_WIND_LOCATIONS)} lokacije, finija rezolucija "
          f"({', '.join(MARINE_WIND_MODELS)})")
    winds = []
    for loc in MARINE_WIND_LOCATIONS:
        wdf = _fetch_marine_wind(loc['lat'], loc['lon'], loc['name'])
        if wdf is not None:
            winds.append({**loc, 'df': wdf})

    if waves is None and not winds:
        print("  Marine: ništa nije fetched.")
        return None
    return {'waves': waves, 'winds': winds}


def _build_wave_block(waves_df):
    """Build the SHARED wave block (single location). Hourly + daily."""
    if waves_df is None or waves_df.empty:
        return None

    now_ts = local_now().floor('h')
    cutoff_48h = now_ts + pd.Timedelta(hours=48)

    hourly = []
    for _, row in waves_df.iterrows():
        if row['datetime'] >= cutoff_48h:
            break
        wh = row.get('wave_height_mean')
        wp = row.get('wave_period_mean')
        wd = row.get('wave_direction_mean')
        ww_h = row.get('wind_wave_height_mean')
        sw_h = row.get('swell_wave_height_mean')
        sw_p = row.get('swell_wave_period_mean')
        sst = row.get('sea_surface_temperature_mean')
        sea = adriatic_sea_state(wh, wd)
        hourly.append({
            "datetime": row['datetime'].isoformat(),
            "hour": int(row['datetime'].hour),
            "date": row['datetime'].strftime('%Y-%m-%d'),
            "wave_height": round(float(wh), 2) if pd.notna(wh) else None,
            "wave_period": round(float(wp), 1) if pd.notna(wp) else None,
            "wave_direction": round(float(wd), 0) if pd.notna(wd) else None,
            "wind_wave_height": round(float(ww_h), 2) if pd.notna(ww_h) else None,
            "swell_wave_height": round(float(sw_h), 2) if pd.notna(sw_h) else None,
            "swell_wave_period": round(float(sw_p), 1) if pd.notna(sw_p) else None,
            "sea_surface_temperature": round(float(sst), 1) if pd.notna(sst) else None,
            "sea_state": sea,
            "sea_state_label": DOUGLAS_LABELS.get(sea, "") if sea is not None else "",
        })

    df = waves_df.copy()
    df['_date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    daily = []
    for date_str, grp in df.groupby('_date', sort=True):
        wh_series = pd.to_numeric(grp.get('wave_height_mean', pd.Series()), errors='coerce')
        wh_max = wh_series.max()
        wp_max = pd.to_numeric(grp.get('wave_period_mean', pd.Series()), errors='coerce').max()
        sst_avg = pd.to_numeric(grp.get('sea_surface_temperature_mean', pd.Series()), errors='coerce').mean()
        # Direction at the hour of the daily maximum, so the direction cap
        # judges the same wave the label describes.
        wd_at_max = None
        if wh_series.notna().any() and 'wave_direction_mean' in grp.columns:
            wd_at_max = pd.to_numeric(
                grp['wave_direction_mean'], errors='coerce').get(wh_series.idxmax())
        sea_max = adriatic_sea_state(wh_max, wd_at_max) if pd.notna(wh_max) else None
        daily.append({
            "date": date_str,
            "day_name": pd.Timestamp(date_str).strftime('%A'),
            "wave_height_max": round(float(wh_max), 2) if pd.notna(wh_max) else None,
            "wave_period_max": round(float(wp_max), 1) if pd.notna(wp_max) else None,
            "wave_dir_at_max": round(float(wd_at_max), 0) if wd_at_max is not None and pd.notna(wd_at_max) else None,
            "sst_avg": round(float(sst_avg), 1) if pd.notna(sst_avg) else None,
            "sea_state_max": sea_max,
            "sea_state_label": DOUGLAS_LABELS.get(sea_max, "") if sea_max is not None else "",
        })

    return {"hourly": hourly, "daily_summary": daily[:7]}


def _build_wind_block(wind_df, shared_waves_by_dt, shared_waves_daily_by_date):
    """Build a wind location block. Sailing score combines this location's
    wind Bft with the shared wave height at the same hour/day."""
    if wind_df is None or wind_df.empty:
        return None

    now_ts = local_now().floor('h')
    cutoff_48h = now_ts + pd.Timedelta(hours=48)

    hourly = []
    for _, row in wind_df.iterrows():
        if row['datetime'] >= cutoff_48h:
            break
        ws = row.get('wind_speed_10m_mean')
        wg = row.get('wind_gusts_10m_mean')
        wdir = row.get('wind_direction_10m_mean')
        bft = beaufort_from_wind(ws)
        wind_name = cg_wind_name(wdir, ws)
        # Pair with shared wave height + direction at the same hour.
        dt_key = row['datetime']
        wh_here, wd_here = shared_waves_by_dt.get(dt_key, (None, None))
        score_color, score_label = sailing_score(
            bft, wh_here, wave_direction=wd_here,
            wind_direction=wdir, wind_gusts_ms=wg)
        hourly.append({
            "datetime": dt_key.isoformat(),
            "hour": int(dt_key.hour),
            "date": dt_key.strftime('%Y-%m-%d'),
            "wind_speed_10m": round(float(ws), 1) if pd.notna(ws) else None,
            "wind_gusts_10m": round(float(wg), 1) if pd.notna(wg) else None,
            "wind_direction_10m": round(float(wdir), 0) if pd.notna(wdir) else None,
            "beaufort": bft,
            "wind_name": wind_name,
            "sailing_score": score_label,
            "sailing_color": score_color,
        })

    df = wind_df.copy()
    df['_date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    daily = []
    for date_str, grp in df.groupby('_date', sort=True):
        ws_max = pd.to_numeric(grp.get('wind_speed_10m_mean', pd.Series()), errors='coerce').max()
        wg_max = pd.to_numeric(grp.get('wind_gusts_10m_mean', pd.Series()), errors='coerce').max()
        peak_dir = None
        ws_series = pd.to_numeric(grp.get('wind_speed_10m_mean', pd.Series()), errors='coerce')
        if ws_series.notna().any():
            idx = ws_series.idxmax()
            peak_dir = grp.loc[idx, 'wind_direction_10m_mean'] if 'wind_direction_10m_mean' in grp.columns else None
        bft_max = beaufort_from_wind(ws_max) if pd.notna(ws_max) else None
        wind_name = cg_wind_name(peak_dir, ws_max)
        wh_here, wd_here = shared_waves_daily_by_date.get(date_str, (None, None))
        # Score off the sustained gust, not the daily peak: one spike at 23h
        # must not cost the whole day a rung. The card still shows the peak.
        wg_persist = persistent_daytime_gust(grp)
        score_color, score_label = sailing_score(
            bft_max, wh_here, wave_direction=wd_here,
            wind_direction=peak_dir, wind_gusts_ms=wg_persist)
        daily.append({
            "date": date_str,
            "day_name": pd.Timestamp(date_str).strftime('%A'),
            "wind_speed_max": round(float(ws_max), 1) if pd.notna(ws_max) else None,
            "wind_gusts_max": round(float(wg_max), 1) if pd.notna(wg_max) else None,
            "wind_gusts_persistent": round(wg_persist, 1) if wg_persist is not None else None,
            "wind_direction_peak": round(float(peak_dir), 0) if peak_dir is not None and pd.notna(peak_dir) else None,
            "beaufort_max": bft_max,
            "wind_name": wind_name,
            "sailing_score": score_label,
            "sailing_color": score_color,
        })

    return {"hourly": hourly, "daily_summary": daily[:7]}


def build_marine_output(marine_results):
    """marine_results = {'waves': df_or_None, 'winds': [{loc_meta, 'df': df}]}.
    Returns dict with shared waves block + per-location wind blocks."""
    if not marine_results:
        return None

    waves_block = _build_wave_block(marine_results.get('waves'))

    # Build lookups so wind locations can pair their sailing score with the
    # shared wave height AND direction at the matching hour/day — the score
    # needs the bearing to tell a jugo sea from an off-the-land artifact.
    shared_waves_by_dt = {}
    shared_waves_daily = {}
    if waves_block:
        for h in waves_block['hourly']:
            wh = h.get('wave_height')
            if wh is not None:
                shared_waves_by_dt[pd.Timestamp(h['datetime'])] = (
                    wh, h.get('wave_direction'))
        for d in waves_block['daily_summary']:
            wh = d.get('wave_height_max')
            if wh is not None:
                shared_waves_daily[d['date']] = (wh, d.get('wave_dir_at_max'))

    wind_locations = []
    for loc in marine_results.get('winds', []):
        body = _build_wind_block(loc['df'], shared_waves_by_dt, shared_waves_daily)
        if body is None:
            continue
        wind_locations.append({
            "id": loc['id'],
            "name": loc['name'],
            "desc": loc.get('desc', ''),
            "lat": loc['lat'],
            "lon": loc['lon'],
            "hourly": body['hourly'],
            "daily_summary": body['daily_summary'],
        })

    if not waves_block and not wind_locations:
        return None

    return {
        "note": (
            "Talasi su modelirani na rezoluciji od 5.5km, pa pokazujemo jednu vrijednost "
            "za cijelu pomorsku zonu Budve (Bečićka plaža i otvoreno more dijele istu "
            "prognozu). Vjetar je precizniji (~2 km rezolucija) i razlikuje se "
            "između zaliva i otvorenog mora. Stanje mora uzima u obzir i smjer talasa: "
            "velike talase u Bečićima donosi samo jugo (JI–J), bura (SI) pravi kratke i "
            "oštre ali manje talase, a smjerovi sa kopna (S, Z) ne mogu dati više od "
            "blagih talasa."
        ),
        "wave_location": {
            "lat": MARINE_WAVE_LOCATION['lat'],
            "lon": MARINE_WAVE_LOCATION['lon'],
            "name": MARINE_WAVE_LOCATION['name'],
            "desc": MARINE_WAVE_LOCATION['desc'],
        },
        "waves": waves_block,
        "wind_locations": wind_locations,
    }


def escalate_storm_code(code, wind_ms, thunder):
    """Raise a rain code to a stormier one when wind/thunder make a light-rain
    icon misleading: in Budva a few mm with strong wind or thunder reads as a
    storm, not "Slaba/Sitna kiša". Only rain codes (51-82) escalate; clear /
    cloud / fog / snow and existing thunder (>=95) pass through unchanged.

        thunder present                  -> 95 Grmljavina
        sustained wind > STORM_WIND_MS   -> 80 Pljuskovi (light 51/61)
                                            82 Jaki pljuskovi (else)
    """
    try:
        c = int(code)
    except (TypeError, ValueError):
        return code
    # Only actual rain / drizzle / showers escalate — NOT snow (71-77) or
    # freezing drizzle/rain (56, 57, 66, 67), which also live inside 51-82.
    if c not in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        return code
    if thunder:
        return 95
    if wind_ms is not None and not pd.isna(wind_ms) and float(wind_ms) > STORM_WIND_MS:
        return 80 if c in (51, 61) else 82
    return code


def correct_weather_code_row(row, raw_row=None):
    """
    Models often report rain (WC >= 51) during winter overcast conditions
    when it's actually just cloudy. XGBoost precipitation correction is more accurate,
    so we trust it over the raw weather code mode.
    
    This is how we're going to fix:
    - If XGBoost says precip below display threshold AND raw WC is rain/drizzle → downgrade to cloud-based code
    - If XGBoost says precip > 0 but raw WC is clear → upgrade to appropriate rain code
    - Use cloud cover to determine the correct non-rain code
    """
    wc_raw_val = row.get('weather_code_raw', row.get('weather_code', 0))
    if wc_raw_val is None or pd.isna(wc_raw_val):
        wc_raw = 0
    else:
        wc_raw = int(wc_raw_val)
    
    precip_xgb = row.get('precipitation_xgb', None)
    cloud_xgb = row.get('cloud_cover_xgb', None)
    
    # Do not classify snow or freezing precipitation as ordinary rain merely
    # because their WMO codes happen to lie numerically between 51 and 82.
    is_rain_code = wc_raw in (51, 53, 55, 61, 63, 65, 80, 81, 82)
    is_thunderstorm = wc_raw >= 95
    
    # We don't want to mess with thunderstorm codes, as they are often correct and XGBoost precip can be underestimated in convective events
    if is_thunderstorm:
        return wc_raw
    
    if is_rain_code and precip_xgb is not None and pd.notna(precip_xgb) and precip_xgb < CORRECTED_RAIN_THRESHOLD_MM:
        if cloud_xgb is not None and pd.notna(cloud_xgb):
            if cloud_xgb > 80:
                return 3   
            elif cloud_xgb > 50:
                return 2   
            elif cloud_xgb > 20:
                return 1   
            else:
                return 0   
        return 3 
    
    if 51 <= wc_raw <= 55 and precip_xgb is not None and pd.notna(precip_xgb):
        if precip_xgb < CORRECTED_RAIN_THRESHOLD_MM:
            if cloud_xgb is not None and pd.notna(cloud_xgb) and cloud_xgb > 85:
                return 3  
    
    if wc_raw <= 3 and precip_xgb is not None and pd.notna(precip_xgb) and precip_xgb >= CORRECTED_RAIN_THRESHOLD_MM:
        if precip_xgb > 3.0:
            return 63  
        elif precip_xgb > 1.0:
            return 61  
        else:
            return 51  
    
    if wc_raw <= 3 and cloud_xgb is not None and pd.notna(cloud_xgb):
        if cloud_xgb > 80:
            return 3   
        elif cloud_xgb > 50:
            return 2   
        elif cloud_xgb > 20:
            return 1  
        else:
            return 0   
    
    return wc_raw


class TrustedRainGateError(RuntimeError):
    """Raised when the trusted rain model is unavailable for the gate.

    Used so the main script can specifically catch THIS failure mode and fall
    back to the previous run, instead of writing a silently-dry forecast.
    """
    pass


def _detect_short_rain_bursts(precip_vals, wet_threshold,
                              max_hours_base, max_hours_extended,
                              ext_intensity_mm, ext_sum_mm):
    """Return mask of hours that belong to a short isolated wet run.

    Coastal Adriatic summers produce convective cells that drift over Budva.
    Short (1-2h) cells are always treated as bursts. Slow-moving heavy cells
    (3-4h) also count if they carry enough water — either a single hour at
    >= ext_intensity_mm or a total run sum >= ext_sum_mm. Anything longer is
    stratiform / synoptic and gets ignored here.
    """
    n = len(precip_vals)
    burst_mask = np.zeros(n, dtype=bool)
    if n == 0:
        return burst_mask
    wet_mask = precip_vals >= wet_threshold
    i = 0
    while i < n:
        if wet_mask[i]:
            j = i
            while j < n and wet_mask[j]:
                j += 1
            run_len = j - i
            if run_len <= max_hours_base:
                burst_mask[i:j] = True
            elif run_len <= max_hours_extended:
                run_slice = precip_vals[i:j]
                if run_slice.max() >= ext_intensity_mm or run_slice.sum() >= ext_sum_mm:
                    burst_mask[i:j] = True
            i = j
        else:
            i += 1
    return burst_mask


def _burst_ambient_ceiling(corrected, base, factor, floor_base):
    """Highest value the burst floor may lift `base` to, per hour.

    Reads the hour's calibrated q90 band, falling back to the pre-boost
    ensemble point value for model bundles that predate the quantile upgrade,
    and to the base floor where neither is finite.
    """
    ambient = corrected.get(f'{base}_q90')
    if ambient is None:
        ambient = corrected.get(f'{base}_ensemble')
    if ambient is None:
        return np.full(len(corrected), floor_base)
    vals = pd.to_numeric(ambient, errors='coerce').values.astype(float)
    return np.where(np.isnan(vals), floor_base, np.maximum(vals * factor, floor_base))


def _apply_burst_wind_boost(corrected, fc):
    """Boost wind & gust predictions for short isolated rain bursts.

    Operates AFTER the param loop, modifying `corrected` in place. Reads the
    trusted-model precip column from `fc`, detects bursts (1-2h always, 3-4h
    only when intensity carries), then applies the wind boost only on hours
    where italia precip >= BURST_BOOST_PRECIP_MM. Floors scale linearly with
    in-hour intensity and are bounded by the hour's own q90 band; halo (+/-1h
    around boosted hours) gets a weak additive bump.

    NaN handling: NaN inputs stay NaN — we don't manufacture observations.
    """
    trusted_col = f'{TRUSTED_RAIN_MODEL}_precipitation_model'
    if trusted_col not in fc.columns:
        return

    # Raw Open-Meteo precipitation is end-of-hour labelled. Align it to the
    # already shifted output rows before deciding which instantaneous wind/gust
    # values to boost.
    italia_precip = pd.to_numeric(fc[trusted_col], errors='coerce').shift(-1).fillna(0).values
    burst_mask = _detect_short_rain_bursts(
        italia_precip, TRUSTED_RAIN_THRESHOLD,
        BURST_MAX_HOURS, BURST_MAX_HOURS_EXTENDED,
        BURST_EXTENDED_MAX_MM, BURST_EXTENDED_SUM_MM,
    )

    if not burst_mask.any():
        return

    # Wind boost only fires inside a burst when italia precip clears the
    # in-hour threshold. Slabe kise (<2 mm/h) ne diraju vjetar.
    boost_mask = burst_mask & (italia_precip >= BURST_BOOST_PRECIP_MM)

    n_burst = int(burst_mask.sum())
    if not boost_mask.any():
        print(f"  Kratki pljuskovi: {n_burst} sat(a) detektovano, "
              f"nijedan ne prelazi {BURST_BOOST_PRECIP_MM:.1f} mm/h -> "
              f"vjetar netaknut.")
        return

    # Halo: +/-1h around boosted hours (not all burst hours), excluding boost itself.
    halo_mask = np.zeros_like(boost_mask)
    halo_mask[:-1] |= boost_mask[1:]
    halo_mask[1:] |= boost_mask[:-1]
    halo_mask &= ~boost_mask

    # Dynamic floors per hour, scaling with italia precip above the gate.
    excess = np.maximum(italia_precip - BURST_BOOST_PRECIP_MM, 0.0)
    wind_floor_per_hour = np.minimum(
        BURST_WIND_FLOOR_BASE + BURST_WIND_FLOOR_SLOPE * excess,
        BURST_WIND_FLOOR_CAP,
    )
    gust_floor_per_hour = np.minimum(
        BURST_GUST_FLOOR_BASE + BURST_GUST_FLOOR_SLOPE * excess,
        BURST_GUST_FLOOR_CAP,
    )

    n_boost = int(boost_mask.sum())
    n_halo = int(halo_mask.sum())

    boost_spec = [
        ('wind_gusts_10m', gust_floor_per_hour, BURST_HALO_GUST_DELTA, BURST_GUST_MAX,
         BURST_AMBIENT_GUST_FACTOR, BURST_GUST_FLOOR_BASE),
        ('wind_speed_10m', wind_floor_per_hour, BURST_HALO_WIND_DELTA, BURST_WIND_MAX,
         BURST_AMBIENT_WIND_FACTOR, BURST_WIND_FLOOR_BASE),
    ]

    for base, floor_per_hour, halo_delta, hard_cap, amb_factor, floor_base in boost_spec:
        ceiling = _burst_ambient_ceiling(corrected, base, amb_factor, floor_base)
        effective_floor = np.minimum(floor_per_hour, ceiling)
        for suffix in ['_xgb', '_ensemble']:
            col = f'{base}{suffix}'
            if col not in corrected.columns:
                continue
            w = corrected[col].values.astype(float).copy()
            # Core: keep original if already above the dynamic floor; otherwise lift.
            w[boost_mask] = np.minimum(
                np.maximum(w[boost_mask], effective_floor[boost_mask]),
                hard_cap,
            )
            # Halo: weak additive boost, no floor, same cap. NaN propagates.
            w[halo_mask] = np.minimum(w[halo_mask] + halo_delta, hard_cap)
            corrected[col] = w

    boost_times = corrected.loc[boost_mask, 'datetime'].dt.strftime('%d.%m %H:%M').tolist()
    sample = ', '.join(boost_times[:6]) + (' ...' if len(boost_times) > 6 else '')
    print(f"  Kratki pljuskovi: {n_burst} sat(a) detektovano, "
          f"{n_boost} boost (>= {BURST_BOOST_PRECIP_MM:.1f} mm/h) + {n_halo} halo "
          f"-> vjetar/udari skalirani po intenzitetu.")
    print(f"    Boost sati: {sample}")


def apply_correction(fc_df, trained, bias_tables, local_dry_nowcast=False):
    print("\n[5/6] Primjena korekcije...")

    fc = apply_bias_features(fc_df.copy(), bias_tables)
    # true lead time in hours (0 at the current hour); training rows
    # carry 12 (day-0 archive) / 36 / 60 (previous-runs stack).
    _lead_now = local_now().floor('h')
    fc['lead_time'] = ((pd.to_datetime(fc['datetime']) - _lead_now)
                       .dt.total_seconds() / 3600.0).clip(lower=0.0, upper=72.0)
    fc = engineer_features(fc)

    corrected = fc[['datetime']].copy()

    for param in TARGET_PARAMS:
        ens = f'{param}_ens_mean'
        if ens in fc.columns:
            corrected[f'{param}_ensemble'] = fc[ens]

    for param, minfo in trained.items():
        features = minfo['features']
        available = [f for f in features if f in fc.columns]

        if len(available) < len(features) * 0.4:
            print(f"  {TARGET_PARAMS[param]['display']:20s} --- nedovoljno feature-a ({len(available)}/{len(features)})")
            continue

        X = fc[available].copy()
        for c in features:
            if c not in X.columns:
                X[c] = np.nan  # NaN passthrough — XGBoost handles missing natively
        X = X[features]  # keep NaN for XGBoost's sparsity-aware splits

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = pd.to_numeric(X[c], errors='coerce')  # keep NaN

        if param == 'precipitation' and 'precip_info' in minfo:
            pinfo = minfo['precip_info']
            method = pinfo['best_method']

            cls_proba_raw = pinfo['cls_model'].predict_proba(X)[:, 1]
            _iso = pinfo.get('iso_calibrator')
            cls_proba = _iso.transform(cls_proba_raw) if _iso is not None else cls_proba_raw
            thresh = pinfo['threshold']

            if pinfo.get('use_sqrt', False):
                reg_pred = np.square(np.clip(pinfo['reg_model'].predict(X), 0, None))
            else:
                reg_pred = np.clip(pinfo['reg_model'].predict(X), 0, None)

            single_pred = np.clip(pinfo['single_model'].predict(X), 0, None)
            single_pred[single_pred < CORRECTED_RAIN_THRESHOLD_MM] = 0.0

            if method == 'hard':
                pred = np.where(cls_proba >= thresh, reg_pred, 0.0)
            elif method == 'soft':
                pred = cls_proba * reg_pred
            elif method == 'sharp':
                pred = np.where(cls_proba >= thresh, 0.7 * reg_pred + 0.3 * single_pred, single_pred * cls_proba)
            elif method == 'adaptive':
                confidence = np.abs(cls_proba - 0.5) * 2
                pred = np.where(cls_proba >= thresh, confidence * reg_pred + (1 - confidence) * single_pred, (1 - confidence) * single_pred * 0.5)
            elif method == 'tweedie':
                pred = np.clip(pinfo['tweedie_model'].predict(X), 0, None)
            else:
                pred = single_pred

            pred = np.clip(pred, 0, 50)
            p_blend_alpha = pinfo.get('blend_alpha', 1.0)
            if p_blend_alpha < 1.0:
                ens_col_p = f'{param}_ens_mean'
                ens_vals_p = pd.to_numeric(fc[ens_col_p], errors='coerce').fillna(0).values if ens_col_p in fc.columns else np.zeros(len(X))
                pred = p_blend_alpha * pred + (1 - p_blend_alpha) * ens_vals_p
                pred = np.clip(pred, 0, 50)

            # False-alarm clamping with trusted-model rain gate.
            #
            # For this experiment, rain is allowed only when ItaliaMeteo sees
            # at least 0.1mm. KNMI and DMI still feed the ensemble/model, but
            # they do not open the rain gate.
            #
            # Hard-fail policy: if the trusted model column is missing entirely,
            # OR every hour in the 48h horizon is NaN, raise TrustedRainGateError
            # so main can fall back to the previous run instead of silently
            # producing a fully-dry forecast.
            def _trusted_precip_values(model_name):
                col = f'{model_name}_precipitation_model'
                if col not in fc.columns:
                    raise TrustedRainGateError(
                        f"Trusted rain model '{model_name}' nije fetched - "
                        f"kolona '{col}' nedostaje u live prognozama. "
                        f"Trusted gate ne moze da radi. Provjeri "
                        f"fetch_live_forecasts log iznad za FAIL poruke."
                    )
                vals = pd.to_numeric(fc[col], errors='coerce').values
                nan_mask = np.isnan(vals)
                horizon = min(48, len(vals))
                if horizon > 0 and int(nan_mask[:horizon].sum()) >= horizon:
                    raise TrustedRainGateError(
                        f"Trusted rain model '{model_name}' ima sve NaN u "
                        f"prvih {horizon} sati. Trusted gate ne moze da radi."
                    )
                return vals, nan_mask

            trusted_vals, trusted_nan = _trusted_precip_values(TRUSTED_RAIN_MODEL)
            n_nan = int(trusted_nan.sum())
            if n_nan > 0:
                print(f"  UPOZORENJE: {TRUSTED_RAIN_MODEL} ima NaN za "
                      f"{n_nan}/{len(trusted_vals)} sati. Ti sati se "
                      f"oznacavaju kao nepoznati, ne kao suvi.")
            # NaN -> 0 is only an internal comparison convenience. Before the
            # precipitation branch returns, those rows are restored to NaN so
            # "trusted model has no data" can never become a dry forecast.
            trusted_vals_filled = np.where(trusted_nan, 0.0, trusted_vals)
            trusted_signal = trusted_vals_filled >= TRUSTED_RAIN_THRESHOLD
            italiameteo_raw_signal = trusted_signal.copy()

            trusted_amount_alpha = float(
                pinfo.get('trusted_amount_alpha', 1.0)
            )
            if trusted_amount_alpha < 1.0:
                valid_trusted_amount = ~trusted_nan
                pred[valid_trusted_amount] = (
                    trusted_amount_alpha * pred[valid_trusted_amount]
                    + (1.0 - trusted_amount_alpha)
                    * trusted_vals_filled[valid_trusted_amount]
                )
                pred = np.clip(pred, 0.0, 50.0)

            # Hard trusted rain gate:
            # - ItaliaMeteo can trigger rain alone at 0.1mm.
            # - Otherwise (or NaN), the corrected model is forced dry.
            trusted_signal_amount = np.where(trusted_signal, trusted_vals_filled, 0)
            no_signal = ~trusted_signal

            # Summer convection abstention.
            # ItaliaMeteo's own May-2025 admission: ICON-2I systematically over-
            # predicts weakly-forced summer convection. When in summer AND only
            # high-res LAMs see rain (low global agreement), DON'T let ICON-2I
            # alone open the gate. Suppress the false-alarm fingerprint.
            convective_rescue = np.zeros(len(fc), dtype=bool)
            try:
                hours_dt = pd.to_datetime(fc['datetime'])
                month_arr = hours_dt.dt.month.values
                is_warm_season = np.isin(month_arr, [6, 7, 8, 9])
                rain_agree = pd.to_numeric(
                    fc.get('rain_agreement', pd.Series(0, index=fc.index)),
                    errors='coerce').fillna(0).values
                italiameteo_isolated_signal = (
                    is_warm_season & trusted_signal & (rain_agree <= 0.30)
                )
                convective_rescue = _convective_rescue_mask(
                    fc, italiameteo_isolated_signal
                )
                suppress_signal = (
                    italiameteo_isolated_signal & ~convective_rescue
                )
                n_rescued = int(convective_rescue.sum())
                n_suppressed = int(suppress_signal.sum())
                if n_suppressed > 0:
                    # Treat as no_signal unless native/spatial convective
                    # diagnostics independently support the isolated cell.
                    trusted_signal = trusted_signal & ~suppress_signal
                    no_signal = ~trusted_signal
                    trusted_signal_amount = np.where(trusted_signal, trusted_vals_filled, 0)
                    print(f"  Abstention: {n_suppressed} summer hour(s) with isolated "
                          f"ICON-2I rain signal SUPPRESSED (weakly-forced convection regime)")
                if n_rescued > 0:
                    print(f"  Convective rescue: {n_rescued} isolated ICON-2I "
                          f"hour(s) retained by lightning/showers/neighborhood + CAPE/CIN")
            except Exception as _e:
                pass  # be defensive; don't break inference if any field missing

            # Preserve the dedicated ICON-2I support signal separately from the
            # final gate.  The latter may later be replaced by a validated PoP
            # blend or adjusted by SKALA; this field always answers the useful
            # diagnostic question "did the trusted LAM support rain here?".
            italiameteo_signal = trusted_signal.copy()

            # --- calibrated PoP-blend gate (only when it beat the
            # single-LAM veto in the regime-conditional eval). The fetch-level
            # TrustedRainGateError integrity check above applies either way;
            # the summer abstention is ICON-2I-specific and stays trusted-only.
            _pb = pinfo.get('pop_blend') or {}
            pinfo_mode = pinfo.get('rain_gate_mode') or _pb.get('mode', 'trusted')
            if pinfo_mode == 'pop_blend':
                _pb_lr = pinfo.get('pop_blend_lr') or _pb.get('lr')
                _pb_cols = pinfo.get('pop_blend_cols') or _pb.get('cols')
                _pb_tau = float(pinfo.get('pop_blend_tau') or _pb.get('tau') or 0.5)
                if _pb_lr is not None and _pb_cols:
                    try:
                        M_live = _pop_blend_inputs(fc, cls_proba, _pb_cols)
                        pop_blend_p = _pb_lr.predict_proba(M_live)[:, 1]
                        trusted_signal = pop_blend_p >= _pb_tau
                        no_signal = ~trusted_signal
                        _anchor = pd.to_numeric(
                            fc.get('precip_ens_mean_rainy', pd.Series(0, index=fc.index)),
                            errors='coerce').fillna(0).values
                        trusted_signal_amount = np.where(
                            trusted_signal, np.maximum(_anchor, trusted_vals_filled), 0)
                        print(f"  PoP-blend gate aktivan (tau={_pb_tau:.2f}): "
                              f"{int(trusted_signal.sum())}h otvoreno")
                    except Exception as _e:
                        print(f"  PoP-blend gate fail ({_e}) — fallback na trusted gate")

            # 1. Sub-threshold noise -> 0 (always)
            pred[pred < CORRECTED_RAIN_THRESHOLD_MM] = 0.0
            # 2. No trusted gate -> no rain.
            pred[no_signal] = 0.0
            # 3. Amplification cap. Avoid a hard 0.5mm minimum when the ensemble
            #    is mostly dry; trusted support scales the cap with trusted amount.
            if 'precip_ens_p75' in fc.columns:
                p75_vals = pd.to_numeric(fc['precip_ens_p75'], errors='coerce').fillna(0).values
                cap = np.maximum(np.full(len(fc), CORRECTED_RAIN_THRESHOLD_MM), 1.5 * p75_vals)
                trusted_cap = 1.2 * trusted_signal_amount
                cap[trusted_signal] = np.maximum(cap[trusted_signal], trusted_cap[trusted_signal])
                pred = np.minimum(pred, cap)
            # 4. If the trusted gate sees rain, the corrected precipitation must
            #    track ItaliaMeteo more closely. Floor scales as 0.6 * italia
            #    so a 5 mm Italia hour can't collapse to 0.5 mm under a
            #    conservative XGBoost. Cap above (1.2 * italia) still bounds
            #    the upside, so we never fully equal trusted — just stop
            #    diverging when XGBoost is too dry.
            if trusted_signal.any():
                trusted_floor = np.maximum(
                    CORRECTED_RAIN_THRESHOLD_MM,
                    0.6 * trusted_signal_amount,
                )
                pred[trusted_signal] = np.maximum(pred[trusted_signal], trusted_floor[trusted_signal])

            # calibrated PoP (isotonic classifier proba), GATED by the
            # trusted rain gate so the probabilistic view agrees with the
            # deterministic one. Where the gate is closed (ICON-2I dry / summer
            # abstention), the system asserts dry, so PoP -> 0. Set BEFORE the
            # radar block so the radar nowcast can blend it.
            corrected['precipitation_pop'] = np.clip(
                np.where(no_signal, 0.0, np.maximum(cls_proba, thresh)),
                0.0, 1.0,
            )

            # --- SKALA radar nowcast as a WEIGHTED 0-6h member ---
            # Replaces "hard override" thinking: weight w(lead) falls linearly
            # from 1 (now) to 0 (+6h); PoP is blended, amounts are only nudged
            # (dry-suppressed or wet-floored) when the radar is confident.
            skala_rain_support = np.zeros(len(fc), dtype=bool)
            radar_nc = read_radar_nowcast()
            if radar_nc is not None:
                _now_b = local_now().floor('h')
                lead_h = ((fc['datetime'] - _now_b).dt.total_seconds() / 3600.0).values
                w_radar = np.clip(1.0 - lead_h / 6.0, 0.0, 1.0)
                w_radar[lead_h < 0] = 0.0
                p_radar = np.interp(np.clip(lead_h, 0.0, 2.0),
                                    [0.0, 0.25, 0.5, 1.0, 2.0],
                                    [radar_nc['p15'], radar_nc['p15'], radar_nc['p30'],
                                     radar_nc['p60'], radar_nc['p120']])
                if radar_nc['receding'] and not radar_nc['raining_now']:
                    # cell already passed Budva — don't let the 120-min bucket
                    # keep the next hours wet
                    p_radar = np.minimum(p_radar, 0.2)
                skala_rain_support = (w_radar > 0.0) & (p_radar >= 0.5)
                if 'precipitation_pop' in corrected.columns:
                    _pop = corrected['precipitation_pop'].values.astype(float)
                    corrected['precipitation_pop'] = np.clip(
                        (1.0 - w_radar) * _pop + w_radar * p_radar, 0.0, 1.0)
                dry_mask_r = ((w_radar > 0.3) & (p_radar < 0.15)
                              & (pred <= LOCAL_DRY_LIGHT_RAIN_MAX_MM) & (pred > 0))
                if dry_mask_r.any():
                    pred[dry_mask_r] = 0.0
                    print(f"  RADAR nowcast: {int(dry_mask_r.sum())}h slabe bliske kiše "
                          f"ugašeno (p60={radar_nc['p60']:.2f}, age={radar_nc['age_min']}min)")
                if radar_nc['raining_now'] or radar_nc['p30'] >= 0.8:
                    wet_mask_r = ((lead_h >= 0) & (lead_h <= 2)
                                  & (p_radar >= 0.8)
                                  & (pred < CORRECTED_RAIN_THRESHOLD_MM))
                    if wet_mask_r.any():
                        pred[wet_mask_r] = CORRECTED_RAIN_THRESHOLD_MM
                        print(f"  RADAR nowcast: kiša na radaru — floor {int(wet_mask_r.sum())}h "
                              f"na {CORRECTED_RAIN_THRESHOLD_MM}mm")

            if local_dry_nowcast:
                now_local = local_now()
                dry_start = now_local.floor('h')
                if now_local.minute >= 30:
                    dry_start += pd.Timedelta(hours=1)
                dry_end = dry_start + pd.Timedelta(hours=LOCAL_DRY_NOWCAST_HOURS)
                dry_window = (fc['datetime'] >= dry_start) & (fc['datetime'] <= dry_end)
                ens_vals_now = (
                    pd.to_numeric(fc.get('precipitation_ens_mean', pd.Series(0, index=fc.index)),
                                  errors='coerce').fillna(0).values
                )
                storm_wc_count = (
                    pd.to_numeric(fc.get('storm_wc_count', pd.Series(0, index=fc.index)),
                                  errors='coerce').fillna(0).values
                )
                light_rain = pred <= LOCAL_DRY_LIGHT_RAIN_MAX_MM
                strong_near_support = (ens_vals_now >= 0.5) | (storm_wc_count >= 1)
                dry_now_mask = dry_window.values & light_rain & ~strong_near_support
                pred[dry_now_mask] = 0.0

            pred[pred < CORRECTED_RAIN_THRESHOLD_MM] = 0.0

            # Explicit rain-signal contract for downstream consumers.  This is
            # intentionally distinct from precipitation amount: it exposes the
            # trusted-model support, the final ITALIAMETEO/XGB/SKALA decision,
            # and its calibrated probability without making clients reverse-
            # engineer those semantics from a rounded mm value.
            corrected['italiameteo_precipitation'] = np.where(
                trusted_nan, np.nan, trusted_vals_filled
            )
            corrected['italiameteo_rain_signal'] = np.where(
                trusted_nan, np.nan, italiameteo_raw_signal.astype(float)
            )
            corrected['italiameteo_rain_accepted'] = np.where(
                trusted_nan, np.nan, italiameteo_signal.astype(float)
            )
            corrected['rain_signal_convective_rescue'] = np.where(
                trusted_nan, np.nan, convective_rescue.astype(float)
            )
            corrected['rain_signal_skala_support'] = skala_rain_support.astype(float)
            corrected['rain_signal'] = np.where(
                trusted_nan, np.nan,
                (pred >= CORRECTED_RAIN_THRESHOLD_MM).astype(float),
            )
            corrected['rain_signal_confidence'] = corrected[
                'precipitation_pop'
            ].values

            # A missing trusted-gate value is unknown, not evidence of dry
            # weather. Other global models may still extend farther, but this
            # policy cannot make a trusted-gate decision for that hour.
            pred[trusted_nan] = np.nan
            if 'precipitation_pop' in corrected.columns:
                corrected.loc[trusted_nan, 'precipitation_pop'] = np.nan

            # Missing NWP input means unknown, not dry. The precipitation
            # branch returns before the generic missing-data guard below, so it
            # needs its own guard (especially beyond the trusted 48h window).
            precip_source_cols = [
                f'{m}_precipitation_model' for m in MODELS
                if f'{m}_precipitation_model' in fc.columns
            ]
            if precip_source_cols:
                no_precip_data = ~fc[precip_source_cols].notna().any(axis=1).values
                pred[no_precip_data] = np.nan
                if 'precipitation_pop' in corrected.columns:
                    corrected.loc[no_precip_data, 'precipitation_pop'] = np.nan

            corrected[f'{param}_xgb'] = pred
            # (precipitation_pop set above, before the radar block, and gated)
            method_lbl = method + (f'+blend({p_blend_alpha:.2f})' if p_blend_alpha < 1.0 else '')
            _mae_s = (f"{minfo['mae']:.3f}{TARGET_PARAMS[param]['unit']}"
                      if minfo.get('mae') is not None else "n/a (resumed)")
            print(f"  {TARGET_PARAMS[param]['display']:20s} (MAE={_mae_s}) [{method_lbl}]")
            continue

        method_name = minfo.get('method', 'direct')
        ens_col = minfo.get('model_ens_col', f'{param}_ens_mean')
        if ens_col not in fc.columns and minfo.get('dew_deficit_mode'):
            temp_ens = pd.to_numeric(
                fc.get('temperature_2m_ens_mean', pd.Series(np.nan, index=fc.index)),
                errors='coerce',
            )
            dew_ens = pd.to_numeric(
                fc.get('dew_point_2m_ens_mean', pd.Series(np.nan, index=fc.index)),
                errors='coerce',
            )
            fc[ens_col] = (temp_ens - dew_ens).clip(lower=0)
        elif ens_col not in fc.columns and minfo.get('csi_mode'):
            raw_solar = pd.to_numeric(
                fc.get('shortwave_radiation_ens_mean', pd.Series(np.nan, index=fc.index)),
                errors='coerce',
            )
            clear_sky_live = compute_clear_sky(fc['datetime']).clip(lower=1)
            fc[ens_col] = (raw_solar / clear_sky_live).clip(lower=0, upper=1.5)
            fc.loc[clear_sky_live <= 20, ens_col] = 0.0
        try:
            pred = _predict_nonprecip_bundle(minfo, X, fc, ens_col)
        except Exception as _e:
            if _DEVICE_REQUEST == 'cuda':
                raise RuntimeError(
                    f"{TARGET_PARAMS[param]['display']} GPU prediction failed: {_e}"
                ) from _e
            # Old/incomplete artifact bundles may not contain every component.
            # Keep the forecast alive, but make the degradation explicit.
            print(f"  {TARGET_PARAMS[param]['display']}: {minfo.get('method')} predict failed "
                  f"({_e}); fallback na direct model")
            pred = minfo['direct_model'].predict(X)
            method_name = f"{method_name}->direct-fallback"

        if param == 'relative_humidity_2m':
            pred = np.clip(pred, 0, 100)
        elif param == 'cloud_cover':
            pred = _postprocess_cloud_prediction(
                pred, fc, minfo.get('cloud_target_version', 1)
            )
        elif param in ['wind_speed_10m', 'wind_gusts_10m', 'shortwave_radiation']:
            pred = np.clip(pred, 0, None)

        # CSI back-transform for solar radiation
        if param == 'shortwave_radiation' and minfo.get('csi_mode', False):
            cs_prod = compute_clear_sky(fc['datetime']).values
            pred = pred * cs_prod
            pred = np.clip(pred, 0, None)

        # Dew deficit back-transform: Td = T_corrected - deficit
        if param == 'dew_point_2m' and minfo.get('dew_deficit_mode', False):
            pred = np.clip(pred, 0, None)  # deficit ≥ 0
            # Use corrected temperature if available, else ensemble mean
            if 'temperature_2m_xgb' in corrected.columns:
                t_vals = corrected['temperature_2m_xgb'].values
            else:
                t_ens = f'temperature_2m_ens_mean'
                t_vals = pd.to_numeric(fc[t_ens], errors='coerce').fillna(0).values if t_ens in fc.columns else np.zeros(len(pred))
            pred = t_vals - pred  # Td = T - deficit

        # --- Missing-data guard: fixes the long-range "-1/1 °C" bug ---
        # When NO model has data for an hour (far-out long-range hours, or a run
        # where a model's cache delivered nothing for those dates), the ensemble
        # mean is NaN. The residual/blend/stacked/ridge methods above used
        # ensemble.fillna(0), so the correction collapsed to ~0 + residual =
        # garbage (≈ -1/1 °C in summer). Mask those hours to NaN so the hourly
        # JSON omits them and the daily summary's min/max skip them — i.e. show
        # "no data" instead of fabricating a near-zero value.
        ens_col_g = f'{param}_ens_mean'
        if ens_col_g in fc.columns:
            _ens_nan = pd.to_numeric(fc[ens_col_g], errors='coerce').isna().values
            if _ens_nan.any():
                pred = np.asarray(pred, dtype=float).copy()
                pred[_ens_nan] = np.nan
                print(f"    {TARGET_PARAMS[param]['display']}: {int(_ens_nan.sum())} sati bez "
                      f"podataka modela -> NaN (umjesto 0-baziranog šuma)")

        corrected[f'{param}_xgb'] = pred
        _mae_s = (f"{minfo['mae']:.3f}{TARGET_PARAMS[param]['unit']}"
                  if minfo.get('mae') is not None else "n/a (resumed)")
        print(f"  {TARGET_PARAMS[param]['display']:20s} (MAE={_mae_s}) [{method_name}]")

    # --- predictive distributions (multi-quantile + CQR) + exceedance ---
    for param, minfo in trained.items():
        qb = minfo.get('quantiles')
        if not qb:
            continue
        qfeats = qb['features']
        Xq = pd.DataFrame({
            c: (pd.to_numeric(fc[c], errors='coerce') if c in fc.columns
                else pd.Series(np.nan, index=fc.index))
            for c in qfeats
        })
        lower = 0.0 if param in ('wind_speed_10m', 'wind_gusts_10m', 'precipitation') else None
        try:
            qdf = pf.predict_quantiles(qb['models'], Xq, offsets=qb['offsets'],
                                       lower_bound=lower)
        except Exception as _e:
            print(f"  Kvantili {param}: preskočeno ({_e})")
            continue

        # Zero-inflate the precip quantiles.
        # Precipitation is ~92% dry, but the bare quantile model (+ CQR widening)
        # leaks rain mass into the upper tail even on dry hours, which made the
        # exceedance probs and ECC scenarios contradict the (gated) dry point
        # forecast. Fix: a quantile level alpha is DRY whenever alpha <= 1 - pw,
        # where pw is the gated calibrated PoP. So a 1%-rain hour gets all
        # quantiles = 0; a 40%-rain hour keeps only its top quantiles wet.
        pw_precip = None
        if param == 'precipitation' and 'precipitation_pop' in corrected.columns:
            pw_precip = np.clip(corrected['precipitation_pop'].values.astype(float), 0.0, 1.0)
            alphas_arr = np.array(sorted(pf.DEFAULT_ALPHAS))
            qc_sorted = [f"q{int(round(a * 100)):02d}" for a in alphas_arr]
            V = qdf[qc_sorted].values.astype(float)
            V[alphas_arr[None, :] <= (1.0 - pw_precip)[:, None]] = 0.0
            qdf = pd.DataFrame(np.sort(V, axis=1), columns=qc_sorted)

        # mask hours with no model data (same guard as the point forecast)
        ens_col_q = f'{param}_ens_mean'
        _nan_mask = (pd.to_numeric(fc[ens_col_q], errors='coerce').isna().values
                     if ens_col_q in fc.columns else np.zeros(len(fc), dtype=bool))
        for col in qdf.columns:
            vals = qdf[col].values.astype(float).copy()
            vals[_nan_mask] = np.nan
            corrected[f'{param}_{col}'] = vals
        # Exceedance probabilities from the calibrated CDF
        try:
            if param == 'precipitation':
                for thr, name in ((1.0, 'p_precip_gt1'), (5.0, 'p_precip_gt5')):
                    ex = pf.exceedance_from_quantiles(qdf, pf.DEFAULT_ALPHAS, thr)
                    # P(Y > thr>0) can never exceed the wet probability P(Y>0)=pw;
                    # cap it so dry hours don't inherit a residual tail (0.05).
                    if pw_precip is not None:
                        ex = np.minimum(ex, pw_precip)
                    ex[_nan_mask] = np.nan
                    corrected[name] = ex
            elif param == 'wind_gusts_10m':
                for thr, name in ((10.0, 'p_gust_gt10'), (17.0, 'p_gust_gt17')):
                    ex = pf.exceedance_from_quantiles(qdf, pf.DEFAULT_ALPHAS, thr)
                    ex[_nan_mask] = np.nan
                    corrected[name] = ex
        except Exception as _e:
            print(f"  Exceedance {param}: preskočeno ({_e})")
        print(f"  Kvantili {param}: q05..q95 dodani")

    wc_cols = [f"{m}_weather_code_model" for m in MODELS if f"{m}_weather_code_model" in fc.columns]
    if wc_cols:
        corrected['weather_code_raw'] = fc[wc_cols].apply(pd.to_numeric, errors='coerce').mode(axis=1)[0]
        corrected['weather_code'] = corrected.apply(
            lambda r: correct_weather_code_row(r, fc.loc[r.name] if r.name in fc.index else None), axis=1
        )

    for extra in ['apparent_temperature', 'snowfall', 'rain',
                  'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
                  'surface_pressure']:
        ec = [f"{m}_{extra}_model" for m in MODELS if f"{m}_{extra}_model" in fc.columns]
        if ec:
            corrected[f'{extra}_ens'] = fc[ec].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # Wind direction: arithmetic mean of degrees is WRONG across the 0/360 wrap
    # (mean of 350° and 10° = 180°, due south, instead of 0°/north). Use the
    # speed-weighted vector mean computed in engineer_features.
    if 'wind_dir_vec_ens' in fc.columns:
        corrected['wind_direction_10m_ens'] = fc['wind_dir_vec_ens'].values
    else:
        wd_ec = [f"{m}_wind_direction_10m_model" for m in MODELS if f"{m}_wind_direction_10m_model" in fc.columns]
        if wd_ec:  # fallback: unit-vector circular mean
            rad = np.radians(fc[wd_ec].apply(pd.to_numeric, errors='coerce'))
            u = (-np.sin(rad)).mean(axis=1)
            v = (-np.cos(rad)).mean(axis=1)
            corrected['wind_direction_10m_ens'] = (np.degrees(np.arctan2(-u, -v)) % 360)

    # Open-Meteo labels accumulations / preceding-hour maxima at the END of
    # the hour: row T's precip/gusts cover [T-1h, T]. Shift these columns
    # back by 1 so a row labelled T means "from T to T+1h" -- intuitive on
    # the UI (precip at 12:00 = what falls between 12:00 and 13:00).
    # weather_code_raw follows precip (same convention) and shifts with it.
    # Sustained wind is instantaneous and therefore stays anchored to its row;
    # gust is a preceding-hour maximum and shifts with precipitation.
    shift_cols = [
        'precipitation_xgb', 'precipitation_ensemble',
        'wind_gusts_10m_xgb', 'wind_gusts_10m_ensemble',
        'weather_code_raw',
        'rain_ens', 'snowfall_ens',
        # distributional columns of end-of-hour-labelled quantities
        # follow their point forecasts (precip + wind; temp stays instantaneous).
        'precipitation_pop', 'p_precip_gt1', 'p_precip_gt5',
        'italiameteo_precipitation', 'italiameteo_rain_signal',
        'italiameteo_rain_accepted', 'rain_signal_convective_rescue',
        'rain_signal_skala_support', 'rain_signal', 'rain_signal_confidence',
        'p_gust_gt10', 'p_gust_gt17',
    ] + [c for c in corrected.columns
         if (c.startswith('precipitation_q') or c.startswith('wind_gusts_10m_q'))]
    for col in shift_cols:
        if col in corrected.columns:
            corrected[col] = corrected[col].shift(-1)

    # Boost after temporal alignment so instantaneous sustained wind is not
    # moved one hour merely because it is coupled to a shower interval.
    _apply_burst_wind_boost(corrected, fc)

    # weather_code MUST be recomputed AFTER the shift so it sees the shifted
    # precip (for "rain coming" decisions) AND the instantaneous cloud cover
    # at the same row (for cloud-based sky codes). Otherwise hour T's icon
    # represents hour T+1's sky -- e.g., 67% cloud display + "Vedro" icon
    # because the next hour happens to clear.
    if 'weather_code_raw' in corrected.columns:
        def _final_weather_code(r):
            raw_row = fc.loc[r.name] if r.name in fc.index else None
            code = correct_weather_code_row(r, raw_row)
            # Storm escalation: wind/thunder bump a rain code to pljuskovi/grmljavina.
            storm_n = 0
            if raw_row is not None:
                storm_n = pd.to_numeric(raw_row.get('storm_wc_count', 0), errors='coerce')
                if pd.isna(storm_n):
                    storm_n = 0
            return escalate_storm_code(code, r.get('wind_speed_10m_xgb'), storm_n >= 1)
        corrected['weather_code'] = corrected.apply(_final_weather_code, axis=1)

    return corrected


WMO_CODES = {
    0: {"desc": "Vedro", "icon": "clear", "emoji": "\u2600\ufe0f"},
    1: {"desc": "Pretezno vedro", "icon": "mostly_clear", "emoji": "\U0001f324\ufe0f"},
    2: {"desc": "Djelimicno oblacno", "icon": "partly_cloudy", "emoji": "\u26c5"},
    3: {"desc": "Oblacno", "icon": "cloudy", "emoji": "\u2601\ufe0f"},
    45: {"desc": "Magla", "icon": "fog", "emoji": "\U0001f32b\ufe0f"},
    48: {"desc": "Magla (inje)", "icon": "fog", "emoji": "\U0001f32b\ufe0f"},
    51: {"desc": "Sitna kisa", "icon": "light_rain", "emoji": "\U0001f326\ufe0f"},
    53: {"desc": "Sitna kisa, umjerena", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    55: {"desc": "Sitna kisa, jaka", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    61: {"desc": "Slaba kisa", "icon": "light_rain", "emoji": "\U0001f326\ufe0f"},
    63: {"desc": "Umjerena kisa", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    65: {"desc": "Jaka kisa", "icon": "heavy_rain", "emoji": "\U0001f327\ufe0f\U0001f327\ufe0f"},
    71: {"desc": "Slab snijeg", "icon": "snow", "emoji": "\U0001f328\ufe0f"},
    73: {"desc": "Umjeren snijeg", "icon": "snow", "emoji": "\U0001f328\ufe0f"},
    75: {"desc": "Jak snijeg", "icon": "heavy_snow", "emoji": "\u2744\ufe0f\u2744\ufe0f"},
    80: {"desc": "Slabi pljuskovi", "icon": "light_rain", "emoji": "\U0001f326\ufe0f"},
    81: {"desc": "Umjereni pljuskovi", "icon": "rain", "emoji": "\U0001f327\ufe0f"},
    82: {"desc": "Jaki pljuskovi", "icon": "heavy_rain", "emoji": "\u26c8\ufe0f"},
    95: {"desc": "Grmljavina", "icon": "thunderstorm", "emoji": "\u26c8\ufe0f"},
    96: {"desc": "Grmljavina + grad", "icon": "thunderstorm", "emoji": "\u26c8\ufe0f\U0001f9ca"},
    99: {"desc": "Jaka grmljavina", "icon": "thunderstorm", "emoji": "\u26c8\ufe0f\U0001f9ca"},
}


import narrative_variants as nv


def _daily_narrative(grp):
    def _col(name):
        return pd.to_numeric(grp.get(name, pd.Series(dtype=float)), errors='coerce')

    hr = grp['hour'].astype(int)
    cc = _col('cloud_cover')
    pr = _col('precipitation')
    ws = _col('wind_speed_10m')
    wg = _col('wind_gusts_10m')
    tp = _col('temperature_2m')
    wc = _col('weather_code')
    wd = _col('wind_direction_10m')

    def _period(h0, h1):
        mask = (hr >= h0) & (hr < h1)
        sub_cc = cc[mask].dropna()
        sub_pr = pr[mask].dropna()
        sub_wc = wc[mask].dropna()
        sub_ws = ws[mask].dropna()
        return {
            'cloud': float(sub_cc.mean()) if len(sub_cc) else None,
            'precip': float(sub_pr.sum()) if len(sub_pr) else 0,
            'precip_max_h': float(sub_pr.max()) if len(sub_pr) else 0,
            'has_rain': float(sub_pr.sum()) > 0.1 if len(sub_pr) else False,
            'rain_hours': int((sub_pr > 0.1).sum()) if len(sub_pr) else 0,
            'has_thunder': bool((sub_wc >= 95).any()) if len(sub_wc) else False,
            'has_snow': bool(((sub_wc >= 71) & (sub_wc <= 75)).any()) if len(sub_wc) else False,
            'has_fog': bool(((sub_wc >= 45) & (sub_wc <= 48)).any()) if len(sub_wc) else False,
            'wind_max': float(sub_ws.max()) if len(sub_ws) else 0,
            'n': int(mask.sum()),
        }

    night = _period(0, 6)
    morn = _period(6, 12)
    aftn = _period(12, 18)
    eve = _period(18, 24)

    total_precip = float(pr.sum()) if pr.notna().any() else 0
    rain_hours_total = int((pr > 0.1).sum()) if pr.notna().any() else 0
    max_wind = float(ws.max()) if ws.notna().any() else 0
    max_gust = float(wg.max()) if wg.notna().any() else 0
    temp_max = float(tp.max()) if tp.notna().any() else None
    temp_min = float(tp.min()) if tp.notna().any() else None

    wind_dir_str = ""
    if wd.notna().any():
        rad = np.radians(wd.dropna())
        avg_deg = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
        compass = ['S', 'SSI', 'SI', 'ISI', 'I', 'IJI', 'JI', 'JJI',
                    'J', 'JJZ', 'JZ', 'ZJZ', 'Z', 'ZSZ', 'SZ', 'SSZ']
        wind_dir_str = compass[round(avg_deg / 22.5) % 16]

    def sky(c):
        if c is None:
            return 'unknown'
        if c < 30:
            return 'clear'
        if c < 50:
            return 'mostly_clear'
        if c < 65:
            return 'partly_cloudy'
        if c < 85:
            return 'mostly_cloudy'
        return 'cloudy'

    ms, as_, es = sky(morn['cloud']), sky(aftn['cloud']), sky(eve['cloud'])
    rain_m, rain_a, rain_e = morn['has_rain'], aftn['has_rain'], eve['has_rain']
    has_thunder = night['has_thunder'] or morn['has_thunder'] or aftn['has_thunder'] or eve['has_thunder']
    has_snow = night['has_snow'] or morn['has_snow'] or aftn['has_snow'] or eve['has_snow']
    has_fog_morn = morn['has_fog']

    daytime_cc = cc[(hr >= 7) & (hr <= 19)].dropna()
    cloud_day_avg = float(daytime_cc.mean()) if len(daytime_cc) else 50

    if has_thunder:
        day_wc = 95
    elif has_snow and total_precip >= 3:
        day_wc = 75
    elif has_snow:
        day_wc = 73 if total_precip >= 1 else 71
    elif total_precip >= 10:
        day_wc = 65
    elif total_precip >= 3:
        day_wc = 63
    elif total_precip >= 0.5:
        day_wc = 61
    elif total_precip > 0.1:
        day_wc = 51
    elif has_fog_morn:
        day_wc = 45
    elif cloud_day_avg >= 65:
        day_wc = 3
    elif cloud_day_avg >= 50:
        day_wc = 2
    elif cloud_day_avg >= 30:
        day_wc = 1
    else:
        day_wc = 0

    if 51 <= day_wc <= 65 and rain_hours_total <= 4:
        if day_wc >= 63:
            day_wc = 80

    day_wmo = WMO_CODES.get(day_wc, WMO_CODES[0])

    # Per-day seed for deterministic phrasing variety: stable for a given day,
    # varied across days (no forecast-JSON churn). See narrative_variants.py.
    _seed = (f"{ms}{as_}{es}|{round(cloud_day_avg)}|{round(total_precip, 1)}"
             f"|{round(max_wind)}|{round(temp_max) if temp_max is not None else 0}")

    parts = []

    if has_snow:
        if rain_m and rain_a and rain_e:
            parts.append(nv.variant("snow_all_day", _seed) + f" ({total_precip:.0f} mm)")
        elif rain_m and not rain_a:
            parts.append(nv.variant("snow_morn_then_dry", _seed))
        elif not rain_m and rain_a:
            parts.append(nv.variant("snow_dry_then_pm", _seed))
        elif not rain_m and not rain_a and rain_e:
            parts.append(nv.variant("snow_dry_then_eve", _seed))
        else:
            parts.append(nv.variant("snow_heavy_intermittent" if total_precip >= 5
                                    else "snow_intermittent", _seed))
    elif has_thunder:
        if not rain_m and rain_a and not rain_e:
            parts.append(nv.variant("thunder_sun_then_storm", _seed))
        elif rain_m and not rain_a:
            parts.append(nv.variant("thunder_morn_then_calm", _seed))
        elif not rain_m and not rain_a and rain_e:
            parts.append(nv.variant("thunder_eve", _seed))
        elif rain_m and rain_a:
            parts.append(nv.variant("thunder_day", _seed))
        elif night['has_thunder'] and not morn['has_thunder'] and not aftn['has_thunder']:
            parts.append(nv.variant("thunder_night", _seed))
        else:
            parts.append(nv.variant("thunder_unstable", _seed))
        if total_precip >= 5:
            parts[0] += f" ({total_precip:.0f} mm)"
    # Any hour with measurable rain must surface in the narrative — even a
    # 2-hour shower on an otherwise fine day gets its note.
    elif total_precip > 0.2 or rain_hours_total >= 1:
        precip_str = f" ({total_precip:.1f} mm)" if total_precip >= 1 else ""
        if rain_m and rain_a and rain_e:
            if total_precip >= 15:
                parts.append(nv.variant("rain_all_day", _seed))
            elif total_precip >= 5:
                parts.append(f"Kiša tokom cijelog dana ({total_precip:.0f} mm)")
            elif total_precip >= 1:
                parts.append(nv.variant("rain_intermittent", _seed))
            else:
                parts.append(nv.variant("rain_light_intermittent", _seed))
        elif rain_m and rain_a and not rain_e:
            parts.append(nv.variant("rain_day_then_dry_eve", _seed))
        elif rain_m and not rain_a and not rain_e:
            if morn['precip'] >= 3:
                parts.append(f"Jača kiša prijepodne ({morn['precip']:.1f} mm), suvo i vedrije od podneva")
            else:
                parts.append(nv.variant("rain_morn_then_dry", _seed))
        elif rain_m and not rain_a and rain_e:
            parts.append(nv.variant("rain_morn_eve", _seed))
        elif not rain_m and rain_a and rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append(nv.variant("rain_sun_then_rain", _seed))
            else:
                parts.append(f"Kiša od podneva do kraja dana{precip_str}")
        elif not rain_m and rain_a and not rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append(nv.variant("rain_sun_then_rain_pm", _seed))
            else:
                parts.append(nv.variant("rain_cloud_then_rain_pm", _seed))
        elif not rain_m and not rain_a and rain_e:
            parts.append(nv.variant("rain_dry_then_eve", _seed))
        elif night['has_rain'] and not rain_m and not rain_a and not rain_e:
            if ms in ('clear', 'mostly_clear'):
                parts.append(nv.variant("rain_night_then_sun", _seed))
            else:
                parts.append(nv.variant("rain_night_then_dry", _seed))
        else:
            parts.append(f"Povremena kiša{precip_str}")
    elif has_fog_morn:
        if as_ in ('clear', 'mostly_clear'):
            parts.append(nv.variant("fog_then_sun", _seed))
        elif as_ in ('partly_cloudy',):
            parts.append(nv.variant("fog_then_partly", _seed))
        else:
            parts.append(nv.variant("fog_then_cloud", _seed))
    else:
        _short = {
            'clear': 'sky_clear_short', 'mostly_clear': 'sky_mostly_clear_short',
            'partly_cloudy': 'sky_partly_short', 'mostly_cloudy': 'sky_mostly_cloudy_short',
            'cloudy': 'sky_cloudy_short',
        }
        if morn['n'] == 0 and aftn['n'] == 0 and eve['n'] > 0:
            parts.append(nv.variant(_short.get(es, 'variable'), _seed))
        elif morn['n'] == 0 and aftn['n'] > 0:
            parts.append(nv.variant(_short.get(as_, 'variable'), _seed))
        elif ms == as_:
            _allday = {
                'clear': 'clear_all_day', 'mostly_clear': 'mostly_clear_all_day',
                'partly_cloudy': 'partly_steady', 'mostly_cloudy': 'mostly_cloudy_all_day',
                'cloudy': 'cloudy_all_day',
            }
            parts.append(nv.variant(_allday.get(ms, 'variable'), _seed))
        elif ms in ('clear', 'mostly_clear') and as_ in ('mostly_cloudy', 'cloudy'):
            parts.append(nv.variant("sun_to_cloud", _seed))
        elif ms in ('mostly_cloudy', 'cloudy') and as_ in ('clear', 'mostly_clear'):
            parts.append(nv.variant("cloud_to_sun", _seed))
        elif ms in ('clear', 'mostly_clear') and as_ == 'partly_cloudy':
            parts.append(nv.variant("sun_to_partly", _seed))
        elif ms == 'partly_cloudy' and as_ in ('clear', 'mostly_clear'):
            parts.append(nv.variant("partly_to_sun", _seed))
        elif ms == 'partly_cloudy' and as_ in ('mostly_cloudy', 'cloudy'):
            parts.append(nv.variant("increasing_cloud", _seed))
        elif ms in ('mostly_cloudy', 'cloudy') and as_ == 'partly_cloudy':
            parts.append(nv.variant("cloud_to_partly", _seed))
        else:
            parts.append(nv.variant("variable", _seed))

        if es != 'unknown' and len(parts) > 0:
            curr_end = as_ if as_ != 'unknown' else ms
            if curr_end in ('clear', 'mostly_clear') and es in ('mostly_cloudy', 'cloudy'):
                parts[0] += nv.variant("eve_clouding", _seed)
            elif curr_end in ('mostly_cloudy', 'cloudy') and es in ('clear', 'mostly_clear'):
                parts[0] += nv.variant("eve_clearing", _seed)

    wind_part = ""
    if max_wind >= 10:
        adj = nv.variant("wind_strong_adj", _seed)
        wind_part = f"{adj} {wind_dir_str} vjetar" if wind_dir_str else f"{adj} vjetar"
    elif max_wind >= 7:
        wind_part = f"vjetrovito ({wind_dir_str})" if wind_dir_str else "vjetrovito"
    elif max_wind >= 5:
        adj = nv.variant("wind_moderate_adj", _seed)
        wind_part = f"{adj} {wind_dir_str} vjetar" if wind_dir_str else f"{adj} vjetar"

    if wind_part:
        if max_gust >= 15:
            wind_part += f", udari do {max_gust:.0f} m/s"
        parts.append(wind_part)
    elif max_gust >= 15:
        parts.append(f"udari vjetra do {max_gust:.0f} m/s")

    if temp_max is not None:
        if temp_max >= 33:
            parts.append(nv.variant("temp_very_hot", _seed))
        elif temp_max >= 30:
            parts.append(nv.variant("temp_hot", _seed))
    if temp_min is not None:
        if temp_min <= -5:
            parts.append(nv.variant("temp_hard_frost", _seed))
        elif temp_min <= 0:
            parts.append(nv.variant("temp_frost", _seed))

    narrative = "Promjenljivo"
    if len(parts) == 1:
        narrative = parts[0]
    elif len(parts) > 1:
        narrative = f"{parts[0]}; {'; '.join(parts[1:])}"

    return {
        'narrative': narrative,
        'weather_code': day_wc,
        'weather_desc': day_wmo['desc'],
        'weather_icon': day_wmo['icon'],
        'weather_emoji': day_wmo['emoji'],
    }


def _hail_votes_by_datetime(fc_raw):
    """Per-hour count of models explicitly diagnosing thunderstorm with hail
    (WMO 96/99). Open-Meteo has no dedicated hail variable; these codes are
    the direct signal, and only some models emit them (ICON does for Budva),
    so a single vote already matters. Shifted -1 to the same start-of-hour
    convention as precipitation (Open-Meteo labels the preceding hour)."""
    if fc_raw is None:
        return {}
    cols = [f"{m}_weather_code_model" for m in MODELS
            if f"{m}_weather_code_model" in fc_raw.columns]
    if not cols:
        return {}
    vals = fc_raw[cols].apply(pd.to_numeric, errors='coerce')
    votes = ((vals == 96) | (vals == 99)).sum(axis=1).shift(-1)
    out = {}
    for dt, v in zip(fc_raw['datetime'], votes):
        if pd.notna(v) and v > 0:
            out[dt] = int(v)
    return out


def _daily_model_rain_probability(raw_group):
    """Daily NWP rain consensus using only models with data for that day."""
    votes = []
    for model_name in MODELS:
        column = f'{model_name}_precipitation_model'
        if column not in raw_group.columns:
            continue
        values = pd.to_numeric(raw_group[column], errors='coerce')
        valid = values.notna()
        if valid.any():
            votes.append(bool((values[valid] > 0.1).any()))
    if not votes:
        return None
    return round(sum(votes) / len(votes) * 100)


def _build_daily_summary(date_str, day_name, grp_df, fc_raw=None):
    """Build a single daily summary dict from a group of hourly forecast rows.
    Uses _daily_narrative for icon/desc/narrative (unified, not split).
    grp_df must have 'hour' column and XGBoost/ensemble columns.
    """
    def _v(col_xgb, col_ens):
        c = grp_df.get(col_xgb, grp_df.get(col_ens, pd.Series(dtype=float)))
        return pd.to_numeric(c, errors='coerce') if isinstance(c, pd.Series) else pd.Series(dtype=float)

    temp = _v('temperature_2m_xgb', 'temperature_2m_ensemble')
    wind = _v('wind_speed_10m_xgb', 'wind_speed_10m_ensemble')
    gusts = _v('wind_gusts_10m_xgb', 'wind_gusts_10m_ensemble')
    precip = _v('precipitation_xgb', 'precipitation_ensemble')
    humid = _v('relative_humidity_2m_xgb', 'relative_humidity_2m_ensemble')
    pres = _v('pressure_msl_xgb', 'pressure_msl_ensemble')
    cloud = _v('cloud_cover_xgb', 'cloud_cover_ensemble')

    ds = {"date": date_str, "day_name": day_name}
    if temp.notna().any():
        ds['temp_min'] = round(float(temp.min()), 1)
        ds['temp_max'] = round(float(temp.max()), 1)
    if wind.notna().any():
        ds['wind_max'] = round(float(wind.max()), 1)
    if gusts.notna().any():
        ds['gust_max'] = round(float(gusts.max()), 1)
    ds['precip_total'] = (
        round(float(precip.sum()), 1) if precip.notna().any() else None
    )
    # Hours with measurable rain — drives the "must mention rain" guarantee
    # in the narrative and in gemini_narrative.validate().
    ds['rain_hours'] = int((precip > 0.1).sum()) if precip.notna().any() else 0
    if humid.notna().any():
        ds['humidity_avg'] = round(float(humid.mean()), 0)
    if pres.notna().any():
        ds['pressure_avg'] = round(float(pres.mean()), 0)

    wd_s = pd.to_numeric(grp_df.get('wind_direction_10m_ens',
                         grp_df.get('wind_direction_10m', pd.Series(dtype=float))),
                         errors='coerce').dropna()
    if len(wd_s) > 0:
        rad = np.radians(wd_s)
        ds['wind_dir_avg'] = round(float(np.degrees(
            np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360), 0)

    hr = grp_df['hour'].astype(int)
    daytime_mask = (hr >= 7) & (hr <= 19)
    if cloud.notna().any() and daytime_mask.any():
        dc = cloud[daytime_mask].dropna()
        if len(dc) > 0:
            ds['cloud_cover_day'] = round(float(dc.mean()), 0)

    narr_df = pd.DataFrame({
        'hour': hr.values,
        'cloud_cover': cloud.values,
        'precipitation': precip.values,
        'wind_speed_10m': wind.values,
        'wind_gusts_10m': gusts.values,
        'temperature_2m': temp.values,
        'weather_code': pd.to_numeric(
            grp_df.get('weather_code', pd.Series(dtype=float)), errors='coerce').values,
        'wind_direction_10m': wd_s.reindex(grp_df.index).values if len(wd_s) > 0 else np.nan,
    })
    narr = _daily_narrative(narr_df)
    ds.update({
        'weather_code': narr['weather_code'],
        'weather_desc': narr['weather_desc'],
        'weather_icon': narr['weather_icon'],
        'weather_emoji': narr['weather_emoji'],
        'day_narrative': narr['narrative'],
    })

    if fc_raw is not None:
        raw_mask = fc_raw['datetime'].isin(grp_df['datetime'])
        raw_grp = fc_raw[raw_mask]
        rain_probability = _daily_model_rain_probability(raw_grp)
        if rain_probability is not None:
            ds['precip_probability'] = rain_probability

        # Median of per-model daily maxima. The multi-model ensemble MEAN
        # wipes out hourly peaks (models disagree on timing), so beyond 48h
        # every day's wind_max collapses to 1-2 m/s and reads "slab". The
        # median of each model's own daily max keeps a realistic peak.
        # Consumed (then dropped) for long-range days only — see the
        # daily/long_range split.
        for var, key in (('wind_speed_10m', 'wind_max_models'),
                         ('wind_gusts_10m', 'gust_max_models')):
            maxima = []
            for model_name in MODELS:
                column = f'{model_name}_{var}_model'
                if column not in raw_grp.columns:
                    continue
                values = pd.to_numeric(raw_grp[column], errors='coerce')
                if values.notna().any():
                    maxima.append(float(values.max()))
            if maxima:
                ds[key] = round(float(np.median(maxima)), 1)

    return ds


# Gemini narrative generation.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# AI narrative mode — flip this ONE line to switch (no other change needed):
#   'generate' = Gemini writes the FULL narrative from the data (richer wording,
#                but can hallucinate; rule-based _daily_narrative is the fallback);
#   'rephrase' = Gemini only REPHRASES the rule-based sentence, validated by a
#                deterministic guardrail (factual-only; see gemini_narrative.py).
NARRATIVE_MODE = 'generate'   # 'generate' | 'rephrase'
import gemini_narrative

def _gemini_narrative(date_str, hourly_rows):
    """Call Gemini to generate a short weather narrative from hourly data."""
    if not GEMINI_API_KEY or not hourly_rows:
        return None
    lines = []
    for h in hourly_rows:
        hour = h.get('hour', 0)
        temp = h.get('temperature_2m', h.get('temperature_2m_ensemble', '?'))
        hum = h.get('relative_humidity_2m', h.get('relative_humidity_2m_ensemble', '?'))
        wind = h.get('wind_speed_10m', h.get('wind_speed_10m_ensemble', '?'))
        press = h.get('surface_pressure', h.get('pressure_msl', '?'))
        cloud = h.get('cloud_cover', '?')
        # precipitation + weather_code can be NaN at the forecast horizon's
        # last row after the hourly shift; sanitize so the prompt stays clean.
        precip_raw = h.get('precipitation', h.get('precipitation_ensemble', 0))
        precip = precip_raw if precip_raw is not None and pd.notna(precip_raw) else 0
        wc_raw = h.get('weather_code', h.get('weather_code_raw', 0))
        wc = int(wc_raw) if wc_raw is not None and pd.notna(wc_raw) else 0
        icon = WMO_CODES.get(wc, {}).get('icon', 'unknown')
        emoji = h.get('weather_emoji', '')
        lines.append(
            f"  {date_str} {hour:02d}:00 {icon} {emoji}  {temp}°   {hum}%   {wind}   {press}   {cloud}%   {precip}"
        )
    hourly_text = "\n".join(lines)

    prompt = (
        f"Satni podaci za Budvu, {date_str} (sat  ikonica  temp  vlažnost  vjetar_m/s  pritisak_hPa  oblačnost  padavine_mm):\n"
        f"{hourly_text}\n\n"
        "Na osnovu ovih satnih podataka za Budvu, napiši kratak izvještaj.\n\nPravila:\n1. Maksimalno 6-7 riječi.\n2. Fokusiraj se na glavnu promjenu vremena (npr. prelaz iz oblačnog u sunčano).\n3. Navedi doba dana kada se promjena dešava.\n4.Ako nema kiše ili vjetra, ne spominji ih."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200,
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    for attempt in range(4):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40 sec
                print(f"  [Gemini] Rate limit za {date_str}, čekam {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            cand = data['candidates'][0]
            # Reject truncated output (finishReason != STOP) so a half sentence
            # never ships — the rule-based _daily_narrative stays as the fallback.
            if cand.get('finishReason') not in (None, 'STOP'):
                return None
            text = cand['content']['parts'][0]['text'].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            print(f"  [Gemini] Greška za {date_str}: {e}")
            return None
    return None


def _gemini_narrative_daily(date_str, ds):
    """Call Gemini for long-range days that lack hourly data."""
    if not GEMINI_API_KEY:
        return None
    tmin = ds.get('temp_min', '?')
    tmax = ds.get('temp_max', '?')
    cloud = ds.get('cloud_cover_day', ds.get('cloud_cover_avg', '?'))
    precip = ds.get('precip_total', 0)
    precip_text = 'nema podataka' if precip is None else f'{precip}mm'
    wind = ds.get('wind_max', '?')
    pp = ds.get('precip_probability')
    pp_text = 'nema podataka' if pp is None else f'{pp}%'
    summary = (f"Budva {date_str}: temp {tmin}-{tmax}°C, oblačnost {cloud}%, "
               f"padavine {precip_text} (šansa {pp_text}), vjetar do {wind}m/s.")
    prompt = (
        f"{summary}\n\n"
        "Napiši JEDNU rečenicu (max 10 riječi) koja opisuje vremenske uslove.\n"
        "Crnogorski jezik. Bez emotikona. Bez savjeta.\n"
        "Ako oblačnost > 50%, pomeni oblake. Ako padavine >= 0.2mm, pomeni kišu.\n"
        "Primjeri: Djelimično oblačno, moguća slaba kiša. / Pretežno vedro uz umjeren vjetar.\n"
        "Samo rečenicu:"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200,
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  [Gemini] Rate limit za {date_str}, čekam {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            cand = data['candidates'][0]
            # Reject truncated output (finishReason != STOP) so a half sentence
            # never ships — the rule-based _daily_narrative stays as the fallback.
            if cand.get('finishReason') not in (None, 'STOP'):
                return None
            text = cand['content']['parts'][0]['text'].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            print(f"  [Gemini] Greška za {date_str}: {e}")
            return None
    return None


def _enrich_narratives_with_ai(daily_list, hourly_data):
    """Replace day_narrative with AI-generated text where possible.
    Uses a date-keyed cache to avoid redundant Gemini calls."""
    if not GEMINI_API_KEY:
        print("  [Gemini] API ključ nije postavljen, preskačem AI opise.")
        return

    # --- Cache logic (only on CI/GitHub Actions to save API quota) ---
    use_cache = os.environ.get('GITHUB_ACTIONS') == 'true'
    cache_path = os.path.join(OUTPUT_DIR, "gemini_narrative_cache.json")
    cache = {}
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    print("  [Gemini] Generišem AI opise vremena...")
    hourly_by_date = {}
    for h in hourly_data:
        d = h.get('date', h.get('_date', ''))
        if d not in hourly_by_date:
            hourly_by_date[d] = []
        hourly_by_date[d].append(h)

    count = 0
    api_calls = 0
    for ds in daily_list:
        date_str = ds.get('date', '')
        rows = hourly_by_date.get(date_str, [])
        rows.sort(key=lambda r: r.get('hour', 0))
        has_hourly = len(rows) >= 8

        # Cache strategy: only use cache for long-range days (no hourly available).
        # Hourly-covered days are ALWAYS regenerated so the narrative reflects the
        # latest forecast.
        if not has_hourly and date_str in cache:
            ds['day_narrative'] = cache[date_str]
            count += 1
            continue

        if NARRATIVE_MODE == 'rephrase':
            # Guarded Gemini: for hourly days it GENERATES with the legacy hourly
            # prompt; for long-range it rephrases the rule-based sentence. Either
            # way validate() guards it and falls back to `base` on any failure.
            base = ds.get('day_narrative')
            new = gemini_narrative.daily_narrative_ai(
                base, ds, hourly_rows=rows if has_hourly else None,
                wmo_codes=WMO_CODES) if base else None
            narrative = new if (new and new != base) else None
        else:
            # FULL Gemini narrative: generate from the (hourly / daily) data. The
            # rule-based _daily_narrative output is the fallback (no key / API error).
            narrative = (_gemini_narrative(date_str, rows) if has_hourly
                         else _gemini_narrative_daily(date_str, ds))
        api_calls += 1
        if narrative:
            ds['day_narrative'] = narrative
            if not has_hourly:
                cache[date_str] = narrative
            count += 1
        if api_calls < len(daily_list):
            time.sleep(12)

    # Save cache only on CI
    if use_cache:
        today_str = local_now().strftime('%Y-%m-%d')
        cache = {k: v for k, v in cache.items() if k >= today_str}
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    print(f"  [Gemini] Mode={NARRATIVE_MODE}: {count}/{len(daily_list)} opisa "
          f"({api_calls} API poziva, {count - api_calls} iz keša).")


def generate_output(corrected, trained, results, fc_raw=None, marine=None, onset=None):
    print("\n[6/6] Generisanje izlaza...")
    now_str = local_now().isoformat()
    now_ts = local_now().floor('h')
    cutoff_48h = now_ts + pd.Timedelta(hours=48)

    hail_votes = _hail_votes_by_datetime(fc_raw)

    forecast_hours = []
    for _, row in corrected.iterrows():
        if row['datetime'] >= cutoff_48h:
            continue
        # Past hours (today's morning that already happened) stay in corrected
        # for daily aggregation but must NOT appear in the live hourly JSON.
        if row['datetime'] < now_ts:
            continue
        wc = int(row.get('weather_code', 0)) if pd.notna(row.get('weather_code', np.nan)) else 0
        wmo = WMO_CODES.get(wc, WMO_CODES[0])

        entry = {
            "datetime": row['datetime'].isoformat(),
            "hour": int(row['datetime'].hour),
            "date": row['datetime'].strftime('%Y-%m-%d'),
            "day_name": row['datetime'].strftime('%A'),
            "weather_code": wc,
            "weather_desc": wmo['desc'],
            "weather_icon": wmo['icon'],
            "weather_emoji": wmo['emoji'],
        }

        for param, info in TARGET_PARAMS.items():
            xgb_col = f'{param}_xgb'
            ens_col = f'{param}_ensemble'
            val = row.get(xgb_col, row.get(ens_col, None))
            if val is not None and pd.notna(val):
                entry[param] = round(float(val), 2)
            ens_val = row.get(ens_col, None)
            if ens_val is not None and pd.notna(ens_val):
                entry[f'{param}_raw'] = round(float(ens_val), 2)

        for extra in ['apparent_temperature', 'wind_direction_10m', 'surface_pressure',
                      'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'rain', 'snowfall']:
            v = row.get(f'{extra}_ens', None)
            if v is not None and pd.notna(v):
                entry[extra] = round(float(v), 2)

        # distributional fields (quantile bands, PoP, exceedance).
        # Skipped silently when models predate the quantile upgrade.
        for qcol in ['temperature_2m_q05', 'temperature_2m_q10', 'temperature_2m_q25',
                     'temperature_2m_q50', 'temperature_2m_q75', 'temperature_2m_q90',
                     'temperature_2m_q95',
                     'wind_speed_10m_q10', 'wind_speed_10m_q50', 'wind_speed_10m_q90',
                     'wind_gusts_10m_q10', 'wind_gusts_10m_q50', 'wind_gusts_10m_q90',
                     'precipitation_q10', 'precipitation_q50', 'precipitation_q75',
                     'precipitation_q90', 'precipitation_q95']:
            v = row.get(qcol, None)
            if v is not None and pd.notna(v):
                entry[qcol] = round(float(v), 2)
        for pcol in ['precipitation_pop', 'p_precip_gt1', 'p_precip_gt5',
                      'p_gust_gt10', 'p_gust_gt17',
                     'rain_signal_confidence', 'rain_onset_hazard',
                     'rain_onset_probability_by_then', 'rain_onset_threshold']:
            v = row.get(pcol, None)
            if v is not None and pd.notna(v):
                entry[pcol] = round(float(v), 3)

        trusted_amount = row.get('italiameteo_precipitation', None)
        if trusted_amount is not None and pd.notna(trusted_amount):
            entry['italiameteo_precipitation'] = round(float(trusted_amount), 2)
        for signal_col in [
                'italiameteo_rain_signal', 'italiameteo_rain_accepted',
                'rain_signal_convective_rescue', 'rain_signal_skala_support',
                'rain_signal', 'rain_onset_eligible', 'rain_onset_signal',
                'rain_onset_skala_support']:
            signal = row.get(signal_col, None)
            if signal is not None and pd.notna(signal):
                entry[signal_col] = bool(signal)
        if entry.get('rain_signal'):
            sources = ['XGB']
            if entry.get('italiameteo_rain_accepted'):
                sources.insert(0, 'ITALIAMETEO_ICON2I')
            if entry.get('rain_signal_convective_rescue'):
                sources.append('CONVECTIVE_DIAGNOSTICS')
            if entry.get('rain_signal_skala_support'):
                sources.append('SKALA')
            entry['rain_signal_source'] = '+'.join(sources)
        onset_source = row.get('rain_onset_source', None)
        if onset_source and entry.get('rain_onset_signal'):
            entry['rain_onset_source'] = str(onset_source)

        hv = hail_votes.get(row['datetime'])
        if hv:
            entry['hail_models'] = hv

        forecast_hours.append(entry)

    all_data = corrected.copy()
    all_data['_date'] = all_data['datetime'].dt.strftime('%Y-%m-%d')
    all_data['_day_name'] = all_data['datetime'].dt.strftime('%A')
    all_data['hour'] = all_data['datetime'].dt.hour

    today_str = local_now().strftime('%Y-%m-%d')
    today_cache_path = os.path.join(OUTPUT_DIR, "today_daily_cache.json")

    all_daily = {}  # date_str -> summary dict
    for date_str, grp in all_data.groupby('_date'):
        day_name = grp.iloc[0]['_day_name']

        if date_str == today_str:
            first_hour = int(grp['hour'].min())
            if first_hour < 10:
                ds = _build_daily_summary(date_str, day_name, grp, fc_raw=fc_raw)
                all_daily[date_str] = ds
                try:
                    with open(today_cache_path, 'w', encoding='utf-8') as _cf:
                        json.dump(ds, _cf, ensure_ascii=False)
                except Exception:
                    pass
            else:
                cached = None
                if os.path.exists(today_cache_path):
                    try:
                        with open(today_cache_path, 'r', encoding='utf-8') as _cf:
                            cached = json.load(_cf)
                        if cached.get('date') != today_str:
                            cached = None
                    except Exception:
                        cached = None
                if cached:
                    all_daily[date_str] = cached
                else:
                    all_daily[date_str] = _build_daily_summary(
                        date_str, day_name, grp, fc_raw=fc_raw
                    )
        else:
            all_daily[date_str] = _build_daily_summary(
                date_str, day_name, grp, fc_raw=fc_raw
            )

    dates_48h = set()
    if len(forecast_hours) > 0:
        fc_df = pd.DataFrame(forecast_hours)
        dates_48h = set(fc_df['date'].unique())

    daily = []
    long_range = []
    for date_str in sorted(all_daily.keys()):
        ds = all_daily[date_str]
        if date_str in dates_48h:
            daily.append(ds)
        else:
            long_range.append(ds)

    # Long-range days: floor the smoothed daily wind peaks with the model
    # consensus so the cards stop labeling every day "slab". Within 48h the
    # calibrated XGB wind stays authoritative; the helper fields never ship.
    for ds in long_range:
        if ds.get('wind_max_models') is not None:
            ds['wind_max'] = round(max(ds.get('wind_max') or 0.0,
                                       ds['wind_max_models']), 1)
        if ds.get('gust_max_models') is not None:
            ds['gust_max'] = round(max(ds.get('gust_max') or 0.0,
                                       ds['gust_max_models']), 1)
    for ds in daily + long_range:
        ds.pop('wind_max_models', None)
        ds.pop('gust_max_models', None)

    if long_range:
        print(f"  Long range: {len(long_range)} dana")

    # --- ECC scenarios — temporally coherent precip trajectories ---
    # Quantile marginals per hour have no hour-to-hour correlation; reordering
    # samples by the raw-ensemble rank structure restores realistic episode
    # durations, from which we get window PoPs + episode stats.
    precip_scenarios = None
    try:
        qcols = ['precipitation_q05', 'precipitation_q10', 'precipitation_q25',
                 'precipitation_q50', 'precipitation_q75', 'precipitation_q90',
                 'precipitation_q95']
        if fc_raw is not None and all(c in corrected.columns for c in qcols):
            sub = corrected[(corrected['datetime'] >= now_ts)
                            & (corrected['datetime'] < cutoff_48h)].copy()
            member_cols = [f"{m}_precipitation_model" for m in MODELS
                           if f"{m}_precipitation_model" in fc_raw.columns]
            if len(member_cols) >= 4 and len(sub) >= 12:
                raw = fc_raw[['datetime'] + member_cols].copy()
                for c in member_cols:
                    # align raw end-of-hour labels with the shifted corrected rows
                    raw[c] = pd.to_numeric(raw[c], errors='coerce').shift(-1)
                merged_sc = sub[['datetime'] + qcols].merge(raw, on='datetime', how='left')
                qdf_sc = merged_sc[qcols].rename(
                    columns={c: c.replace('precipitation_', '') for c in qcols})
                scen = pf.ecc_scenarios(qdf_sc, pf.DEFAULT_ALPHAS,
                                        merged_sc[member_cols].values)
                precip_scenarios = pf.rain_episode_stats(
                    scen, wet_thresh=0.1, windows=((0, 12), (12, 24), (24, 48)))
                precip_scenarios = {k: round(float(v), 3)
                                    for k, v in precip_scenarios.items()}
                print(f"  ECC scenariji: {precip_scenarios}")
    except Exception as _e:
        print(f"  ECC scenariji preskočeni ({_e})")

    # Enrich narratives with Gemini AI (daily + long_range, uses ALL hourly data)
    all_days_for_ai = daily + long_range
    all_hourly = all_data.to_dict('records')
    _enrich_narratives_with_ai(all_days_for_ai, all_hourly)

    output = {
        "generated": now_str,
        "location": {"name": "Budva, Crna Gora", "lat": LAT, "lon": LON,
                      "timezone": FORECAST_TIMEZONE,
                      "station": "ibudva5 (Weather Underground)"},
        "method": "XGBoost Multi-Model Ensemble + Historical Bias + Forecast Revision v3",
        "description": f"{len(MODELS)} modela, 6 godina podataka (2020-2026), pametna korekcija + Day1/Day2 revizije",
        "models": MODELS,
        "training_metrics": results,
        "daily_summary": daily,
        "hourly_forecast": forecast_hours,
        "long_range": long_range,
    }

    if marine is not None:
        output["marine_forecast"] = marine
    if onset is not None:
        output["rain_onset"] = onset
    if precip_scenarios is not None:
        output["precip_scenarios"] = precip_scenarios

    json_path = os.path.join(OUTPUT_DIR, "forecast_48h.json")
    json_tmp = json_path + '.tmp'
    try:
        with open(json_tmp, 'w', encoding='utf-8') as f:
            # RFC 8259 JSON does not permit NaN/Infinity. Validate while writing
            # a temporary file so a failure cannot truncate the prior forecast.
            json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(json_tmp, json_path)
    finally:
        _remove_if_exists(json_tmp)
    csv_path = os.path.join(OUTPUT_DIR, "forecast_48h.csv")
    corrected.to_csv(csv_path, index=False)

    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    print("\n" + "=" * 72)
    print("  KORIGOVANA PROGNOZA ZA BUDVU --- Naredna 48 sata")
    print("=" * 72)
    for d in daily:
        em = d.get('weather_emoji', '')
        print(f"\n  {em} {d['day_name']} {d['date']}")
        print(f"     Temp: {d.get('temp_min', '?')}° — {d.get('temp_max', '?')}°C  |  Vlaznost: {d.get('humidity_avg', '?')}%")
        print(f"     Vjetar: do {d.get('wind_max', '?')} m/s (udari {d.get('gust_max', '?')} m/s)")
        _daily_precip = d.get('precip_total')
        _daily_precip_text = f"{_daily_precip} mm" if _daily_precip is not None else "N/A"
        print(f"     Padavine: {_daily_precip_text}  |  Pritisak: {d.get('pressure_avg', '?')} hPa")
        print(f"     {d.get('day_narrative', d.get('weather_desc', ''))}")

    print("\n  " + "-" * 68)
    print(f"  {'Sat':>12}  {'Temp':>6}  {'Vlaz':>5}  {'Vjet':>5}  {'Prit':>6}  {'Obl':>4}  {'Kisa':>6}  Opis")
    print(f"  {'':>12}  {'':>6}  {'':>5}  {'':>5}  {'':>6}  {'':>4}  {'':>6}")
    for _, r in corrected.iterrows():
        ts = r['datetime'].strftime('%d.%m %H:%M')
        t = r.get('temperature_2m_xgb', r.get('temperature_2m_ensemble', float('nan')))
        h = r.get('relative_humidity_2m_xgb', r.get('relative_humidity_2m_ensemble', float('nan')))
        w = r.get('wind_speed_10m_xgb', r.get('wind_speed_10m_ensemble', float('nan')))
        p = r.get('pressure_msl_xgb', r.get('pressure_msl_ensemble', float('nan')))
        c = r.get('cloud_cover_xgb', r.get('cloud_cover_ensemble', float('nan')))
        pr = r.get('precipitation_xgb', r.get('precipitation_ensemble', float('nan')))
        wc = int(r.get('weather_code', 0)) if pd.notna(r.get('weather_code', np.nan)) else 0
        em = WMO_CODES.get(wc, WMO_CODES[0])['emoji']

        tf = f"{t:5.1f}°" if pd.notna(t) else "  N/A"
        hf = f"{h:3.0f}%" if pd.notna(h) else " N/A"
        wf = f"{w:4.1f}" if pd.notna(w) else " N/A"
        pf_str = f"{p:5.0f}" if pd.notna(p) else "  N/A"  # not 'pf' — that shadows the prob_forecast module
        cf = f"{c:3.0f}%" if pd.notna(c) else " N/A"
        rf = f"{pr:5.2f}" if pd.notna(pr) else "  N/A"

        print(f"  {ts:>12}  {tf}  {hf}  {wf}  {pf_str}  {cf}  {rf}  {em}")

    return json_path, csv_path


if __name__ == "__main__":
    _allowed_args = {
        '--gpu', '--cpu', '--check-device', '--check_device',
        '--skip-training', '--skip_training', '--dry-now', '--dry_now',
        '--aux-diagnostics',
    }
    _backend_args = [arg for arg in sys.argv[1:]
                     if arg.startswith('--check-backend=')]
    _unknown_args = [arg for arg in sys.argv[1:]
                     if arg not in _allowed_args
                     and not arg.startswith('--check-backend=')]
    if _unknown_args:
        raise ValueError(f"Nepoznati argumenti: {_unknown_args}")
    if len(_backend_args) > 1:
        raise ValueError('Dozvoljen je samo jedan --check-backend argument')

    def _exercise_backend(backend):
        probe_rng = np.random.default_rng(42)
        probe_X = probe_rng.normal(size=(512, 8)).astype(np.float32)
        probe_y = (probe_X[:, 0] - 0.25 * probe_X[:, 1]).astype(np.float32)
        if backend == 'xgboost':
            model = _new_xgb_regressor(
                n_estimators=1, max_depth=2, objective='reg:squarederror',
                random_state=42, verbosity=0,
            )
            model.fit(probe_X, probe_y)
            model.predict(probe_X[:4])
        elif backend == 'catboost':
            model = _new_catboost_regressor(
                iterations=3, depth=3, learning_rate=0.1,
                loss_function='MAE', bootstrap_type='Bernoulli',
                subsample=0.8, random_seed=42, verbose=0,
            )
            model.fit(probe_X, probe_y)
            _catboost_predict(model, probe_X[:4])
        elif backend == 'lightgbm':
            model = _new_lgbm_regressor(
                n_estimators=3, max_depth=3, max_bin=63,
                random_state=42, verbose=-1,
            )
            model.fit(probe_X, probe_y)
            model.predict(probe_X[:4])
        else:
            raise ValueError(f'Nepoznat backend za provjeru: {backend}')

    if _backend_args:
        _backend = _backend_args[0].split('=', 1)[1].strip().lower()
        _exercise_backend(_backend)
        print(f"  BACKEND CHECK OK: {_backend} ({ML_DEVICE})")
        sys.exit(0)

    if '--check-device' in sys.argv or '--check_device' in sys.argv:
        # Windows GPU runtimes can conflict during interpreter teardown when
        # CUDA CatBoost and OpenCL LightGBM live in the same process. Probe each
        # backend in an isolated child and require a clean process exit, not
        # merely a successful fit/predict before a late native crash.
        _device_flag = '--gpu' if USING_GPU else '--cpu'
        sys.stdout.flush()
        for _backend in ('xgboost', 'catboost', 'lightgbm'):
            _completed = subprocess.run(
                [sys.executable, os.path.abspath(__file__), _device_flag,
                 f'--check-backend={_backend}'],
                check=False,
            )
            if _completed.returncode != 0:
                raise RuntimeError(
                    f'{_backend} device probe exited with '
                    f'code {_completed.returncode}'
                )

        print("\n  DEVICE CHECK OK")
        print(f"  XGBoost:  {ML_DEVICE} (fit + predict passed)")
        print(f"  CatBoost: {CATBOOST_DEVICE_PARAMS['task_type']} (fit + predict passed)")
        print(f"  LightGBM: {LIGHTGBM_DEVICE_PARAMS['device_type']} (fit + predict passed)")
        sys.exit(0)

    skip_training = '--skip-training' in sys.argv or '--skip_training' in sys.argv
    local_dry_nowcast = '--dry-now' in sys.argv or '--dry_now' in sys.argv

    if skip_training:
        print("\n  MODE: --skip-training (ucitavam sacuvane modele)")
        trained, results, bias_tables = load_trained_models()
        onset_bundle = load_onset_model()
    else:
        hist = load_historical_data()

        print("\n[2/6] Feature engineering + tabele biasa...")
        bias_tables = compute_bias_tables(hist)
        # lead-36/60 frames built from the RAW frame + train-only
        # bias tables, BEFORE the base frame is engineered below.
        lead_frames = build_lead_stacked_frames(hist, bias_tables)
        hist = apply_bias_features(hist, bias_tables)
        hist['lead_time'] = 12.0  # day-0 archive rows ~ short-lead splice
        hist = engineer_features(hist)
        print(f"  Dimenzije: {hist.shape[0]} x {hist.shape[1]}")

        bias_path = os.path.join(MODEL_DIR, 'bias_tables.json')
        bt_serializable = {}
        for k, v in bias_tables.items():
            bt_serializable[k] = v.to_dict(orient='records')
        _write_json_atomic(bias_path, bt_serializable)
        print(f"  Bias tabele: {bias_path}")

        trained, results = train_all_models(hist, lead_frames=lead_frames)
        try:
            onset_bundle = train_onset_model(hist)
        except Exception as _e:
            _invalidate_onset_artifacts()
            if _DEVICE_REQUEST == 'cuda':
                raise
            print(f"  [Onset] trening preskočen ({_e})")
            onset_bundle = None

    try:
        fc_all = fetch_live_forecasts()
        station_obs = fetch_current_station_observation()
        station_dry_nowcast = station_says_dry_now(station_obs)
        if station_obs:
            rate = station_obs.get("precip_rate_mm")
            age = station_obs.get("age_min")
            age_txt = f"{age:.0f}" if age is not None else "?"
            print(f"\n  WU nowcast: {station_obs.get('station')} rate={rate} mm/h, age={age_txt} min")

        dry_nowcast_active = local_dry_nowcast or station_dry_nowcast
        if local_dry_nowcast:
            print("  NOWCAST: lokalno suvo sada (--dry-now), gasim slabu blisku kisu bez sire podrske")
        elif station_dry_nowcast:
            print("  NOWCAST: WU stanica javlja suvo sada, gasim slabu blisku kisu bez sire podrske")
        elif not WU_API_KEY:
            print("\n  WU nowcast: WU_API_KEY nije postavljen, preskacem automatsku live-stanicu")

        corrected = apply_correction(fc_all, trained, bias_tables, local_dry_nowcast=dry_nowcast_active)

        # Rain-onset timing (report Part B) — NEW output, doesn't touch the
        # existing hourly/precip display. Best-effort: never breaks the forecast.
        onset_info = None
        try:
            _fc_onset = apply_bias_features(fc_all.copy(), bias_tables)
            _onset_now = local_now().floor('h')
            _fc_onset['lead_time'] = (
                (pd.to_datetime(_fc_onset['datetime']) - _onset_now)
                .dt.total_seconds() / 3600.0
            ).clip(lower=0.0, upper=72.0)
            _fc_feat = _align_onset_features_to_display(
                engineer_features(_fc_onset)
            )
            if 'precipitation_xgb' in corrected.columns:
                _state_by_time = corrected.set_index('datetime')['precipitation_xgb']
                _fc_feat['_onset_state_precip'] = pd.to_datetime(
                    _fc_feat['datetime']
                ).map(_state_by_time)
            onset_info = predict_onset_timing(_fc_feat, onset_bundle)
            corrected = attach_hourly_onset_signal(corrected, onset_info)
            if onset_info and onset_info.get('declared'):
                if onset_info.get('already_raining'):
                    print("  [Onset] kiša već pada na početku horizonta.")
                else:
                    print(f"  [Onset] kiša počinje ~{onset_info.get('likely_datetime')} "
                          f"(najranije {onset_info.get('earliest_datetime')}, "
                          f"najkasnije {onset_info.get('latest_datetime')}).")
        except Exception as _e:
            if _DEVICE_REQUEST == 'cuda':
                raise
            print(f"  [Onset] preskočen ({_e})")
            onset_info = None
    except TrustedRainGateError as e:
        print("\n" + "!" * 72)
        print(f"  TRUSTED RAIN GATE FAIL: {e}")
        print("!" * 72)
        prev_json = os.path.join(OUTPUT_DIR, "forecast_48h.json")
        prev_csv = os.path.join(OUTPUT_DIR, "forecast_48h.csv")
        if os.path.exists(prev_json):
            mtime = os.path.getmtime(prev_json)
            age_str = pd.Timestamp.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            print(f"\n  ZADRZAVAM prethodnu prognozu: {prev_json}")
            print(f"  (generisana: {age_str})")
            print("  Nova prognoza NIJE upisana - prethodni run ostaje na snazi.")
            print("=" * 72)
            sys.exit(0)
        else:
            print(f"\n  Prethodna prognoza ne postoji ({prev_json}).")
            print("  Ne postoji fallback - prekidam bez upisa.")
            print("=" * 72)
            sys.exit(1)

    marine_block = None
    try:
        marine_df = fetch_live_marine()
        if marine_df is not None:
            marine_block = build_marine_output(marine_df)
    except Exception as e:
        print(f"  Marine: greska, preskacem ({e})")

    json_path, csv_path = generate_output(corrected, trained, results, fc_raw=fc_all,
                                          marine=marine_block, onset=onset_info)

    print("\n" + "=" * 72)
    print("  GOTOVO! Fajlovi:", OUTPUT_DIR)
    print("=" * 72)
