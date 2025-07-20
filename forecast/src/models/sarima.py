# forecast/src/models/sarima.py
"""
SARIMA helpers — robust to missing / irregular datetime frequency.

Key points
----------
* Grid-search over (p, q, P, Q) with joblib parallelism.
* Cached fitted model keyed by the seasonal-order tuple.
* Forecasts by *number of steps*, then aligns index with test-set timestamps.
"""
from __future__ import annotations
import itertools, numpy as np, pandas as pd, hashlib, os
from pathlib import Path
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------------------------------------------#
_CACHE_DIR = Path("outputs/models")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _sarima_cache_path(order) -> Path:
    h = hashlib.md5(str(order).encode()).hexdigest()[:8]
    return _CACHE_DIR / f"sarima_{order}_{h}.pkl"


# ------------------------------------------------------------------#
def _fit_one(train_ts, sar_cfg, combo):
    p_, q_, P_, Q_ = combo
    try:
        mdl = SARIMAX(
            train_ts,
            order=(p_, 0, q_),
            seasonal_order=(P_, 0, Q_, sar_cfg["seasonal_period"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return mdl.aic, combo
    except Exception:
        return np.inf, None


def _grid_search(train_ts: pd.Series, cfg: dict):
    p, q, P, Q = cfg["p_values"], cfg["q_values"], cfg["P_values"], cfg["Q_values"]
    combos = list(itertools.product(p, q, P, Q))
    print(f"    Exploring {len(combos)} seasonal-order combos … (parallel)")

    n_jobs = cfg.get("n_jobs", os.cpu_count() or 1)
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_one)(train_ts, cfg, c) for c in combos
    )
    best_aic, best_order = min(results, key=lambda t: t[0])
    return best_order, best_aic


# ------------------------------------------------------------------#
def train_sarima(train_df, test_df, sar_cfg, time_col: str):
    """
    Fit SARIMA on *train_df* and forecast the entire length of *test_df*.

    Returns
    -------
    preds : pd.Series  (index = test_df[time_col])
    info  : dict       {"order": (p,q,P,Q), "aic": float}
    """
    train_ts = train_df.set_index(time_col)["y"]

    best_order, best_aic = _grid_search(train_ts, sar_cfg)
    cache_path = _sarima_cache_path(best_order)

    if cache_path.exists():
        mdl = joblib.load(cache_path)
    else:
        p, q, P, Q = best_order
        mdl = SARIMAX(
            train_ts,
            order=(p, 0, q),
            seasonal_order=(P, 0, Q, sar_cfg["seasonal_period"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        joblib.dump(mdl, cache_path)

    steps = len(test_df)
    preds = pd.Series(
        mdl.forecast(steps=steps),
        index=test_df[time_col].values,
        name="y_pred",
    )
    return preds, {"order": best_order, "aic": float(best_aic)}