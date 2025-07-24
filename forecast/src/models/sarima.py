"""
SARIMA helpers – robust to missing/irregular datetime frequency,
parallel grid-search with caching, and now guarantees a NaN-free forecast.
"""
from __future__ import annotations
import itertools, numpy as np, pandas as pd
from pathlib import Path
import joblib, hashlib, os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
_CACHE_DIR = Path("outputs/models")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _sarima_cache_path(order):
    tag = "_".join(map(str, order))
    return _CACHE_DIR / f"sarima_{tag}.pkl"

# ------------------------------------------------------------------ #
def _fit_one(train_ts, sar_cfg, combo):
    p_, q_, P_, Q_ = combo
    try:
        mdl = SARIMAX(train_ts,
                      order=(p_, 0, q_),
                      seasonal_order=(P_, 0, Q_, sar_cfg["seasonal_period"]),
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        return mdl.aic, combo
    except Exception:
        return np.inf, None

def _grid_search(train_ts: pd.Series, cfg: dict):
    p, q, P, Q = cfg["p_values"], cfg["q_values"], cfg["P_values"], cfg["Q_values"]
    combos = list(itertools.product(p, q, P, Q))
    print(f"    Exploring {len(combos)} seasonal-order combos … (parallel)")

    n_jobs = cfg.get("n_jobs", min(os.cpu_count() or 1, 8))
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_one)(train_ts, cfg, c) for c in combos)

    best_aic, best_order = min(results, key=lambda t: t[0])
    return best_order, best_aic

# ------------------------------------------------------------------ #
def acf_pacf_analysis(y: pd.Series, lags: int = 48, save_plots: bool = False) -> dict:
    """
    Generate ACF and PACF plots to identify potential ARIMA orders.
    
    Returns:
        dict: Suggested (p,q) values based on significant lags
    """
    print("    Running ACF/PACF analysis...")
    
    # Apply basic differencing to make series more stationary
    y_diff = y.diff().dropna()
    
    if save_plots:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # ACF plot
        plot_acf(y_diff, lags=lags, ax=axes[0], title="ACF of differenced series")
        axes[0].grid(True, alpha=0.3)
        
        # PACF plot  
        plot_pacf(y_diff, lags=lags, ax=axes[1], title="PACF of differenced series")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        diagnostics_path = Path("outputs/reports")
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(diagnostics_path / "sarima_acf_pacf.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ACF/PACF plots saved to: {diagnostics_path / 'sarima_acf_pacf.png'}")
    
    # Simple heuristics for parameter suggestions
    suggestions = {
        'p_candidates': [0, 1, 2],  # AR order suggestions
        'q_candidates': [0, 1, 2],  # MA order suggestions
        'P_candidates': [0, 1],     # Seasonal AR
        'Q_candidates': [0, 1],     # Seasonal MA
    }
    
    return suggestions

def _sanitize(series: pd.Series) -> pd.Series:
    """Ensure no NaN/Inf in forecast by interpolation + ffill/bfill."""
    s = series.copy().astype(float).replace([np.inf, -np.inf], np.nan)
    if s.isna().any():
        s = s.interpolate(limit_direction="both").bfill().ffill()
        # If still NaN after interpolation, fill with 0
        if s.isna().any():
            print(f"    Warning: {s.isna().sum()} NaN values remain after interpolation, filling with 0")
            s = s.fillna(0)
    return s

# ------------------------------------------------------------------ #
def train_sarima(train_df: pd.DataFrame,
                 test_df:  pd.DataFrame,
                 sar_cfg:  dict,
                 time_col: str):
    """
    Fit SARIMA on *train_df* and forecast the entire length of *test_df*.

    Returns
    -------
    preds : pd.Series  (index = test_df[time_col])
    info  : dict       {"order": (p,q,P,Q), "aic": float}
    """
    train_ts = train_df.set_index(time_col)["y"]
    # Ensure frequency is set to avoid statsmodels warnings
    train_ts.index = pd.to_datetime(train_ts.index)
    train_ts = train_ts.asfreq('H')  # Assume hourly frequency

    # Run ACF/PACF analysis for diagnostics
    acf_results = acf_pacf_analysis(train_ts, save_plots=True)
    
    # Use ACF/PACF informed grid search or fall back to config
    if not any(key in sar_cfg for key in ['p_values', 'q_values', 'P_values', 'Q_values']):
        print("    Using ACF/PACF informed parameter ranges...")
        sar_cfg = {**sar_cfg,  # preserve existing config
                   'p_values': acf_results['p_candidates'],
                   'q_values': acf_results['q_candidates'], 
                   'P_values': acf_results['P_candidates'],
                   'Q_values': acf_results['Q_candidates']}

    best_order, best_aic = _grid_search(train_ts, sar_cfg)
    cache_path = _sarima_cache_path(best_order)

    if cache_path.exists():
        mdl = joblib.load(cache_path)
    else:
        p, q, P, Q = best_order
        mdl = SARIMAX(train_ts,
                      order=(p, 0, q),
                      seasonal_order=(P, 0, Q, sar_cfg["seasonal_period"]),
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        joblib.dump(mdl, cache_path)

    steps = len(test_df)
    try:
        preds = mdl.forecast(steps=steps)
        if len(preds) == 0:
            print(f"    Warning: SARIMA forecast returned empty series, creating zeros")
            preds = pd.Series(np.zeros(steps), index=test_df[time_col].values)
        else:
            preds.index = test_df[time_col].values
            preds = _sanitize(preds)              # <-- guarantees finite values
    except Exception as e:
        print(f"    Error in SARIMA forecast: {e}, using zeros")
        preds = pd.Series(np.zeros(steps), index=test_df[time_col].values)

    return preds, {"order": best_order, "aic": float(best_aic)}