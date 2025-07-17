"""
SARIMA helpers with minimal verbosity.
"""
import itertools, numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _grid_search(train_ts: pd.Series, cfg: dict):
    p, q      = cfg["p_values"], cfg["q_values"]
    P, Q      = cfg["P_values"], cfg["Q_values"]
    s         = cfg["seasonal_period"]
    combos    = len(p)*len(q)*len(P)*len(Q)
    print(f"    Exploring {combos} seasonal-order combos …")

    best_aic, best_order = np.inf, None
    for p_, q_, P_, Q_ in itertools.product(p, q, P, Q):
        try:
            mdl = SARIMAX(train_ts,
                          order=(p_, 0, q_),
                          seasonal_order=(P_, 0, Q_, s),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            if mdl.aic < best_aic:
                best_aic, best_order = mdl.aic, (p_, q_, P_, Q_)
        except Exception:
            continue
    return best_order, best_aic


def train_sarima(train_df, test_df, sar_cfg, time_col: str):
    train_ts = train_df.set_index(time_col)["y"]
    best_order, best_aic = _grid_search(train_ts, sar_cfg)
    p, q, P, Q = best_order
    print(f"    Best order found: (p,q,P,Q) = {best_order}  AIC {best_aic:.1f}")

    mdl = SARIMAX(train_ts,
                  order=(p, 0, q),
                  seasonal_order=(P, 0, Q, sar_cfg["seasonal_period"]),
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)

    preds = mdl.predict(start=test_df[time_col].iloc[0],
                        end=test_df[time_col].iloc[-1])
    return preds, {"order": best_order, "aic": float(best_aic)}