# forecast/src/models/gbdt.py
"""
Gradient-Boosting helpers

Exports
-------
build_feature_sets   : search Manual / BSFS / FwdSFS / BayesOpt feature subsets
train_gbdt           : fit one GBDT model and return predictions + model object
_cv_mae_folds        : internal – per-fold MAE list (time-series CV)
"""
from __future__ import annotations
import numpy as np, pandas as pd, hashlib, joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector

# ------------------------------------------------------------------#
_CACHE_DIR = Path("outputs/models")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _gbt_cache_path(label: str, feats: list[str]) -> Path:
    h = hashlib.md5(",".join(sorted(feats)).encode()).hexdigest()[:8]
    return _CACHE_DIR / f"gbt_{label}_{h}.pkl"


# ------------------------------------------------------------------#
def _cv_mae_folds(X: pd.DataFrame, y: np.ndarray, cv_cfg: dict, rs: int) -> list[float]:
    tscv = TimeSeriesSplit(n_splits=cv_cfg["n_splits"], test_size=cv_cfg["horizon_hours"])
    out = []
    for tr, va in tscv.split(X):
        m = GradientBoostingRegressor(random_state=rs)
        m.fit(X.iloc[tr], y[tr])
        out.append(mean_absolute_error(y[va], m.predict(X.iloc[va])))
    return out


# ------------------------------------------------------------------#
def build_feature_sets(
    manual_feats,
    rich_df,
    n_keep,
    forward_pool,
    bo_k_max,
    cv_cfg,
    rs,
):
    """
    Returns
    -------
    sets_dict   : label → feature list
    mean_scores : label → mean CV MAE
    gain_series : Series (global gain ranking)
    bo_trace    : list[(k, mean-CV-MAE)]
    cv_scores   : label → list[fold MAE]
    """
    sets, mean_scores, cv_scores = {}, {}, {}

    # ---- global gain ranking -------------------------------------
    full_feats  = [c for c in rich_df.columns if c not in ("time", "y")]
    rank_model  = GradientBoostingRegressor(random_state=rs).fit(
        rich_df[full_feats],
        rich_df["y"].values,
    )
    gain_series = pd.Series(rank_model.feature_importances_, index=full_feats).sort_values(ascending=False)
    topN = gain_series.head(n_keep).index.tolist()

    # === 1. Manual =================================================
    sets["Manual"] = manual_feats
    folds = _cv_mae_folds(rich_df[manual_feats], rich_df["y"].values, cv_cfg, rs)
    mean_scores["Manual"] = float(np.mean(folds))
    cv_scores["Manual"]   = folds

    # === 2. Backward SFS ==========================================
    print("      Backward SFS …")
    sfs_back = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=rs),
        direction="backward",
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=cv_cfg["n_splits"], test_size=cv_cfg["horizon_hours"]),
        tol=1e-3,
        n_jobs=-1,
    )
    sfs_back.fit(rich_df[topN], rich_df["y"].values)
    bsfs_feats = [str(f) for f in np.array(topN)[sfs_back.get_support()]]
    sets["BSFS"] = bsfs_feats
    folds = _cv_mae_folds(rich_df[bsfs_feats], rich_df["y"].values, cv_cfg, rs)
    mean_scores["BSFS"] = float(np.mean(folds))
    cv_scores["BSFS"]   = folds

    # === 3. Forward SFS ===========================================
    seed = gain_series.head(forward_pool).index.tolist()
    print(f"      Forward SFS seed → {seed}")
    sfs_fwd = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=rs),
        direction="forward",
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=cv_cfg["n_splits"], test_size=cv_cfg["horizon_hours"]),
        tol=1e-3,
        n_jobs=-1,
    )
    sfs_fwd.fit(rich_df[topN], rich_df["y"].values)
    fwd_feats = [str(f) for f in np.array(topN)[sfs_fwd.get_support()]]
    sets["FwdSFS"] = fwd_feats
    folds = _cv_mae_folds(rich_df[fwd_feats], rich_df["y"].values, cv_cfg, rs)
    mean_scores["FwdSFS"] = float(np.mean(folds))
    cv_scores["FwdSFS"]   = folds

    # === 4. Bayesian search over k ================================
    print(f"      k-search (3 … {bo_k_max}) on top-gain list …")
    bo_trace, best_k, best_mae, best_cv = [], None, np.inf, None
    k_upper = min(bo_k_max, len(gain_series))
    for k in range(3, k_upper + 1):
        feats = gain_series.head(k).index.tolist()
        folds = _cv_mae_folds(rich_df[feats], rich_df["y"].values, cv_cfg, rs)
        mae_k = float(np.mean(folds))
        bo_trace.append((int(k), float(mae_k)))
        if mae_k < best_mae:
            best_mae, best_k, best_cv = mae_k, k, folds
    bo_feats = gain_series.head(best_k).index.tolist()
    sets["BayesOpt"]     = bo_feats
    mean_scores["BayesOpt"] = best_mae
    cv_scores["BayesOpt"]   = best_cv
    print(f"        ↳ best k = {best_k}  MAE {best_mae:.4f}")

    # ---- ranking preview -----------------------------------------
    print("\n      Top-10 gain features:")
    print(gain_series.head(10).to_string())
    print("\n      Bottom-10 gain features:")
    print(gain_series.tail(10).to_string(), "\n")

    return sets, mean_scores, gain_series, bo_trace, cv_scores


# ------------------------------------------------------------------#
def train_gbdt(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feats:    list[str],
    rs:       int,
    label:    str | None = None,
):
    """
    Fit-or-load a GradientBoostingRegressor for *label*.
    """
    label = label or "model"
    cache = _gbt_cache_path(label, feats)

    if cache.exists():
        model = joblib.load(cache)
    else:
        X_tr, y_tr = train_df[feats], train_df["y"].values
        model = GradientBoostingRegressor(random_state=rs).fit(X_tr, y_tr)
        joblib.dump(model, cache)

    preds = model.predict(test_df[feats])
    return preds, model