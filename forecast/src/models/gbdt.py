# forecast/src/models/gbdt.py
"""
Gradient-Boosting helpers — delivers four sets:
  • Manual      (fixed)
  • BSFS        (backward SFS on Top-N)
  • FwdSFS      (forward SFS seeded with top-K)
  • BayesOpt    (grid search over subset size k)
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector

# ------------------------------------------------------------------ #
def _cv_mae(X, y, cv_cfg, rs):
    tscv = TimeSeriesSplit(n_splits=cv_cfg["n_splits"],
                           test_size=cv_cfg["horizon_hours"])
    out = []
    for tr, va in tscv.split(X):
        m = GradientBoostingRegressor(random_state=rs)
        m.fit(X.iloc[tr], y[tr])
        out.append(mean_absolute_error(y[va], m.predict(X.iloc[va])))
    return float(np.mean(out))

# ------------------------------------------------------------------ #
def build_feature_sets(manual_feats,
                       rich_df,
                       n_keep,
                       forward_pool,
                       bo_k_max,
                       cv_cfg,
                       rs):
    """
    Returns:
        sets_dict    label -> feature list
        scores_dict  label -> mean CV MAE
        gain_series  Series (all engineered → gain importance)
        bo_trace     list of (k, CV MAE) tuples
    """
    sets, scores = {}, {}

    # --- global gain ranking --------------------------------------
    full_feats  = [c for c in rich_df.columns if c not in ("time", "y")]
    rank_model  = GradientBoostingRegressor(random_state=rs).fit(
                     rich_df[full_feats], rich_df["y"].values)
    gain_series = pd.Series(rank_model.feature_importances_,
                            index=full_feats).sort_values(ascending=False)
    topN = gain_series.head(n_keep).index.tolist()

    # === 1. Manual set ============================================
    sets["Manual"] = manual_feats
    scores["Manual"] = _cv_mae(rich_df[manual_feats], rich_df["y"].values,
                                 cv_cfg, rs)

    # === 2. Back-SFS ==============================================
    print("      Backward SFS …")
    sfs_back = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=rs),
        direction="backward",
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=cv_cfg["n_splits"],
                           test_size=cv_cfg["horizon_hours"]),
        tol=1e-3,
        n_jobs=-1
    )
    sfs_back.fit(rich_df[topN], rich_df["y"].values)
    bsfs_feats = list(np.array(topN)[sfs_back.get_support()])
    sets["BSFS"]  = bsfs_feats
    scores["BSFS"] = _cv_mae(rich_df[bsfs_feats], rich_df["y"].values, cv_cfg, rs)

    # === 3. Forward-SFS ===========================================
    seed = gain_series.head(forward_pool).index.tolist()
    print(f"      Forward SFS seed → {seed}")
    sfs_fwd = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=rs),
        direction="forward",
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=cv_cfg["n_splits"],
                           test_size=cv_cfg["horizon_hours"]),
        tol=1e-3,
        n_jobs=-1)
    sfs_fwd.fit(rich_df[topN], rich_df["y"].values)
    fwd_feats = list(np.array(topN)[sfs_fwd.get_support()])
    sets["FwdSFS"]  = fwd_feats
    scores["FwdSFS"] = _cv_mae(rich_df[fwd_feats], rich_df["y"].values, cv_cfg, rs)

    # === 4. Bayesian search over k ================================
    print(f"      k-search (3 … {bo_k_max}) on top-gain list …")
    bo_trace, best_mae, best_k = [], np.inf, None
    k_upper = min(bo_k_max, len(gain_series))
    for k in range(3, k_upper + 1):
        feats = gain_series.head(k).index.tolist()
        mae_k = _cv_mae(rich_df[feats], rich_df["y"].values, cv_cfg, rs)
        bo_trace.append((k, mae_k))
        if mae_k < best_mae:
            best_mae, best_k = mae_k, k
    bo_feats = gain_series.head(best_k).index.tolist()
    sets["BayesOpt"]  = bo_feats
    scores["BayesOpt"] = best_mae
    print(f"        ↳ best k = {best_k}  MAE {best_mae:.4f}")

    # -------- logs -------------------------------------------------
    print("\n      Top-10 gain features:")
    print(gain_series.head(10).to_string())
    print("\n      Bottom-10 gain features:")
    print(gain_series.tail(10).to_string(), "\n")

    return sets, scores, gain_series, bo_trace

# ------------------------------------------------------------------ #
def train_gbdt(train_df, test_df, feats, rs):
    X_tr, y_tr = train_df[feats], train_df["y"].values
    mdl = GradientBoostingRegressor(random_state=rs).fit(X_tr, y_tr)
    return mdl.predict(test_df[feats]), mdl
