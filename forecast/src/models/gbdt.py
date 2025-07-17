"""
Gradient-Boosting helpers — now builds three sets:
  • Manual C
  • Backward SFS on top-N (BSFS)
  • Forward SFS seeded with the best K features
"""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector


# ------------------------------------------------------------------ #
def _cv_mae(X, y, n_splits, horizon, rs):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
    return np.mean([
        mean_absolute_error(y[val],
            GradientBoostingRegressor(random_state=rs)
                .fit(X.iloc[tr], y[tr])
                .predict(X.iloc[val]))
        for tr, val in tscv.split(X)
    ])


def build_feature_sets(manual_feats, rich_df, n_keep,
                       backward_keep, forward_pool, cv_cfg, rs):
    """
    Returns:
        sets_dict    dict[label] -> feature list
        scores_dict  dict[label] -> CV MAE
        gain_series  pandas Series of gain importance (all engineered cols)
    """
    sets, scores = {}, {}

    # -- full gain ranking (single fit) -----------------------------
    full_feats  = [c for c in rich_df.columns if c not in ("time", "y")]
    rank_model  = GradientBoostingRegressor(random_state=rs).fit(
                     rich_df[full_feats], rich_df["y"].values)
    gain_series = pd.Series(rank_model.feature_importances_,
                            index=full_feats).sort_values(ascending=False)

    print("\n      Top-10 gain features:")
    print(gain_series.head(10).to_string())
    print("\n      Bottom-10 gain features:")
    print(gain_series.tail(10).to_string(), "\n")

    # ---------- Manual (C) ----------------------------------------
    sets["Manual"] = manual_feats
    scores["Manual"] = _cv_mae(rich_df[manual_feats], rich_df["y"].values,
                               cv_cfg["n_splits"], cv_cfg["horizon_hours"], rs)

    # ---------- pick top-N for SFS --------------------------------
    topN = gain_series.head(n_keep).index.tolist()

    # ---------- (1) Backward SFS ----------------------------------
    print("      Backward SFS … ({:d} ➜ {:d} features)".format(
          len(topN), backward_keep))
    sfs_back = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=rs),
        direction="backward",
        n_features_to_select=backward_keep,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=cv_cfg["n_splits"],
                           test_size=cv_cfg["horizon_hours"]),
        tol=1e-3,            # stop earlier if no improvement
        n_jobs=-1)
    sfs_back.fit(rich_df[topN], rich_df["y"].values)
    bsfs_feats = list(np.array(topN)[sfs_back.get_support()])

    sets["BSFS"]   = bsfs_feats
    scores["BSFS"] = _cv_mae(rich_df[bsfs_feats], rich_df["y"].values,
                             cv_cfg["n_splits"], cv_cfg["horizon_hours"], rs)

    # ---------- (2) Forward SFS -----------------------------------
    seed_feats = gain_series.head(forward_pool).index.tolist()
    print(f"      Forward SFS seed → {seed_feats}")
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

    sets["FwdSFS"]   = fwd_feats
    scores["FwdSFS"] = _cv_mae(rich_df[fwd_feats], rich_df["y"].values,
                               cv_cfg["n_splits"], cv_cfg["horizon_hours"], rs)

    return sets, scores, gain_series


# ------------------------------------------------------------------ #
def train_gbdt(train_df, test_df, feats, rs):
    X_tr, y_tr = train_df[feats], train_df["y"].values
    mdl = GradientBoostingRegressor(random_state=rs).fit(X_tr, y_tr)
    return mdl.predict(test_df[feats]), mdl