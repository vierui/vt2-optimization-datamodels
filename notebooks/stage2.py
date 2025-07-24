#notebooks/stage2.py
"""
Stage 2 — automated feature‑selection notebook script
====================================================
Adds a **backward sequential feature selector (BSFS)** step to hunt for the
feature subset that minimises mean‑absolute‑error (MAE).  The workflow is:

1. **Engineering** ‑ replicate the basic and rich tracks from the original.
2. **Baseline set C** ‑ best manual set from Stage 0.
3. **Full set** ‑ every engineered column (rich track).
4. **BSFS** ‑ start from the full set and prune features with cross‑validated
   BSFS (scoring on MAE with a time‑series split).
5. Compare CV MAE of **set C** vs **BSFS subset**; keep whichever scores lower.
6. Fit the final model on all 2015‑2023 data; evaluate on the untouched 2024
   hold‑out.

The notebook‑style cell markers ("# %%") are preserved so that this script can
still be opened as a single‑file Jupyter notebook.
"""

"""
Stage 1 — faster feature‑selection notebook script
=================================================
Replaces the expensive **backward sequential feature selector (BSFS)** that ran
on the *entire* feature space with a **two‑step, much faster routine**:

1. **Quick model fit** on all engineered features (rich track) → rank by
   built‑in *gain* importance.
2. **Keep the top‑N features** (default *N = 40*).  Optional: run a smaller
   BSFS on just those N to squeeze a few more points if you like.

Finally we compare:
* **C (manual best)**
* **Trimmed‑N** (or **Trimmed+BSFS**) subset

and pick whichever minimises mean cross‑validated MAE.

All original notebook cell markers (`# %%`) are preserved.
"""

# %% 0.0 Imports 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector

# %% 0.1 Load data & hold‑out 2024 
DATA_PATH = (
    "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/"
    "data/renewables/dataset.csv"
)

df = pd.read_csv(DATA_PATH, comment="#")
df["time"] = pd.to_datetime(df["time"])

TEST_START = "2024-01-01"
train_df = df[df["time"] < TEST_START].copy()
test_df  = df[df["time"] >= TEST_START].copy()
print(f"Train+Val rows: {len(train_df):,}\t|\tTest rows: {len(test_df):,}")

target = "electricity"

# %% 0.2 Helper functions 

def regression_metrics(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }


def add_time_features(df):
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_lags(df, cols, lags):
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_means(df, cols, windows=(24, 168)):
    df = df.copy()
    for col in cols:
        for w in windows:
            df[f"{col}_roll{w}"] = df[col].rolling(window=w, min_periods=w).mean()
    return df

lag_cols   = ["electricity", "irradiance_direct"]
lags_basic = [1, 24]
lags_rich  = [1, 24, 48, 168]

# %% 0.3 BASIC engineering (baseline C) 
train_df = add_time_features(train_df)
train_df = add_lags(train_df, lag_cols, lags_basic)
train_df.dropna(inplace=True)

C_feats = [
    "irradiance_direct", "irradiance_diffuse", "t2m", "cldtot",
    "hour", "month",
    "electricity_lag1", "electricity_lag24",
    "irradiance_direct_lag1", "irradiance_direct_lag24",
]
print(f"Baseline set C has {len(C_feats)} features")

# %% 0.4 RICH engineering (candidate pool) 
train_df_rich = add_lags(train_df.copy(), lag_cols, lags_rich)
train_df_rich = add_rolling_means(train_df_rich, lag_cols, (24, 168))
train_df_rich.dropna(inplace=True)

FULL_feats = [c for c in train_df_rich.columns if c not in ("time", target)]
print(f"Full engineered space = {len(FULL_feats)} features")

# %% 0.5 CV splitter 
SPLITS, HORIZON = 5, 24 * 30

tscv = TimeSeriesSplit(n_splits=SPLITS, test_size=HORIZON)
print(tscv)

# %% 0.6 Utility: cross‑validated MAE 
def cv_mae(X, y):
    maes = []
    for tr_ix, val_ix in tscv.split(X):
        m = GradientBoostingRegressor(random_state=42)
        m.fit(X.iloc[tr_ix], y[tr_ix])
        maes.append(mean_absolute_error(y[val_ix], m.predict(X.iloc[val_ix])))
    return np.array(maes)

# Baseline C 
print("\nBaseline C …")
C_maes = cv_mae(train_df[C_feats], train_df[target].values)
print("Per‑fold MAE:", C_maes.round(3))
mean_C = C_maes.mean()
print(f"Mean MAE C = {mean_C:.3f}\n")

# %% 0.7 Fast importance‑based trimming $
N_KEEP       = 13   # how many top features to keep -> env. 50%
RUN_BSFS     = True # whether to fine‑tune further with BSFS

print("Ranking features by gain importance (single fit) …")
X_FULL, y_FULL = train_df_rich[FULL_feats], train_df_rich[target].values

rank_model = GradientBoostingRegressor(random_state=42)
rank_model.fit(X_FULL, y_FULL)

gain_series = pd.Series(rank_model.feature_importances_, index=FULL_feats)
TOP_feats   = gain_series.sort_values(ascending=False).head(N_KEEP).index.tolist()
print(f"Kept top {len(TOP_feats)} features → {TOP_feats[:6]} …")

print("Cross‑validating trimmed set …")
trim_maes = cv_mae(X_FULL[TOP_feats], y_FULL)
mean_trim = trim_maes.mean()
print("Per‑fold MAE:", trim_maes.round(3))
print(f"Mean MAE trimmed = {mean_trim:.3f}\n")

# %% 0.8 Optional BSFS on trimmed set 
if RUN_BSFS:
    print("Running BSFS on trimmed features …")
    bsfs = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=42),
        direction="backward",
        n_features_to_select="auto",
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        tol=1e-4,
    )
    bsfs.fit(X_FULL[TOP_feats], y_FULL)
    SEL_feats = list(np.array(TOP_feats)[bsfs.get_support()])
    print(f"Selected {len(SEL_feats)} features after BSFS")
    sel_maes  = cv_mae(X_FULL[SEL_feats], y_FULL)
    mean_sel  = sel_maes.mean()
    print("Per‑fold MAE:", sel_maes.round(3))
    print(f"Mean MAE trimmed+BSFS = {mean_sel:.3f}\n")
else:
    SEL_feats, mean_sel = TOP_feats, mean_trim

# %% 0.9 Decide winning set --------------------------------------
candidates = {"C (manual)": (C_feats, mean_C),
              "Trimmed":      (TOP_feats, mean_trim),
              "Trimmed+BSFS": (SEL_feats, mean_sel)}

best_label, (best_feats, best_mae) = min(candidates.items(), key=lambda x: x[1][1])
print(f"Winning set → {best_label} with mean CV MAE {best_mae:.3f}\n")

# %% 1.0 Assemble final train/test DFs ----------------------------
if best_label == "C (manual)":
    tr_df_final = train_df
    te_df_final = add_lags(add_time_features(test_df.copy()), lag_cols, lags_basic)
else:
    tr_df_final = train_df_rich
    te_df_final = add_rolling_means(
        add_lags(add_time_features(test_df.copy()), lag_cols, lags_rich),
        lag_cols, (24, 168),
    )

# Drop any rows in hold‑out that still have NaNs ---------------
rows_before = len(te_df_final)
te_df_final = te_df_final.dropna(subset=best_feats + [target])
print(f"Dropped {rows_before - len(te_df_final)} NaN row(s) from hold‑out")

# %% 1.1 Fit on 2015‑2023, score on 2024 --------------------------
X_train, y_train = tr_df_final[best_feats], tr_df_final[target].values
X_test,  y_test  = te_df_final[best_feats], te_df_final[target].values

final_model = GradientBoostingRegressor(random_state=42)
final_model.fit(X_train, y_train)
final_pred  = final_model.predict(X_test)
final_score = regression_metrics(y_test, final_pred)
print("2024 hold‑out →", final_score)

# %% 1.2 Permutation importance of final model ---------
# -----------
imp = permutation_importance(final_model, X_test, y_test, n_repeats=3, random_state=42)
imp_df = (pd.DataFrame({"Perm": imp.importances_mean}, index=best_feats)
            .sort_values("Perm", ascending=False))
print("\nTop 12 features (perm importance):")
print(imp_df.head(12))

# %% 1.3 Summary ---------------------------------------------------
print("""
### Stage 1 Summary
Best feature set → {}  (mean CV MAE {:.3f})
2024 hold‑out  → MAE {:.3f} | RMSE {:.3f} | R² {:.3f}
#Features kept → {}
""".format(
    best_label, best_mae,
    final_score["MAE"], final_score["RMSE"], final_score["R2"],
    len(best_feats),
))

# %%
