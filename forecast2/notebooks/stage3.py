#notebooks/stage1.py
"""
Stage 3 — fast selection **plus dual‑model comparison (robust window)**
========================================================================
* Fast importance trimming + optional mini‑BSFS
* Baseline **C** stays intact
* **Both** models trained on 2015‑2023, evaluated on 2024 **and** the first
  available 72 h that **both models share** after lag/rolling NaNs are dropped.
* Plots: true vs preds and absolute errors in that common 3‑day window.

This version fixes the *0‑sample* crash by dynamically computing a common
window that contains data for **all** models.
"""

# %% 0.0 Imports ---------------------------------------------------
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

# Matplotlib defaults ------------------------------------------------
plt.rcParams["axes.grid"] = True

# %% 0.1 Load data & split -----------------------------------------
DATA_PATH = (
    "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/"
    "data/renewables/dataset.csv"
)

df = pd.read_csv(DATA_PATH, comment="#")
df["time"] = pd.to_datetime(df["time"])

TEST_START = "2024-01-01"
train_df = df[df["time"] < TEST_START].copy()
test_df  = df[df["time"] >= TEST_START].copy()
print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

target = "electricity"

# %% 0.2 Helper functions -----------------------------------------

def regression_metrics(y_true, y_pred):
    """Return dict of MAE, RMSE, R² (sk‑learn‑version‑agnostic)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def add_time_features(df):
    df = df.copy()
    df["hour"]  = df["time"].dt.hour
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

# %% 0.3 Baseline C -----------------------------------------------
train_base = add_time_features(train_df)
train_base = add_lags(train_base, lag_cols, lags_basic)
train_base.dropna(inplace=True)

C_feats = [
    "irradiance_direct", "irradiance_diffuse", "t2m", "cldtot",
    "hour", "month",
    "electricity_lag1", "electricity_lag24",
    "irradiance_direct_lag1", "irradiance_direct_lag24",
]
print(f"Baseline C → {len(C_feats)} features")

# %% 0.4 Rich engineered DF ---------------------------------------
train_rich = add_lags(train_base.copy(), lag_cols, lags_rich)
train_rich = add_rolling_means(train_rich, lag_cols, (24, 168))
train_rich.dropna(inplace=True)

FULL_feats = [c for c in train_rich.columns if c not in ("time", target)]
print(f"Full engineered space = {len(FULL_feats)} features")

# %% 0.5 TimeSeries CV helper -------------------------------------
SPLITS, HORIZON = 5, 24 * 30

tscv = TimeSeriesSplit(n_splits=SPLITS, test_size=HORIZON)


def cv_mae(X, y):
    out = []
    for tr_ix, val_ix in tscv.split(X):
        m = GradientBoostingRegressor(random_state=42)
        m.fit(X.iloc[tr_ix], y[tr_ix])
        out.append(mean_absolute_error(y[val_ix], m.predict(X.iloc[val_ix])))
    return np.mean(out)

# %% 0.6 Fast importance trimming ---------------------------------
N_KEEP   = min(17, len(FULL_feats))
RUN_BSFS = True

print("Ranking by gain importance …")
rank_model = GradientBoostingRegressor(random_state=42)
rank_model.fit(train_rich[FULL_feats], train_rich[target].values)
rank = pd.Series(rank_model.feature_importances_, index=FULL_feats).sort_values(ascending=False)
TOP_feats = rank.head(N_KEEP).index.tolist()

# %% 0.7 Optional mini‑BSFS ---------------------------------------
if RUN_BSFS:
    print("Mini‑BSFS …")
    sfs = SequentialFeatureSelector(
        GradientBoostingRegressor(random_state=42),
        direction="backward",
        n_features_to_select="auto",
        scoring="neg_mean_absolute_error",
        cv=tscv,
        tol=1e-4,
        n_jobs=-1,
    )
    sfs.fit(train_rich[TOP_feats], train_rich[target].values)
    BSFS_feats = list(np.array(TOP_feats)[sfs.get_support()])
else:
    BSFS_feats = TOP_feats

print(f"Trim+BSFS → {len(BSFS_feats)} features")

# %% 0.8 Build test sets ------------------------------------------

def prep_test(base_df, rich: bool):
    if rich:
        d = add_lags(add_time_features(base_df.copy()), lag_cols, lags_rich)
        d = add_rolling_means(d, lag_cols, (24, 168))
    else:
        d = add_lags(add_time_features(base_df.copy()), lag_cols, lags_basic)
    d.dropna(inplace=True)
    return d

TEST_SETS = {
    "C":         prep_test(test_df, rich=False),
    "Trim+BSFS": prep_test(test_df, rich=True),
}

MODELS = {
    "C":         {"train_df": train_base, "feats": C_feats},
    "Trim+BSFS": {"train_df": train_rich, "feats": BSFS_feats},
}

# Identify **common** start time (latest first row across models) --
start_times = {lbl: df["time"].iloc[0] for lbl, df in TEST_SETS.items()}
common_start = max(start_times.values())  # latest start → everyone has data
window_end   = common_start + pd.Timedelta(days=3)
print(f"Common 3‑day window → {common_start:%Y-%m-%d %H:%M} → {window_end:%Y-%m-%d %H:%M}")

# %% 1.0 Train, predict, metric calc ------------------------------
results = {"full": {}, "3d": {}}
plot_dfs = {}

for lbl, cfg in MODELS.items():
    feats = cfg["feats"]
    tr_df = cfg["train_df"]
    te_df = TEST_SETS[lbl]

    X_tr, y_tr = tr_df[feats], tr_df[target].values
    X_te, y_te = te_df[feats], te_df[target].values

    mdl = GradientBoostingRegressor(random_state=42)
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)

    # full‑year metrics
    results["full"][lbl] = regression_metrics(y_te, y_pred)

    # common 3‑day slice
    mask = (te_df["time"] >= common_start) & (te_df["time"] < window_end)
    y_true_3d, y_pred_3d = y_te[mask], y_pred[mask]

    if len(y_true_3d) == 0:
        raise RuntimeError(f"Model '{lbl}' has no data in common window — check lag gaps.")

    results["3d"][lbl] = regression_metrics(y_true_3d, y_pred_3d)

    plot_dfs[lbl] = pd.DataFrame({
        "time": te_df.loc[mask, "time"].values,
        "pred": y_pred_3d,
    })

# ground‑truth for plotting
truth_df = pd.DataFrame({
    "time": plot_dfs["C"]["time"],
    "true": TEST_SETS["C"].loc[(TEST_SETS["C"]["time"] >= common_start) & (TEST_SETS["C"]["time"] < window_end), target].values,
})

# %% 1.1 Plot predictions vs true ---------------------------------
plt.figure(figsize=(11, 5))
plt.plot(truth_df["time"], truth_df["true"], label="True", linewidth=2)
for lbl, dfp in plot_dfs.items():
    plt.plot(dfp["time"], dfp["pred"], label=f"Pred {lbl}")
plt.title("Predicted vs True • common 3‑day window")
plt.ylabel("Electricity")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

# %% 1.2 Absolute error plot --------------------------------------
plt.figure(figsize=(11, 4))
for lbl, dfp in plot_dfs.items():
    abs_err = np.abs(dfp["pred"].values - truth_df["true"].values)
    plt.plot(dfp["time"], abs_err, label=f"|Error| {lbl}")
plt.title("Absolute error • common 3‑day window")
plt.ylabel("|Prediction − True|")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

# %% 1.3 Summary printout -----------------------------------------
print("\n===== FULL 2024 METRICS =====")
for lbl, m in results["full"].items():
    print(f"{lbl:11s} MAE {m['MAE']:.3f} | RMSE {m['RMSE']:.3f} | R² {m['R2']:.3f}")

print("\n===== COMMON 3‑DAY METRICS =====")
for lbl, m in results["3d"].items():
    print(f"{lbl:11s} MAE {m['MAE']:.3f} | RMSE {m['RMSE']:.3f} | R² {m['R2']:.3f}")

# %%
