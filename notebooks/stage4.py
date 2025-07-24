# notebooks/stage4.py
"""
Stage 4 — Baseline GBT-C **vs** SARIMA
======================================
Goal
----
* Benchmark the earlier Gradient-Boosting **C-set** model against a **SARIMA**
  pure-time-series model.
* Assess stationarity (ADF test) → decide differencing order *d*.
* Suggest small search grids for AR (*p*) and MA (*q*) orders via ACF/PACF.
* Train both models on 2015-2023, forecast 2024, **plus the common 3-day
  window**.
* Plot (same colours for both charts) and print summary metrics.

Loss choice
-----------
We continue to use **MAE** as primary loss because:
* It is scale-preserving (kWh) and directly answers “average absolute error”.
* It is more robust than RMSE to occasional spikes in energy demand.
SARIMA will be fitted by maximising log-likelihood (default in
`statsmodels`), but the **evaluation** uses MAE/RMSE/R² for fair comparison.
"""

# %% 0.0 Imports ---------------------------------------------------
import warnings, itertools
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["axes.grid"] = True
COLORS = {"C": "tab:blue", "SARIMA": "tab:orange"}  # ensure consistent colours

# %% 0.1 Load & split --------------------------------------------
DATA_PATH = (
    "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/"
    "data/renewables/dataset.csv"
)

df = pd.read_csv(DATA_PATH, comment="#")
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)  # convenient for SARIMA

target = "electricity"

TRAIN_END = "2023-12-31 23:00:00"
train_ts = df.loc[:TRAIN_END, target]

test_df  = df.loc["2024-01-01":].copy()

print(f"Training horizon: {train_ts.index[0]} — {train_ts.index[-1]}  ({len(train_ts):,} points)")
print(f"Test horizon:     {test_df.index[0]} — {test_df.index[-1]}  ({len(test_df):,} points)")

# %% 0.2 Stationarity diagnostics (ADF) ---------------------------
print("\nAugmented Dickey–Fuller test on raw series:")
adf_raw = adfuller(train_ts)
print(f"ADF statistic = {adf_raw[0]:.3f}, p-value = {adf_raw[1]:.4f}")

# If p > 0.05, difference once and test again
if adf_raw[1] > 0.05:
    train_diff = train_ts.diff().dropna()
    adf_diff = adfuller(train_diff)
    print("ADF after 1st differencing:")
    print(f"  stat = {adf_diff[0]:.3f}, p = {adf_diff[1]:.4f}")
    d_order = 1 if adf_diff[1] < 0.05 else 2  # if still non-stationary, allow 2
else:
    d_order = 0
print(f"Chosen differencing order d = {d_order}\n")

# %% 0.3 Quick ACF / PACF peek to suggest p & q -------------------
# Only plotted if interactive; here we just extract reasonable ranges
lag_max = 24  # one day of hourly lags
plot_acf(train_ts.diff(d_order).dropna(), lags=lag_max)
plot_pacf(train_ts.diff(d_order).dropna(), lags=lag_max, method="ywm")
plt.show()

# Heuristic: search p,q in [0,1,2], seasonal P,Q in [0,1]
P = Q = range(0, 2)
d = D = d_order
p = q = range(0, 3)
S = 24  # daily seasonality for hourly data

# limit grid for tutorial-speed search
order_grid      = list(itertools.product(p, [d], q))
seasonal_grid   = list(itertools.product(P, [D], Q))
print(f"Grid sizes → orders {len(order_grid)} × seasonal {len(seasonal_grid)} = {len(order_grid)*len(seasonal_grid)} models\n")

# %% 0.4 Select SARIMA via small grid search ----------------------
min_aic, best_cfg = np.inf, None
for (p_, d_, q_) in order_grid:
    for (P_, D_, Q_) in seasonal_grid:
        try:
            model = SARIMAX(train_ts, order=(p_, d_, q_),
                            seasonal_order=(P_, D_, Q_, S),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if res.aic < min_aic:
                min_aic, best_cfg = res.aic, (p_, d_, q_, P_, D_, Q_)
        except Exception:
            continue
print(f"Best SARIMA config (AIC {min_aic:.1f}) → order={best_cfg[:3]}, seasonal={best_cfg[3:]}\n")
# Best SARIMA config (AIC -337959.8) → order=(2, 0, 2), seasonal=(1, 0, 1)

# %% 0.5 Fit final SARIMA on train -------------------------------
(p_, d_, q_, P_, D_, Q_) = best_cfg
sarima_model = SARIMAX(train_ts, order=(p_, d_, q_),
                       seasonal_order=(P_, D_, Q_, S),
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)

sarima_pred  = sarima_model.predict(start=test_df.index[0], end=test_df.index[-1])

# %% 0.6 Prepare C-set features & fit GBT -------------------------
# Re-use C feature engineering like Stage 1 but on train/test split we have now

def add_time(df):
    df = df.copy()
    df["hour"]  = df.index.hour
    df["month"] = df.index.month
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)
    return df

def add_lag_cols(df, cols, lags):
    df = df.copy()
    for col in cols:
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
    return df

lag_cols, lags = ["electricity", "irradiance_direct"], [1, 24]

train_feat = add_time(df.loc[:TRAIN_END].copy())
train_feat = add_lag_cols(train_feat, lag_cols, lags).dropna()

C_cols = [
    "irradiance_direct", "irradiance_diffuse", "t2m", "cldtot",
    "hour", "month",
    "electricity_lag1", "electricity_lag24",
    "irradiance_direct_lag1", "irradiance_direct_lag24",
]

X_tr, y_tr = train_feat[C_cols], train_feat[target].values

# build test features (careful: cannot shift beyond test horizon)
all_feat = add_time(df.copy())
all_feat = add_lag_cols(all_feat, lag_cols, lags)
X_te = all_feat.loc[test_df.index, C_cols].fillna(method="bfill")  # back-fill initial NaNs

c_gbt = GradientBoostingRegressor(random_state=42)
c_gbt.fit(X_tr, y_tr)

c_pred = c_gbt.predict(X_te)

# %% 0.7 Metrics ---------------------------------------------------

def metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "R2": r2_score(true, pred),
    }

full_metrics = {
    "C":      metrics(test_df[target].values, c_pred),
    "SARIMA": metrics(test_df[target].values, sarima_pred.values),
}

# common 3-day window (ensure both have same index)
start_3d = test_df.index[0]
end_3d   = start_3d + pd.Timedelta(days=3)
mask_3d  = (test_df.index >= start_3d) & (test_df.index < end_3d)

metrics_3d = {
    "C":      metrics(test_df.loc[mask_3d, target].values, c_pred[mask_3d]),
    "SARIMA": metrics(test_df.loc[mask_3d, target].values, sarima_pred.values[mask_3d]),
}

# %% 0.8 Plots -----------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(test_df.index[mask_3d], test_df.loc[mask_3d, target], label="True", color="black", linewidth=2)
plt.plot(test_df.index[mask_3d], c_pred[mask_3d], label="Pred C", color=COLORS["C"])
plt.plot(test_df.index[mask_3d], sarima_pred.values[mask_3d], label="Pred SARIMA", color=COLORS["SARIMA"])
plt.title("Predicted vs True • first 3 days of 2024")
plt.xlabel("Time")
plt.ylabel("Electricity")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 4))
abs_err_c     = np.abs(c_pred[mask_3d] - test_df.loc[mask_3d, target].values)
abs_err_sar   = np.abs(sarima_pred.values[mask_3d] - test_df.loc[mask_3d, target].values)
plt.plot(test_df.index[mask_3d], abs_err_c, label="|Error| C", color=COLORS["C"])
plt.plot(test_df.index[mask_3d], abs_err_sar, label="|Error| SARIMA", color=COLORS["SARIMA"])
plt.title("Absolute Error • first 3 days of 2024")
plt.xlabel("Time")
plt.ylabel("|Prediction − True|")
plt.legend()
plt.tight_layout()
plt.show()

# %% 0.9 Summary ---------------------------------------------------
print("===== HOLD-OUT 2024 METRICS =====")
for m, s in full_metrics.items():
    print(f"{m:6s} | MAE {s['MAE']:.3f} | RMSE {s['RMSE']:.3f} | R² {s['R2']:.3f}")
print("\n===== FIRST 3 DAYS METRICS =====")
for m, s in metrics_3d.items():
    print(f"{m:6s} | MAE {s['MAE']:.3f} | RMSE {s['RMSE']:.3f} | R² {s['R2']:.3f}")



# %%
# Notes
"""
Loss discussion
---------------
* **Training objective**: SARIMA uses maximum-likelihood under Gaussian
  residuals; GBT minimises squared error.
* **Evaluation**: We choose **MAE** and RMSE but *rank* by MAE.  MAE is more
  suitable when the business cares about average absolute deviation and when
  large spikes are not to be over-penalised (RMSE squares them).  Electricity
  load can have occasional peaks (e.g. cold evenings); MAE keeps their impact
  proportional, making it the primary score here.
"""

# %%