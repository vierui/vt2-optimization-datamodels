#notebooks/stage1.py
"""
Stage 0 + Stage 1 — single-file notebook script (corrected)
==========================================================
Fixes the KeyError by keeping **two separate engineering tracks**:
* **Basic track** → features up to lag24 (sets A-D)
* **Rich track**  → extra lags 48/168 + rolling means (sets E-F)

Each track uses its own dataframe, so column look-ups never fail.
"""

# %% 0.0 Imports ---------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

# %% 0.1 Load data & hold-out 2024 --------------------------------
path = "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/data/renewables/dataset.csv"
df   = pd.read_csv(path, comment="#")
df["time"] = pd.to_datetime(df["time"])

test_df  = df[df["time"] >= "2024-01-01"].copy()  # never touched until final test
train_df = df[df["time"] <  "2024-01-01"].copy()
print(f"Train+Val rows: {len(train_df):,}\t|\tTest rows: {len(test_df):,}")

target = "electricity"  # label column

# %% 0.2 Helper functions -----------------------------------------

def regression_metrics(y, y_pred):
    return {"MAE": mean_absolute_error(y, y_pred),
            "RMSE": mean_squared_error(y, y_pred),
            "R2": r2_score(y, y_pred)}

def add_time_features(df):
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)
    return df

def add_lags(df, cols, lags):
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def add_rolling_means(df, cols, windows=(24,168)):
    df = df.copy()
    for col in cols:
        for w in windows:
            df[f"{col}_roll{w}"] = df[col].rolling(window=w, min_periods=w).mean()
    return df

lag_cols     = ["electricity", "irradiance_direct"]
lags_basic   = [1, 24]
lags_rich    = [1, 24, 48, 168]

# %% 0.3 BASIC engineering track (A-D) -----------------------------
train_df = add_time_features(train_df)
train_df = add_lags(train_df, lag_cols, lags_basic)
train_df.dropna(inplace=True)

test_df  = add_time_features(test_df)
test_df  = add_lags(test_df,  lag_cols, lags_basic)
# we won’t dropna on test yet (tiny NaNs at start acceptable or can drop)

a_d_sets = [
    (["irradiance_direct", "hour", "month"],
     "A: irr_dir + time"),

    (["irradiance_direct", "t2m", "cldtot", "hour", "month"],
     "B: irr_dir + t2m + cldtot + time"),

    (["irradiance_direct", "irradiance_diffuse", "t2m", "cldtot", "hour", "month"],
     "C: irr_dir + irr_diff + t2m + cldtot + time"),

    (["irradiance_direct", "t2m", "cldtot", "hour_sin", "hour_cos", "month_sin", "month_cos",
      "electricity_lag1", "electricity_lag24",
      "irradiance_direct_lag1", "irradiance_direct_lag24"],
     "D: irr_dir + t2m + cldtot + cyclical + lags")
]

# %% 0.4 RICH engineering track (E-F) ------------------------------
train_df_rich = add_lags(train_df.copy(), lag_cols, lags_rich)  # adds 48/168
train_df_rich = add_rolling_means(train_df_rich, lag_cols, (24,168))
train_df_rich.dropna(inplace=True)

test_df_rich  = add_lags(test_df.copy(),  lag_cols, lags_rich)
test_df_rich  = add_rolling_means(test_df_rich, lag_cols, (24,168))

# feature sets E & F (note: F grabs every engineered col except time & label)
e_f_sets = [
    (["irradiance_direct", "t2m", "cldtot", "hour_sin", "hour_cos", "month_sin", "month_cos",
      "electricity_lag1", "electricity_lag24", "electricity_lag48", "electricity_lag168",
      "irradiance_direct_lag1", "irradiance_direct_lag24", "irradiance_direct_lag48",
      "electricity_roll24", "electricity_roll168"],
     "E: + richer lags & rolls"),

    ([c for c in train_df_rich.columns if c not in ("time", target)],
     "F: ALL engineered features")
]

# %% 0.5 TimeSeriesSplit object ------------------------------------
tscv = TimeSeriesSplit(n_splits=5, test_size=24*30, gap=0)
print(tscv)

# %% 0.6 Cross-validation helper -----------------------------------
cv_results, feature_lookup = {}, {}

def run_cv(df, sets, label_suffix=""):
    for feats, label in sets:
        X, y = df[feats], df[target].values
        scores = []
        for fold, (tr_ix, val_ix) in enumerate(tscv.split(X), 1):
            m = GradientBoostingRegressor(random_state=42)
            m.fit(X.iloc[tr_ix], y[tr_ix])
            scores.append(regression_metrics(y[val_ix], m.predict(X.iloc[val_ix])))
            print(f"{label_suffix}{label} | Fold {fold}: {scores[-1]}")
        cv_results[label] = scores
        feature_lookup[label] = feats
        mean_mae = np.mean([d["MAE"] for d in scores])
        print(f"→ {label_suffix}{label} | Mean CV MAE: {mean_mae:.3f}\n{'-'*70}")

run_cv(train_df,      a_d_sets, "")
run_cv(train_df_rich, e_f_sets, "")

# %% 0.7 Pick best set by mean CV MAE 
cv_mae = {lbl: np.mean([d["MAE"] for d in folds]) for lbl, folds in cv_results.items()}
cv_mae_series = pd.Series(cv_mae).sort_values()
print("\nCV MAE summary:")
print(cv_mae_series)

best_label = cv_mae_series.idxmin()
best_feats = feature_lookup[best_label]
print(f"\nBest feature set: {best_label}\n")

# Choose appropriate engineered DF
if best_label.startswith(("E", "F")):
    tr_df_final, te_df_final = train_df_rich, test_df_rich
else:
    tr_df_final, te_df_final = train_df, test_df

# %% 0.8 Fit final model on 2015-2023, evaluate on 2024 ------------
X_train, y_train = tr_df_final[best_feats], tr_df_final[target].values
X_test,  y_test  = te_df_final[best_feats], te_df_final[target].values

final_model = GradientBoostingRegressor(random_state=42)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)
final_score = regression_metrics(y_test, final_pred)
print(f"{best_label} | 2024 hold-out: {final_score}\n")

# %% 0.9 Importance (gain & permutation) ---------------------------

# sort descending by permutation importance (more robust than gain)
imp_df = (pd.concat([
    pd.Series(final_model.feature_importances_, index=best_feats, name="Gain"),
    pd.Series(permutation_importance(final_model, X_test, y_test,
                                     n_repeats=5, random_state=42, n_jobs=-1)
              .importances_mean,
              index=best_feats, name="Perm")], axis=1)
          .sort_values("Perm", ascending=False))

n_total = len(imp_df)
print("Top features by permutation importance (up to 20):")
print(imp_df.head(min(20, n_total)))

# safe bottom-N slice
n_bottom = min(10, n_total)
least_df = imp_df.tail(n_bottom).sort_values("Perm")  # ascending
print(f"Least‑useful {n_bottom} feature(s) (perm.):")
print(least_df)
least_useful = least_df.index.tolist()


# %% 1.0 Summary ---------------------------------------------------
print("""
### Stage 1 Summary
Best CV set → {}  (mean MAE {:.3f})
2024 hold-out → MAE {:.3f}  |  RMSE {:.3f}  |  R² {:.3f}
Least-useful 10 features: {}
""".format(best_label, cv_mae[best_label],
           final_score["MAE"], final_score["RMSE"], final_score["R2"],
           ", ".join(least_useful)))

# %%
# personnal notes 
"""
**Stage 1 complete**
###CV
What CV gives you
• More robust estimate of how that feature set generalises (vs a single split).
• Ranking of sets.
Final model is not an average; after you pick the winning set you retrain one last 
model on the entire 2015-2023 data (no validation folds) and then score on 2024.So the final model is not an average of the 5 CV models; it’s a fresh fit on all pre-2024 data.


###Richer lags and rolling means : Why
	•	24 h mean smooths diurnal noise.
	•	168 h mean captures weekly rhythm.
 
###table
Item
Result
Added higher-order lags (48 h, 168 h)
Better persistence capture
Added rolling means (24 h, 168 h)
Smoothed inputs for trees
Computed per-fold importances
Objective basis to prune features
Identified a ~95 %-variance feature subset
Input for faster Stage 2 grid 

###feature importance
Anything with very low importance (≤ 0.5 % total gain) can be dropped to
shrink training time.


"""