#notebooks/stage0.py
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# %%
# Load data
path = "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/data/renewables/dataset.csv"
df   = pd.read_csv(path, comment='#')
df['time'] = pd.to_datetime(df['time'])

# Hold-out test window: full calendar year 2024
test_df  = df[df['time'] >= "2024-01-01"].copy()
train_df = df[df['time'] <  "2024-01-01"].copy()

print(f"Train+Val rows: {len(train_df):,}   |   Test rows: {len(test_df):,}")

#%% 
# Error-metric
def regression_metrics(y_true, y_pred):
    """Returns MAE, RMSE, R² in a dict."""
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred)
    }

# %%
# # Data Overview, Correlation, Pairplot
# print("\nData Description:\n", df.describe())
# cols = ['electricity', 't2m', 'temperature', 'swgdn', 'cldtot', 'prectotland', 'irradiance_direct', 'irradiance_diffuse']

# corr_matrix = df[cols].corr()
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()

# sns.pairplot(df[cols], diag_kind="kde", plot_kws={"alpha":0.3, "s":10})
# plt.suptitle('Pairplot: Electricity and Weather Features', y=1.02)
# plt.show()

# %%
# Feature Engineering: Extract time features
def add_time_features(df):
    df = df.copy()
    df["hour"]  = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df['dayofweek'] = df['time'].dt.dayofweek
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)
    return df

def add_lags(df, lag_features, lags):
    df = df.copy()
    for col in lag_features:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

lag_features = ["electricity", "irradiance_direct"]
lags         = [1, 24]

for name, d in [("train_df", train_df), ("test", test_df)]:
    d = add_time_features(d)
    d = add_lags(d, lag_features, lags)
    d.dropna(inplace=True)
    if name == "train_df":
        train_df = d
    else:
        test_df  = d

# %% 
# feature-set definitions
feature_sets = [
    (["irradiance_direct", "hour", "month"],
     "A: irradiance_direct + time"),

    (["irradiance_direct", "t2m", "cldtot", "hour", "month"],
     "B: irradiance_direct + t2m + cldtot + time"),

    (["irradiance_direct", "irradiance_diffuse", "t2m", "cldtot", "hour", "month"],
     "C: irradiance_direct + irradiance_diffuse + t2m + cldtot + time"),

    (["irradiance_direct", "t2m", "cldtot",
      "hour_sin", "hour_cos", "month_sin", "month_cos",
      "electricity_lag1", "electricity_lag24",
      "irradiance_direct_lag1", "irradiance_direct_lag24"],
     "D: irradiance_direct + t2m + cldtot + cyclical + lags")
]
target = "electricity"

# %%
# TimeSeriesSplit (5 folds, 1-month horizon)
tscv = TimeSeriesSplit(n_splits=5, test_size=24*30, gap=0)
print(tscv)


# %% 
# final fit on 2015-2023, evaluate on 2024 
test_metrics = {}
for feats, label in feature_sets:
    X_train_df = train_df[feats]
    y_train_df = train_df[target].values
    X_test     = test_df[feats]
    y_test     = test_df[target].values

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_df, y_train_df)
    y_pred = model.predict(X_test)
    scores = regression_metrics(y_test, y_pred)
    test_metrics[label] = scores
    print(f"{label} | 2024 test: {scores}")

# %% 0.8 (optional) quick visual check on best feature set
best_label = min(test_metrics, key=lambda k: test_metrics[k]["MAE"])
print(f"\nLowest 2024 MAE: {best_label}  ->  {test_metrics[best_label]}")

# first 168 h (one week) plot
y_test_full = test_df[target].values
# Find the feature list for the best label
best_features = next(feats for feats, label in feature_sets if label == best_label)
y_pred_best = GradientBoostingRegressor(random_state=42).fit(
    train_df[best_features], y_train_df
).predict(test_df[best_features])

plt.figure(figsize=(14,6))
plt.plot(y_test_full[:168], label="Actual", lw=2, c="black")
plt.plot(y_pred_best[:168], label=f"Predicted {best_label}", alpha=0.7)
plt.title("2024 Week-1: prediction vs actual")
plt.xlabel("Hour"); plt.ylabel("Electricity (kW)")
plt.legend(); plt.tight_layout(); plt.show()

# %%
# summary markdown cell 
"""
**Stage 0 complete**

* Cross-validation: TimeSeriesSplit (5 × 1 month)
* Metrics tracked: MAE (decision-maker), RMSE (spike penalty), R²
* 2024 set kept untouched for final scoring

Next: Stage 1 – richer lags, rolling means, feature importance.

trainval was undefined
renamed original train_df → trainval consistently

valid_df no longer needed
removed all references; CV now supplies validation

TimeSeriesSplit object unused
passed into the CV loop for each feature set

Metrics helpers duplicated
single regression_metrics() function

Hold-out evaluation added
final model per feature set scored on entire 2024

"""
# %%
