# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# 1. Load Data
path = "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/data/renewables/dataset.csv"
df = pd.read_csv(path, comment='#')
df['time'] = pd.to_datetime(df['time'])

# %%
# 2. Data Overview, Correlation, Pairplot
print("\nData Description:\n", df.describe())
cols = ['electricity', 't2m', 'temperature', 'swgdn', 'cldtot', 'prectotland', 'irradiance_direct', 'irradiance_diffuse']

corr_matrix = df[cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df[cols], diag_kind="kde", plot_kws={"alpha":0.3, "s":10})
plt.suptitle('Pairplot: Electricity and Weather Features', y=1.02)
plt.show()

# %%
# 3. Split: Train, Validation, Test
train_df = df[df['time'] < "2019-07-01"].copy()
valid_df = df[(df['time'] >= "2019-07-01") & (df['time'] < "2020-01-01")].copy()
test_df  = df[(df['time'] >= "2020-01-01") & (df['time'] < "2020-01-04")].copy()

# %%
# 4. Feature Engineering: Extract time features
for d in [train_df, valid_df, test_df]:
    d['hour'] = d['time'].dt.hour
    d['month'] = d['time'].dt.month
    d['dayofweek'] = d['time'].dt.dayofweek

    d['hour_sin'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * (d['month']-1) / 12)
    d['month_cos'] = np.cos(2 * np.pi * (d['month']-1) / 12)


def add_lags(df, lag_features, lags):
    for col in lag_features:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

lag_features = ['electricity', 'irradiance_direct']
lags = [1, 24]

for d in [train_df, valid_df, test_df]:
    add_lags(d, lag_features, lags)
    d.dropna(inplace=True)  # drop first rows with missing lags

# %%
# 5. Define Feature Sets and Labels
feature_sets = [
    (['irradiance_direct', 'hour', 'month'],
     "A: irradiance_direct + time"),
    (['hour', 'month'],
     "B: time"),
    (['irradiance_direct', 'irradiance_diffuse', 't2m', 'cldtot', 'hour', 'month'],
     "C: irradiance_direct + irradiance_diffuse + t2m + cldtot + time"),
    (['irradiance_direct', 't2m', 'cldtot', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos','electricity_lag1', 'electricity_lag24', 'irradiance_direct_lag1', 'irradiance_direct_lag24'],
     "D: irradiance_direct + t2m + cldtot + cyclical + lags" )
]
target = 'electricity'

# %%
# 6. Utility Functions
def prepare_set(df, features, target):
    X = df[features]
    y = df[target].values
    return X, y

def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test, label):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    metrics = {}
    for split, y, y_pred in [
        ("Valid", y_valid, y_valid_pred),
        ("Test",  y_test,  y_test_pred)
    ]:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics[split] = (rmse, mae, r2)
        print(f"{label} - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  ")
    print("-" * 45)
    return model, y_valid_pred, y_test_pred, metrics

def plot_predictions_all(y_test, test_preds, labels, n=168):
    plt.figure(figsize=(14,6))
    plt.plot(y_test[:n], label='Actual', linewidth=2, color='black')
    for i, y_pred in enumerate(test_preds):
        plt.plot(y_pred[:n], label=f'Predicted {labels[i]}', alpha=0.7)
    plt.title('Test Set: Prediction vs Actual (First Week)')
    plt.xlabel('Hour')
    plt.ylabel('Electricity (kW)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def scatter_plot_predictions_all(y_test, test_preds, labels):
    plt.figure(figsize=(18,5))
    for i, y_pred in enumerate(test_preds):
        plt.subplot(1, len(test_preds), i+1)
        plt.scatter(y_test, y_pred, s=10, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()

# %%
# 7. Run All Models, Collect Predictions
all_test_preds = []
all_valid_preds = []
labels = []
metrics_dict = {}

for features, label in feature_sets:
    X_train, y_train = prepare_set(train_df, features, target)
    X_valid, y_valid = prepare_set(valid_df, features, target)
    X_test, y_test   = prepare_set(test_df,  features, target)
    model, y_valid_pred, y_test_pred, metrics = evaluate_model(
        X_train, y_train, X_valid, y_valid, X_test, y_test, label)
    all_valid_preds.append(y_valid_pred)
    all_test_preds.append(y_test_pred)
    labels.append(label)
    metrics_dict[label] = metrics

# %%
# 8. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
for i, (y_pred, label) in enumerate(zip(all_test_preds, labels)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1]})
    
    # Left plot: Time series forecast (48 hours = 2 days) - rectangular
    n_hours = 48
    ax1.plot(y_test[:n_hours], label='Actual', linewidth=2, color='black')
    ax1.plot(y_pred[:n_hours], label=f'Predicted', alpha=0.7, color='blue')
    ax1.set_title('Forecast (First 2 Days)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Electricity (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Predicted vs Actual scatter - square with same height
    ax2.scatter(y_test, y_pred, s=10, alpha=0.5, color='blue')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title('Predicted vs Actual')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Display metrics
    test_metrics = metrics_dict[label]["Test"]
    rmse, mae, r2 = test_metrics
    print(f"{label} - Test Metrics:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.3f}")
    print("-" * 50)

# %%
# 10. Summary
"""
Next steps:
- Add lagged features: past values of 'electricity', 'irradiance', etc.
- Add cycle indicators (e.g., sin(hour/24*2*pi)) for smoother cyclical features.
- Try XGBoost or LightGBM for often faster/stronger models.
"""