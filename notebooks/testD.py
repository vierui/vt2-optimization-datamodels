# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# %%
# 1. Load Data
path = "/Users/rvieira/Documents/Master/vt2-optimization-datamodels/data/renewables/dataset.csv"
df = pd.read_csv(path, comment='#')
df['time'] = pd.to_datetime(df['time'])

# %%
# 2. Split: Train, Validation, Test - OPERATIONAL FORECASTING
# Proper lag buffer handling for forecasting from 2024-01-01
max_lag = 168  # 7 days (our longest lag)
forecast_start = pd.Timestamp('2024-01-01')
test_start = forecast_start - pd.Timedelta(hours=max_lag)  # Buffer for lags
test_end = forecast_start + pd.Timedelta(days=7)  # 7 days evaluation

print(f"Forecast starts: {forecast_start}")
print(f"Test data starts: {test_start} (includes {max_lag}h lag buffer)")
print(f"Test data ends: {test_end} (7 days evaluation)")

train_df = df[df['time'] <= "2020-12-31"].copy()
valid_df = df[(df['time'] >= "2021-01-01") & (df['time'] <= "2023-12-24")].copy()
test_df = df[(df['time'] >= test_start) & (df['time'] < test_end)].copy()

print(f"\nData shapes:")
print(f"Train: {train_df.shape[0]} hours")
print(f"Valid: {valid_df.shape[0]} hours")
print(f"Test: {test_df.shape[0]} hours (includes lag buffer)")

# %%
# 3. Feature Engineering: Time features with sin/cos
for d in [train_df, valid_df, test_df]:
    d['hour'] = d['time'].dt.hour
    d['month'] = d['time'].dt.month
    d['dayofweek'] = d['time'].dt.dayofweek
    d['dayofyear'] = d['time'].dt.dayofyear
    
    # Cyclical encoding
    d['hour_sin'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * (d['month']-1) / 12)
    d['month_cos'] = np.cos(2 * np.pi * (d['month']-1) / 12)
    d['dayofweek_sin'] = np.sin(2 * np.pi * d['dayofweek'] / 7)
    d['dayofweek_cos'] = np.cos(2 * np.pi * d['dayofweek'] / 7)
    d['dayofyear_sin'] = np.sin(2 * np.pi * d['dayofyear'] / 365)
    d['dayofyear_cos'] = np.cos(2 * np.pi * d['dayofyear'] / 365)

# %%
# 4. Add Lag Features - OPERATIONAL: Only target variable lags (no exogenous)
def add_target_lags(df, target, lags):
    """Add only target variable lags for operational forecasting"""
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df

# Only use target (electricity) lags - no future exogenous information
target = 'electricity'
lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week lags

for d in [train_df, valid_df, test_df]:
    add_target_lags(d, target, lags)
    d.dropna(inplace=True)  # Remove rows with NaN due to lags

# For test_df, identify forecast period (excluding lag buffer)
test_df['is_forecast_period'] = test_df['time'] >= forecast_start
forecast_mask = test_df['is_forecast_period'].values

print(f"\nAfter adding lags and removing NaN:")
print(f"Test data: {test_df.shape[0]} hours")
print(f"Forecast period: {forecast_mask.sum()} hours (actual evaluation period)")
print(f"Lag buffer: {(~forecast_mask).sum()} hours (for lag features only)")

# %%
# 5. Define Feature Sets - Set H Only (Operational)
# target already defined above

# Set H: Lag/Pattern-Based Features Only (NO EXOGENOUS/WEATHER DATA)
# Focus on operational forecasting with available information only
time_features = ['hour', 'month', 'dayofweek', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']

# Target lag features only (electricity past values)
target_lag_cols = [col for col in train_df.columns if col.startswith('electricity_lag')]

# Set H: Time features + Target lags only
set_h_features = time_features + target_lag_cols

# %%
# 6. Utility Functions
def prepare_set(df, features, target):
    # Only use features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target].values
    return X, y, available_features

def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test, label, **params):
    model = GradientBoostingRegressor(
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        **params
    )
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
        print(f"{label} - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
    print("-" * 60)
    return model, y_valid_pred, y_test_pred, metrics

# %%
# 7. Prepare data for Set H (Lag/Pattern-Based Modeling)
print("Preparing Set H: Lag/Pattern-Based Features for operational forecasting...")
X_train_h, y_train, available_features_h = prepare_set(train_df, set_h_features, target)
X_valid_h, y_valid, _ = prepare_set(valid_df, set_h_features, target)
X_test_h, y_test, _ = prepare_set(test_df, set_h_features, target)

# Extract forecast period data for evaluation
X_test_forecast = X_test_h[forecast_mask]
y_test_forecast = y_test[forecast_mask]

print(f"Set H available features: {len(available_features_h)}")
print(f"Training data shape: {X_train_h.shape}")
print(f"Test data shape (full): {X_test_h.shape}")
print(f"Forecast period shape: {X_test_forecast.shape}")
print(f"Features: {available_features_h}")

# %%
# 8. Bayesian Optimization for Set H (Lag/Pattern-Based Modeling)
print("\nRunning Bayesian optimization for Set H (Lag/Pattern-Based)...")

# Define search space for hyperparameter tuning
dimensions_h = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 12, name='max_depth'),
    Real(0.1, 1.0, name='subsample'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf'),
]

@use_named_args(dimensions_h)
def objective_h(n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf):
    # Model training with Set H features only
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    model.fit(X_train_h, y_train)
    y_valid_pred = model.predict(X_valid_h)
    
    mae = mean_absolute_error(y_valid, y_valid_pred)
    return mae

result_h = gp_minimize(objective_h, dimensions_h, n_calls=50, random_state=42, verbose=False)
best_params_h = {
    'n_estimators': result_h.x[0],
    'learning_rate': result_h.x[1],
    'max_depth': result_h.x[2],
    'subsample': result_h.x[3],
    'min_samples_split': result_h.x[4],
    'min_samples_leaf': result_h.x[5]
}

print(f"Set H - Best parameters:")
for param, value in best_params_h.items():
    print(f"  {param}: {value}")
print(f"  Best MAE: {result_h.fun:.3f}")

# Train Set H model - but evaluate only on forecast period
model_h = GradientBoostingRegressor(
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=10,
    **best_params_h
)
model_h.fit(X_train_h, y_train)

# Make predictions
y_valid_pred_h = model_h.predict(X_valid_h)
y_test_pred_full = model_h.predict(X_test_h)
y_test_pred_h = y_test_pred_full[forecast_mask]  # Only forecast period

# Calculate metrics for both validation and forecast period
metrics_h = {}
for split, y_true, y_pred in [
    ("Valid", y_valid, y_valid_pred_h),
    ("Test", y_test_forecast, y_test_pred_h)  # Only forecast period
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_h[split] = (rmse, mae, r2)
    print(f"H: Lag/Pattern-Based - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
print("-" * 60)

# %%
# 9. Feature Importance Analysis for Set H
print("\nAnalyzing feature importance for Set H...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': available_features_h,
    'importance': model_h.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances - Set H (Lag/Pattern-Based)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%
# 10. Results for Set H Only - Forecast Period Evaluation
all_test_preds = [y_test_pred_h]  # Only forecast period predictions
all_valid_preds = [y_valid_pred_h]
labels = ["H: Lag/Pattern-Based"]
metrics_dict = {"H: Lag/Pattern-Based": metrics_h}

# Use forecast period targets for plotting
y_test_plot = y_test_forecast

# %%
# 11. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
for i, (y_pred, label) in enumerate(zip(all_test_preds, labels)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1]})
    
    # Left plot: Time series forecast (48 hours = 2 days) - rectangular
    n_hours = 48
    ax1.plot(y_test_plot[:n_hours], label='Actual', linewidth=2, color='black')
    ax1.plot(y_pred[:n_hours], label=f'Predicted (Set H)', alpha=0.7, color='blue')
    ax1.set_title('Operational Forecast - Lag/Pattern-Based (First 2 Days from 2024-01-01)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Electricity (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Predicted vs Actual scatter - square with same height
    ax2.scatter(y_test_plot, y_pred, s=10, alpha=0.5, color='blue')
    ax2.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'r--', lw=2)
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
# 12. Daily Metrics Table (7-Day Forecast Period Only)
def calculate_daily_metrics(y_test, test_preds, labels, test_df, forecast_mask):
    # Only use forecast period data
    forecast_df = test_df[forecast_mask].copy()
    forecast_df['date'] = forecast_df['time'].dt.date
    unique_dates = sorted(forecast_df['date'].unique())
    
    daily_metrics = []
    
    for label, y_pred in zip(labels, test_preds):
        for i, date in enumerate(unique_dates):
            day_mask = forecast_df['date'] == date
            day_indices = forecast_df[day_mask].index - forecast_df.index[0]
            
            if len(day_indices) > 0:  # Check if there's data for this day
                y_true_day = y_test[day_indices]
                y_pred_day = y_pred[day_indices]
                
                mae = mean_absolute_error(y_true_day, y_pred_day)
                rmse = np.sqrt(mean_squared_error(y_true_day, y_pred_day))
                r2 = r2_score(y_true_day, y_pred_day)
                
                daily_metrics.append({
                    'Model': label,
                    'Date': str(date),
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                })
    
    return pd.DataFrame(daily_metrics)

daily_metrics_df = calculate_daily_metrics(y_test_plot, all_test_preds, labels, test_df, forecast_mask)

print("Daily Metrics Table (First Week):")
print("=" * 80)
for model in labels:
    model_data = daily_metrics_df[daily_metrics_df['Model'] == model]
    print(f"\n{model}:")
    print("-" * 60)
    print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
    print("-" * 60)
    for _, row in model_data.iterrows():
        print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

print("\n" + "=" * 80)
# %%
