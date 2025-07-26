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

print(f"\\nData shapes:")
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
# 4. Add Lag Features
def add_target_lags(df, target, lags):
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df

target = 'electricity'
lags = [1, 2, 3, 6, 12, 24, 48, 168]

for d in [train_df, valid_df, test_df]:
    add_target_lags(d, target, lags)
    d.dropna(inplace=True)  # Remove rows with NaN due to lags

# For test_df, identify forecast period (excluding lag buffer)
test_df['is_forecast_period'] = test_df['time'] >= forecast_start
forecast_mask = test_df['is_forecast_period'].values

print(f"\\nForecast period analysis:")
print(f"Total test data: {test_df.shape[0]} hours")
print(f"Forecast period: {forecast_mask.sum()} hours (actual evaluation period)")
print(f"Lag buffer: {(~forecast_mask).sum()} hours (for lag features)")

# %%
# 5. Define Feature Sets (Time/Cyclical features only - no weather)
# All time features (including cyclical encoding)
time_features = ['hour', 'month', 'dayofweek', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']

# Target lag features
target_lag_cols = [col for col in train_df.columns if col.startswith('electricity_lag')]

# Combine time and lag features for optimization (NO WEATHER FEATURES)
all_cyclical_features = time_features + target_lag_cols

target = 'electricity'

# %%
# 6. Utility Functions
def prepare_set(df, features, target):
    # Only use features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target].values
    return X, y, available_features

def prepare_forecast_set(df, features, target, forecast_mask):
    """Prepare data with forecast period extraction"""
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target].values
    X_forecast = X[forecast_mask]
    y_forecast = y[forecast_mask]
    return X, y, X_forecast, y_forecast

def evaluate_model(X_train, y_train, X_valid, y_valid, X_test_forecast, y_test_forecast, label, **params):
    model = GradientBoostingRegressor(
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        **params
    )
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test_forecast)  # Only forecast period
    
    metrics = {}
    for split, y, y_pred in [
        ("Valid", y_valid, y_valid_pred),
        ("Test",  y_test_forecast,  y_test_pred)  # Only forecast period
    ]:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics[split] = (rmse, mae, r2)
        print(f"{label} - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
    print("-" * 60)
    return model, y_valid_pred, y_test_pred, metrics

# %%
# 7. Staged Bayesian Optimization: Feature Selection First, Then Hyperparameters
print("Preparing data for staged Bayesian optimization...")
X_train_all, y_train, available_features = prepare_set(train_df, all_cyclical_features, target)
X_valid_all, y_valid, _ = prepare_set(valid_df, all_cyclical_features, target)
X_test_all, y_test, _ = prepare_set(test_df, all_cyclical_features, target)

# Extract forecast period data for evaluation
X_test_forecast_all = X_test_all[forecast_mask]
y_test_forecast = y_test[forecast_mask]

print(f"Total available features: {len(available_features)}")
print(f"Training data shape: {X_train_all.shape}")
print(f"Validation data shape: {X_valid_all.shape}")
print(f"Features available for optimization (Time+Lag only): {available_features}")

# STAGE 1: Feature Selection Optimization with Cross Validation on Training Data Only
print("\n=== STAGE 1: Feature Selection Optimization ===")
dimensions_stage1 = [
    Integer(5, min(15, len(available_features)), name='n_features'),
]

@use_named_args(dimensions_stage1)
def objective_stage1(n_features):
    # Feature selection
    selector = SelectKBest(f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train_all, y_train)
    
    # Model with fixed hyperparameters (same as testB2.py)
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    # Time series cross-validation on TRAINING DATA ONLY
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train_selected, y_train, cv=tscv, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # Return negative of mean score (since cross_val_score returns negative MAE)
    return -scores.mean()

print("Running Stage 1: Feature selection optimization with cross validation...")
result_stage1 = gp_minimize(objective_stage1, dimensions_stage1, n_calls=40, random_state=42, verbose=False)

best_n_features = result_stage1.x[0]
print(f"Stage 1 Results:")
print(f"  Best n_features: {best_n_features}")
print(f"  Best CV MAE: {result_stage1.fun:.3f}")

# STAGE 2: Hyperparameter Optimization with Best Feature Set
print("\n=== STAGE 2: Hyperparameter Optimization ===")
# Select features using best n_features from stage 1
selector_best = SelectKBest(f_regression, k=best_n_features)
X_train_selected_best = selector_best.fit_transform(X_train_all, y_train)
X_valid_selected_best = selector_best.transform(X_valid_all)

dimensions_stage2 = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 12, name='max_depth'),
    Real(0.7, 1.0, name='subsample'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf'),
]

@use_named_args(dimensions_stage2)
def objective_stage2(n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf):
    # Model with hyperparameters to optimize
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Time series cross-validation on TRAINING DATA ONLY
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train_selected_best, y_train, cv=tscv, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # Return negative of mean score (since cross_val_score returns negative MAE)
    return -scores.mean()

print("Running Stage 2: Hyperparameter optimization with cross validation...")
result_stage2 = gp_minimize(objective_stage2, dimensions_stage2, n_calls=80, random_state=42, verbose=False)

# Extract best hyperparameters
best_params_stage2 = {
    'n_estimators': result_stage2.x[0],
    'learning_rate': result_stage2.x[1],
    'max_depth': result_stage2.x[2],
    'subsample': result_stage2.x[3],
    'min_samples_split': result_stage2.x[4],
    'min_samples_leaf': result_stage2.x[5]
}

print(f"Stage 2 Results:")
print(f"  Best hyperparameters: {best_params_stage2}")
print(f"  Best CV MAE: {result_stage2.fun:.3f}")

# Combine results
best_params_cv = {
    'n_features': best_n_features,
    **best_params_stage2
}

print(f"\nFinal Combined Results:")
print(f"  n_features: {best_n_features}")
for param, value in best_params_stage2.items():
    print(f"  {param}: {value}")
print(f"  Total optimization calls: {40 + 80} = 120")

# %%
# 8. Train final model with staged optimization results
print("\\nTraining final model with staged optimization results...")
# Use the feature selector from stage 1 (already fitted)
X_train_selected = X_train_selected_best  # Already computed in stage 2
X_valid_selected = X_valid_selected_best  # Already computed in stage 2
X_test_selected = selector_best.transform(X_test_all)
X_test_forecast_selected = selector_best.transform(X_test_forecast_all)

selected_features_cv = [available_features[i] for i in selector_best.get_support(indices=True)]
print(f"Selected features ({len(selected_features_cv)}): {selected_features_cv}")

# Model hyperparameters (excluding n_features)
model_params_cv = {k: v for k, v in best_params_cv.items() if k != 'n_features'}

best_model_cv = GradientBoostingRegressor(
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=10,
    **model_params_cv
)

best_model_cv.fit(X_train_selected, y_train)
y_valid_pred_best_cv = best_model_cv.predict(X_valid_selected)
y_test_pred_best_cv = best_model_cv.predict(X_test_forecast_selected)

# Calculate metrics for staged optimized model
metrics_optimized_cv = {}
for split, y_true, y_pred in [
    ("Valid", y_valid, y_valid_pred_best_cv),
    ("Test", y_test_forecast, y_test_pred_best_cv)  # Only forecast period
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_optimized_cv[split] = (rmse, mae, r2)
    print(f"D: Staged Bayesian Optimization + CV - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
print("-" * 60)

# %%
# 9. Results Setup
all_test_preds = [y_test_pred_best_cv]
all_valid_preds = [y_valid_pred_best_cv]
labels = ["D: Staged Bayesian Optimization + Cross Validation (Time+Lag Features)"]
metrics_dict = {"D: Staged Bayesian Optimization + Cross Validation (Time+Lag Features)": metrics_optimized_cv}

# Use forecast period targets for plotting
y_test_plot = y_test_forecast

# %%
# 10. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
for i, (y_pred, label) in enumerate(zip(all_test_preds, labels)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1]})
    
    # Left plot: Time series forecast (48 hours = 2 days) - rectangular
    n_hours = 48
    ax1.plot(y_test_plot[:n_hours], label='Actual', linewidth=2, color='black')
    ax1.plot(y_pred[:n_hours], label=f'Predicted', alpha=0.7, color='blue')
    ax1.set_title('Operational Forecast - Staged Bayesian Optimization + CV (First 2 Days from 2024-01-01)')
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
# 11. Daily Metrics Table (7-Day Forecast Period Only)
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

print("Daily Metrics Table (7-Day Forecast Period):")
print("=" * 80)
for model in labels:
    model_data = daily_metrics_df[daily_metrics_df['Model'] == model]
    print(f"\\n{model}:")
    print("-" * 60)
    print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
    print("-" * 60)
    for _, row in model_data.iterrows():
        print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

print("\\n" + "=" * 80)

# %%
# 12. Feature Importance Analysis
print("\\nAnalyzing feature importance for staged optimized model with cross validation...")

# Get feature importances
feature_importance_cv = pd.DataFrame({
    'feature': selected_features_cv,
    'importance': best_model_cv.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
for i, (_, row) in enumerate(feature_importance_cv.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")

# Plot feature importance with color coding (same scheme as testF.py)
plt.figure(figsize=(12, 6))
top_features = feature_importance_cv.head(8)

# Color code by feature type (same as testF.py)
colors = []
for feat in top_features['feature']:
    if any(t in feat for t in ['hour', 'month', 'day']):
        colors.append('green')   # Time features
    else:
        colors.append('blue')    # Target lags

plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 8 Feature Importances - Staged Bayesian Optimization + Cross Validation Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# %%