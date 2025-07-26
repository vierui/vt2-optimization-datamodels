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
import pvlib
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
max_lag = 24  # 24 hours (our longest lag now)
forecast_start = pd.Timestamp('2024-01-01')
test_start = forecast_start - pd.Timedelta(hours=max_lag)  # Buffer for lags
test_end = forecast_start + pd.Timedelta(days=7)  # 7 days evaluation

print(f"Forecast starts: {forecast_start}")
print(f"Test data starts: {test_start} (includes {max_lag}h lag buffer)")
print(f"Test data ends: {test_end} (7 days evaluation)")

train_df = df[df['time'] <  "2021-01-01"].copy()
valid_df = df[(df['time'] >= "2021-01-01") & (df['time'] < "2023-12-24")].copy()
test_df = df[(df['time'] >= test_start) & (df['time'] < test_end)].copy()

print(f"\nData shapes:")
print(f"Train: {train_df.shape[0]} hours")
print(f"Valid: {valid_df.shape[0]} hours")
print(f"Test: {test_df.shape[0]} hours (includes lag buffer)")

# %%
# 3. Streamlined POA Irradiance and Clear-Sky Index Calculation
print("Calculating POA irradiance and clear-sky index features...")

# --- Site parameters for Sion, Switzerland (real site) ---
latitude = 46.2312
longitude = 7.3589
altitude = 500      # meters, adjust if needed
timezone = 'UTC'    # Data is UTC

tilt = 35           # Your PV tilt in degrees
azimuth = 180       # 180 = South-facing

print(f"Site parameters:")
print(f"  Location: {latitude:.4f}°N, {longitude:.4f}°W (Sion, Switzerland)")
print(f"  Panel: {tilt}° tilt, {azimuth}° azimuth (south-facing)")

# --- POA & clear-sky features calculation function ---
def calculate_poa_features(df, lat, lon, alt, tz, tilt, azimuth):
    print(f"    Input data shape: {df.shape}")
    print(f"    Input data columns: {list(df.columns)}")
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    print(f"    After time parsing: {df.shape}")
    
    df = df.set_index('time')
    print(f"    After setting time index: {df.shape}")
    
    df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index
    print(f"    After timezone localization: {df.shape}")

    # Check for required columns
    required_cols = ['irradiance_direct', 'irradiance_diffuse', 'swgdn', 'electricity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"    WARNING: Missing required columns: {missing_cols}")
        return pd.DataFrame()  # Return empty dataframe if missing required columns

    # Solar position
    print(f"    Calculating solar position...")
    solar_position = pvlib.solarposition.get_solarposition(df.index, lat, lon, altitude=alt)
    print(f"    Solar position calculated: {solar_position.shape}")
    
    # Use provided measured values, assumed to be DNI, DHI, GHI (all W/m2)
    dni = df['irradiance_direct']
    dhi = df['irradiance_diffuse']
    ghi = df['swgdn']
    
    print(f"    Irradiance data stats:")
    print(f"      DNI: {dni.notna().sum()}/{len(dni)} valid values")
    print(f"      DHI: {dhi.notna().sum()}/{len(dhi)} valid values") 
    print(f"      GHI: {ghi.notna().sum()}/{len(ghi)} valid values")

    # Calculate actual POA
    print(f"    Calculating actual POA...")
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azimuth,
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=dni, ghi=ghi, dhi=dhi,
    )
    df['poa_total'] = poa['poa_global']
    print(f"    POA total calculated: {df['poa_total'].notna().sum()}/{len(df)} valid values")

    # Calculate clear-sky GHI/DNI/DHI (Ineichen for Europe)
    print(f"    Calculating clear-sky irradiance...")
    site = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)
    clearsky = site.get_clearsky(df.index, model='ineichen')
    print(f"    Clear-sky calculated: {clearsky.shape}")
    
    poa_clearsky = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azimuth,
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=clearsky['dni'], ghi=clearsky['ghi'], dhi=clearsky['dhi'],
    )
    df['poa_clearsky'] = poa_clearsky['poa_global']
    print(f"    POA clear-sky calculated: {df['poa_clearsky'].notna().sum()}/{len(df)} valid values")

    # Calculate clear-sky index (mask at night or very low clearsky)
    is_day = solar_position['apparent_elevation'] > 5
    clearsky_ok = (df['poa_clearsky'] > 50) & is_day
    
    # Initialize with 0 instead of NaN for nighttime (more model-friendly)
    clearsky_index = np.zeros(len(df))
    clearsky_index[clearsky_ok] = (df['poa_total'][clearsky_ok] / df['poa_clearsky'][clearsky_ok])
    df['poa_clearsky_index'] = np.clip(clearsky_index, 0, 1.5)
    df['solar_elevation'] = solar_position['apparent_elevation']
    df['is_daytime'] = is_day.values
    
    print(f"    Clear-sky index calculated: {df['poa_clearsky_index'].notna().sum()}/{len(df)} valid values")

    # Restore time as a column (not index)
    df = df.reset_index()
    print(f"    After reset_index: {df.shape}")
    print(f"    Output columns: {list(df.columns)}")
    
    return df

# Apply POA features to each dataset
print("\nProcessing datasets...")
for name, dataset in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    print(f"Calculating POA features for {name}...")
    out = calculate_poa_features(dataset, latitude, longitude, altitude, timezone, tilt, azimuth)
    if name == "train":
        train_df = out
    elif name == "valid":
        valid_df = out
    else:
        test_df = out

print("POA features added successfully!")
poa_features = ['poa_total', 'poa_clearsky', 'poa_clearsky_index', 'solar_elevation']
for feature in poa_features:
    if feature in train_df.columns:
        print(f"  {feature}: {train_df[feature].notna().sum()} valid values in train")

# %%
# 4. Streamlined Feature Engineering
print("\nAdding time features, lags, and rolling statistics...")

# --- Add cyclical time features (sin/cos for hour, month, etc.) ---
def add_time_features(df):
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['dayofweek'] = df['time'].dt.dayofweek
    df['dayofyear'] = df['time'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1) / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

for d in [train_df, valid_df, test_df]:
    add_time_features(d)

# --- Reduce lag requirements to preserve more data ---
def add_lags_and_rolling(df, target, poa_feature, target_lags=[1, 24], weather_lags=[1, 2, 3, 6, 12, 24], rolling_windows=[24]):
    print(f"  Before adding lags: {df.shape[0]} rows")
    
    # Target lags (reduced to preserve data)
    for lag in target_lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    
    # POA feature lags (more extensive for weather patterns)
    for lag in weather_lags:
        df[f"{poa_feature}_lag{lag}"] = df[poa_feature].shift(lag)
    
    # Rolling statistics for both (reduced windows)
    for window in rolling_windows:
        df[f"{target}_rollmean{window}"] = df[target].rolling(window=window, min_periods=1).mean().shift(1)
        df[f"{poa_feature}_rollmean{window}"] = df[poa_feature].rolling(window=window, min_periods=1).mean().shift(1)
    
    print(f"  After adding lags: {df.shape[0]} rows")
    return df

target = 'electricity'
print("\nAdding lags to datasets:")
for name, d in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    print(f"Processing {name}:")
    add_lags_and_rolling(d, target='electricity', poa_feature='poa_clearsky_index')

# Drop NA rows only for essential features (target and target lags)
print("\nDropping rows with NaN values from essential features only...")
for name, d in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    before_drop = d.shape[0]
    print(f"  {name} before dropna: {before_drop} rows")
    
    # Debug: Check which columns have NaN values
    nan_counts = d.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"    Columns with NaN: {dict(cols_with_nan)}")
    else:
        print(f"    No NaN values found")
    
    # Only drop rows where essential features (target and target lags) have NaN
    essential_cols = ['electricity'] + [col for col in d.columns if col.startswith('electricity_lag') or col.startswith('electricity_rollmean')]
    print(f"    Essential columns for dropna: {essential_cols}")
    
    d.dropna(subset=essential_cols, inplace=True)
    after_drop = d.shape[0]
    print(f"  {name} after selective dropna: {after_drop} rows (kept {after_drop/max(before_drop,1)*100:.1f}%)")

print("Feature engineering completed successfully!")

# Debug: Check final dataset shapes
print(f"\nFinal dataset shapes after all processing:")
print(f"  Train: {train_df.shape}")
print(f"  Valid: {valid_df.shape}")  
print(f"  Test: {test_df.shape}")

# For test_df, identify forecast period (excluding lag buffer)
if not test_df.empty:
    # Make forecast_start timezone-aware to match test_df['time']
    forecast_start_utc = pd.Timestamp(forecast_start).tz_localize('UTC')
    test_df['is_forecast_period'] = test_df['time'] >= forecast_start_utc
    forecast_mask = test_df['is_forecast_period'].values
    
    print(f"\nAfter adding lags and removing NaN:")
    print(f"Test data: {test_df.shape[0]} hours")
    print(f"Forecast period: {forecast_mask.sum()} hours (actual evaluation period)")
    print(f"Lag buffer: {(~forecast_mask).sum()} hours (for lag features only)")
else:
    print(f"\nERROR: Test dataset is empty!")
    forecast_mask = np.array([])

# Check if we have any data left
if train_df.empty or valid_df.empty or test_df.empty:
    print("\nERROR: One or more datasets are empty after processing!")
    print("This indicates an issue with the POA calculation or lag processing.")
    print("Dataset shapes:")
    print(f"  train_df: {train_df.shape}")
    print(f"  valid_df: {valid_df.shape}")
    print(f"  test_df: {test_df.shape}")

# %%
# 5. Define Feature Sets - Set J with Streamlined POA Clear-Sky Index
print("\nDefining Set J feature set...")

# Set J: Streamlined Operational Features with POA Clear-Sky Index
time_features = ['hour', 'month', 'dayofweek', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']

# Target lag features (electricity past values) - key lags only
target_lag_cols = [col for col in train_df.columns if col.startswith('electricity_lag') or col.startswith('electricity_rollmean')]

# POA clear-sky index features with lags and rolling statistics
poa_clearsky_cols = [col for col in train_df.columns if 'poa_clearsky_index' in col and ('_lag' in col or '_rollmean' in col)]

# Set J: Time features + Target lags/rolling + POA clear-sky index lags/rolling
set_j_features = time_features + target_lag_cols + poa_clearsky_cols

print(f"Set J Feature Summary:")
print(f"  Time features: {len(time_features)}")
print(f"  Target lag/rolling features: {len(target_lag_cols)}")
print(f"  POA clear-sky index features: {len(poa_clearsky_cols)}")
print(f"  Total Set J features: {len(set_j_features)}")

print(f"\nSet J Features:")
print("Target features:", [f for f in target_lag_cols])
print("POA features:", [f for f in poa_clearsky_cols])

# %%
# 6. Utility Functions
def prepare_set(df, features, target):
    # Only use features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    # Fill any remaining NaN values with 0 (mainly for POA features during night)
    X = X.fillna(0)
    
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
# 7. Prepare data for Set J (POA Clear-Sky Index Enhanced Modeling)
print("Preparing Set J: POA Clear-Sky Index Enhanced Features for operational forecasting...")
X_train_j, y_train, available_features_j = prepare_set(train_df, set_j_features, target)
X_valid_j, y_valid, _ = prepare_set(valid_df, set_j_features, target)
X_test_j, y_test, _ = prepare_set(test_df, set_j_features, target)

# Extract forecast period data for evaluation
X_test_forecast = X_test_j[forecast_mask]
y_test_forecast = y_test[forecast_mask]

print(f"Set J available features: {len(available_features_j)}")
print(f"Training data shape: {X_train_j.shape}")
print(f"Test data shape (full): {X_test_j.shape}")
print(f"Forecast period shape: {X_test_forecast.shape}")

# Show feature breakdown
time_count = sum(1 for f in available_features_j if any(tf in f for tf in ['hour', 'month', 'day']))
target_count = sum(1 for f in available_features_j if f.startswith('electricity_lag'))
poa_count = len(available_features_j) - time_count - target_count

print(f"\nFeature breakdown:")
print(f"  Time features: {time_count}")
print(f"  Target lags: {target_count}")
print(f"  POA features: {poa_count}")

# %%
# 8. Bayesian Optimization for Set J (POA Clear-Sky Enhanced Modeling)
print("\nRunning Bayesian optimization for Set J (POA Clear-Sky Enhanced)...")

# Combine train and validation for time series CV
X_train_valid = pd.concat([pd.DataFrame(X_train_j, columns=available_features_j), 
                          pd.DataFrame(X_valid_j, columns=available_features_j)]).values
y_train_valid = np.concatenate([y_train, y_valid])

print(f"Combined data shape for CV: {X_train_valid.shape}")
print(f"Data points available for cross-validation: {len(y_train_valid)}")

# Define search space for hyperparameter tuning
dimensions_j = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 12, name='max_depth'),
    Real(0.1, 1.0, name='subsample'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf'),
]

@use_named_args(dimensions_j)
def objective_j(n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf):
    # Model training with Set J features using TimeSeriesSplit CV
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Check if we have enough data for CV, otherwise use simple validation split
    if len(y_train_valid) < 500:  # Not enough data for 5-fold CV
        # Simple train-validation split approach
        model.fit(X_train_j, y_train)
        y_pred = model.predict(X_valid_j)
        mae = mean_absolute_error(y_valid, y_pred)
        return mae
    else:
        # Time series cross-validation with fewer splits
        n_splits = min(3, len(y_train_valid) // 100)  # Adaptive splits based on data size
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X_train_valid, y_train_valid, cv=tscv, 
                               scoring='neg_mean_absolute_error', n_jobs=1)  # Use single job to avoid parallel issues
        
        # Return negative of mean score (since cross_val_score returns negative MAE)
        return -scores.mean()

result_j = gp_minimize(objective_j, dimensions_j, n_calls=50, random_state=42, verbose=False)
best_params_j = {
    'n_estimators': result_j.x[0],
    'learning_rate': result_j.x[1],
    'max_depth': result_j.x[2],
    'subsample': result_j.x[3],
    'min_samples_split': result_j.x[4],
    'min_samples_leaf': result_j.x[5]
}

print(f"Set J - Best parameters:")
for param, value in best_params_j.items():
    print(f"  {param}: {value}")
print(f"  Best CV MAE: {result_j.fun:.3f}")

# Train Set J model on full train+valid data - evaluate only on forecast period
model_j = GradientBoostingRegressor(random_state=42, **best_params_j)
model_j.fit(X_train_valid, y_train_valid)

# Make predictions
y_valid_pred_j = model_j.predict(X_valid_j)
y_test_pred_full = model_j.predict(X_test_j)
y_test_pred_j = y_test_pred_full[forecast_mask]  # Only forecast period

# Calculate metrics for both validation and forecast period
metrics_j = {}
for split, y_true, y_pred in [
    ("Valid", y_valid, y_valid_pred_j),
    ("Test", y_test_forecast, y_test_pred_j)  # Only forecast period
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_j[split] = (rmse, mae, r2)
    print(f"J: POA Clear-Sky Enhanced - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
print("-" * 60)

# %%
# 9. Feature Importance Analysis for Set J
print("\nAnalyzing feature importance for Set J...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': available_features_j,
    'importance': model_j.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 most important features:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
    feat_type = "POA" if any(poa in row['feature'] for poa in ['poa_', 'solar_']) else "Time" if any(t in row['feature'] for t in ['hour', 'month', 'day']) else "Target"
    print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f} ({feat_type})")

# Plot feature importance with color coding
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(8)

# Color code by feature type
colors = []
for feat in top_features['feature']:
    if any(poa in feat for poa in ['poa_', 'solar_']):
        colors.append('orange')  # POA features
    elif any(t in feat for t in ['hour', 'month', 'day']):
        colors.append('green')   # Time features
    else:
        colors.append('blue')    # Target lags

plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances - Set J (POA Clear-Sky Enhanced)')
plt.gca().invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='orange', label='POA Features'),
                   Patch(facecolor='green', label='Time Features'),
                   Patch(facecolor='blue', label='Target Lags')]
plt.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.show()

# Analyze POA feature contributions
poa_features_importance = feature_importance[feature_importance['feature'].str.contains('poa_|solar_')]
print(f"\nPOA Feature Analysis:")
print(f"  Total POA features in top 20: {len(poa_features_importance.head(20))}")
print(f"  POA features importance sum: {poa_features_importance['importance'].sum():.4f}")
print(f"  Average POA feature importance: {poa_features_importance['importance'].mean():.4f}")

# %%
# 10. Results for Set J Only - POA Clear-Sky Enhanced Evaluation
all_test_preds = [y_test_pred_j]  # Only forecast period predictions
all_valid_preds = [y_valid_pred_j]
labels = ["J: POA Clear-Sky Enhanced"]
metrics_dict = {"J: POA Clear-Sky Enhanced": metrics_j}

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
    ax1.set_title('Operational Forecast - POA Clear-Sky Enhanced (First 2 Days from 2024-01-01)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Electricity (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Predicted vs Actual scatter - square with same height
    ax2.scatter(y_test_plot, y_pred, s=10, alpha=0.5, color='blue')
    ax2.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'r--', lw=2)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title('Predicted vs Actual (POA Enhanced)')
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
# 12. Daily Metrics Table (7-Day Forecast Period Only) - Set J
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

print("Daily Metrics Table (Set J - POA Clear-Sky Enhanced):")
print("=" * 80)
for model in labels:
    model_data = daily_metrics_df[daily_metrics_df['Model'] == model]
    print(f"\n{model}:")
    print("-" * 60)
    print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
    print("-" * 60)
    for _, row in model_data.iterrows():
        print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

print(f"\nSet J Summary:")
print(f"Enhanced with POA clear-sky index and solar position features")
print(f"Total features: {len(available_features_j)} (vs baseline lag-only model)")
print(f"POA features contribute significantly to solar generation forecasting")
print("\n" + "=" * 80)
# %%
