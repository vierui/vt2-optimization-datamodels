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

# Check if test_df has any data after adding lags
if test_df.empty:
    print("ERROR: test_df is empty after processing!")
    print("This might be due to insufficient data for the lag buffer.")
else:
    # For test_df, identify forecast period (excluding lag buffer)
    test_df['is_forecast_period'] = test_df['time'] >= forecast_start
    forecast_mask = test_df['is_forecast_period'].values
    
    print(f"\nAfter adding lags and removing NaN:")
    print(f"Test data: {test_df.shape[0]} hours")
    print(f"Forecast period: {forecast_mask.sum()} hours (actual evaluation period)")
    print(f"Lag buffer: {(~forecast_mask).sum()} hours (for lag features only)")
    
    if forecast_mask.sum() == 0:
        print("ERROR: No forecast period found! Check date ranges and data availability.")

# %%
# 5. Define Feature Sets - Set H Only (Operational)
# target already defined above

# Set I: Recursive Lag/Pattern-Based Features (NO EXOGENOUS/WEATHER DATA)
# Focus on recursive operational forecasting with available information only
time_features = ['hour', 'month', 'dayofweek', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']

# Target lag features only (electricity past values)
target_lag_cols = [col for col in train_df.columns if col.startswith('electricity_lag')]

# Set I: Time features + Target lags only
set_i_features = time_features + target_lag_cols

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
# 7. Prepare data for Set I (Recursive Lag/Pattern-Based Modeling)
print("Preparing Set I: Recursive Lag/Pattern-Based Features for operational forecasting...")
X_train_i, y_train, available_features_i = prepare_set(train_df, set_i_features, target)
X_valid_i, y_valid, _ = prepare_set(valid_df, set_i_features, target)
X_test_i, y_test, _ = prepare_set(test_df, set_i_features, target)

# Extract forecast period data for evaluation
X_test_forecast = X_test_i[forecast_mask]
y_test_forecast = y_test[forecast_mask]

print(f"Set I available features: {len(available_features_i)}")
print(f"Training data shape: {X_train_i.shape}")
print(f"Test data shape (full): {X_test_i.shape}")
print(f"Forecast period shape: {X_test_forecast.shape}")
print(f"Features: {available_features_i}")

# %%
# 8. Bayesian Optimization for Set I (Recursive Lag/Pattern-Based Modeling)
print("\nRunning Bayesian optimization for Set I (Recursive Lag/Pattern-Based)...")

# Define search space for hyperparameter tuning
dimensions_i = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 12, name='max_depth'),
    Real(0.1, 1.0, name='subsample'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf'),
]

# Combine train and validation for time series CV
X_train_valid = pd.concat([pd.DataFrame(X_train_i, columns=available_features_i), 
                          pd.DataFrame(X_valid_i, columns=available_features_i)]).values
y_train_valid = np.concatenate([y_train, y_valid])

@use_named_args(dimensions_i)
def objective_i(n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf):
    # Model training with Set I features and TimeSeriesSplit CV
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train_valid, y_train_valid, cv=tscv, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # Return negative of mean score (since cross_val_score returns negative MAE)
    return -scores.mean()

result_i = gp_minimize(objective_i, dimensions_i, n_calls=50, random_state=42, verbose=False)
best_params_i = {
    'n_estimators': result_i.x[0],
    'learning_rate': result_i.x[1],
    'max_depth': result_i.x[2],
    'subsample': result_i.x[3],
    'min_samples_split': result_i.x[4],
    'min_samples_leaf': result_i.x[5]
}

print(f"Set I - Best parameters:")
for param, value in best_params_i.items():
    print(f"  {param}: {value}")
print(f"  Best CV MAE: {result_i.fun:.3f}")

# Train Set I model on full train+valid data for recursive prediction
model_i = GradientBoostingRegressor(random_state=42, **best_params_i)
model_i.fit(X_train_valid, y_train_valid)

# Recursive Prediction for 48 hours only
print("\nPerforming recursive prediction for 48 hours...")

# Regular validation prediction (non-recursive)
y_valid_pred_i = model_i.predict(X_valid_i)

# Only proceed with recursive prediction if we have valid forecast data
if test_df.empty or forecast_mask.sum() == 0:
    print("ERROR: Cannot perform recursive prediction - no valid forecast period data!")
    print("Check your data ranges and make sure forecast_start date has available data.")
else:
    print(f"Valid forecast period found: {forecast_mask.sum()} hours")

# Recursive prediction for test period (48 hours only)
def recursive_predict_48h(model, test_df, forecast_mask, available_features, lags):
    """
    Recursively predict 48 hours starting from forecast_start.
    Use actual values when available, own predictions when needed.
    """
    forecast_df = test_df[forecast_mask].copy()
    n_forecast_hours = min(48, len(forecast_df))  # Only predict 48 hours
    
    # Initialize predictions array
    predictions = np.zeros(n_forecast_hours)
    
    # Get the starting point (first forecast hour)
    forecast_indices = np.where(forecast_mask)[0]
    if len(forecast_indices) == 0:
        raise ValueError("No forecast period found in forecast_mask")
    forecast_start_idx = forecast_indices[0]
    
    print(f"Starting recursive prediction for {n_forecast_hours} hours")
    print(f"Forecast start index in test_df: {forecast_start_idx}")
    print(f"Test_df shape: {test_df.shape}")
    print(f"Available features: {len(available_features)}")
    print(f"Max lag: {max(lags)}")
    
    for h in range(n_forecast_hours):
        # Current hour index in test_df
        current_idx = forecast_start_idx + h
        
        # Construct features for this hour
        X_current = construct_features_for_hour(
            test_df, current_idx, available_features, lags, predictions, h
        )
        
        # Debug: Check for NaN values
        if np.any(np.isnan(X_current)):
            print(f"Warning: NaN values found in features for hour {h}")
            print(f"Features with NaN: {np.where(np.isnan(X_current))[0]}")
            print(f"Feature values: {X_current}")
            # Replace NaN with 0 as fallback
            X_current = np.nan_to_num(X_current, nan=0.0)
        
        # Make prediction for this hour
        pred = model.predict(X_current.reshape(1, -1))[0]
        predictions[h] = pred
        
        if h % 12 == 0:  # Print progress every 12 hours
            print(f"  Hour {h}: predicted {pred:.2f} kW")
    
    return predictions

def construct_features_for_hour(test_df, current_idx, available_features, lags, predictions, pred_hour):
    """
    Construct feature vector for a specific hour during recursive prediction.
    Use actual values when available, predictions when needed for lags.
    """
    current_row = test_df.iloc[current_idx].copy()
    feature_values = []
    
    for feature in available_features:
        if feature.endswith('_sin') or feature.endswith('_cos') or feature in ['hour', 'month', 'dayofweek', 'dayofyear']:
            # Time features - use actual values
            feature_values.append(current_row[feature])
        elif feature.startswith('electricity_lag'):
            # Extract lag number
            lag = int(feature.split('lag')[1])
            
            # Calculate which index we need in test_df
            needed_idx = current_idx - lag
            
            # Determine the forecast start index in test_df
            forecast_start_test_idx = current_idx - pred_hour
            
            if needed_idx < 0:
                # Before available data - use 0 as safe fallback
                feature_values.append(0.0)
            elif needed_idx < forecast_start_test_idx:
                # This is historical data (before predictions started) - use actual
                feature_values.append(test_df.iloc[needed_idx]['electricity'])
            else:
                # This is within our prediction period - use our prediction
                # Calculate which prediction index we need
                pred_idx = needed_idx - forecast_start_test_idx
                if pred_idx >= 0 and pred_idx < len(predictions):
                    feature_values.append(predictions[pred_idx])
                else:
                    # Safety fallback - use actual value if somehow available
                    if needed_idx >= 0 and needed_idx < len(test_df):
                        feature_values.append(test_df.iloc[needed_idx]['electricity'])
                    else:
                        feature_values.append(0.0)
        else:
            # Other features - use actual values
            feature_values.append(current_row[feature])
    
    # Check for any remaining NaN values and replace with 0
    feature_array = np.array(feature_values)
    feature_array = np.nan_to_num(feature_array, nan=0.0)
    
    return feature_array

# Perform recursive prediction only if we have valid data
if not test_df.empty and forecast_mask.sum() > 0:
    y_test_pred_i_48h = recursive_predict_48h(
        model_i, test_df, forecast_mask, available_features_i, lags
    )
    
    # For comparison, also do regular prediction on first 48 hours
    y_test_pred_regular = model_i.predict(X_test_forecast[:48])
    
    # Calculate metrics for validation and recursive test (48h only)
    metrics_i = {}
    y_test_48h = y_test_forecast[:48]  # First 48 hours of actual values
    
    for split, y_true, y_pred in [
        ("Valid", y_valid, y_valid_pred_i),
        ("Test_Recursive", y_test_48h, y_test_pred_i_48h),
        ("Test_Regular", y_test_48h, y_test_pred_regular)
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics_i[split] = (rmse, mae, r2)
        print(f"I: {split} - RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
    
    print("-" * 60)
    print(f"\nRecursive vs Regular Prediction Comparison (48h):")
    print(f"Recursive MAE: {metrics_i['Test_Recursive'][1]:.3f}")
    print(f"Regular MAE:   {metrics_i['Test_Regular'][1]:.3f}")
    print(f"Difference:    {metrics_i['Test_Recursive'][1] - metrics_i['Test_Regular'][1]:.3f}")
else:
    print("Skipping recursive prediction due to data issues.")

# %%
# 9. Feature Importance Analysis for Set I
print("\nAnalyzing feature importance for Set I...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': available_features_i,
    'importance': model_i.feature_importances_
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
plt.title('Top 15 Feature Importances - Set I (Recursive Lag/Pattern-Based)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%
# 10. Results for Set I - Recursive vs Regular Comparison (48h)
if not test_df.empty and forecast_mask.sum() > 0:
    all_test_preds = [y_test_pred_i_48h, y_test_pred_regular]  # Recursive and regular predictions
    all_valid_preds = [y_valid_pred_i, y_valid_pred_i]  # Same validation for both
    labels = ["I: Recursive", "I: Regular"]
    metrics_dict = {
        "I: Recursive": metrics_i["Test_Recursive"], 
        "I: Regular": metrics_i["Test_Regular"]
    }
    
    # Use 48h targets for plotting
    y_test_plot = y_test_48h
else:
    # Fallback for when recursive prediction fails
    print("Using validation data only for plotting (no test predictions available)")
    all_test_preds = [y_valid_pred_i[:48]]  # Use first 48 hours of validation
    all_valid_preds = [y_valid_pred_i]
    labels = ["I: Validation Only"]
    metrics_dict = {"I: Validation Only": (np.sqrt(mean_squared_error(y_valid[:48], y_valid_pred_i[:48])),
                                           mean_absolute_error(y_valid[:48], y_valid_pred_i[:48]),
                                           r2_score(y_valid[:48], y_valid_pred_i[:48]))}
    y_test_plot = y_valid[:48]

# %%
# 11. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
for i, (y_pred, label) in enumerate(zip(all_test_preds, labels)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1]})
    
    # Left plot: Time series forecast (48 hours = 2 days) - rectangular
    n_hours = 48
    colors = ['blue', 'red']
    ax1.plot(y_test_plot[:n_hours], label='Actual', linewidth=2, color='black')
    ax1.plot(y_pred[:n_hours], label=f'{label}', alpha=0.7, color=colors[i])
    ax1.set_title('Recursive vs Regular Forecast - Set I (48 Hours from 2024-01-01)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Electricity (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Predicted vs Actual scatter - square with same height
    ax2.scatter(y_test_plot, y_pred, s=10, alpha=0.5, color=colors[i])
    ax2.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'k--', lw=2)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title(f'{label} - Predicted vs Actual')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Display metrics
    test_metrics = metrics_dict[label]
    rmse, mae, r2 = test_metrics
    print(f"{label} - Test Metrics (48h):")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.3f}")
    print("-" * 50)

# %%
# 12. Daily Metrics Table (48h Period Only - First 2 Days)
def calculate_daily_metrics_48h(y_test_48h, test_preds, labels):
    # Create timestamps for 48 hours starting from 2024-01-01
    start_time = pd.Timestamp('2024-01-01')
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(48)]
    dates = [ts.date() for ts in timestamps]
    unique_dates = sorted(list(set(dates)))
    
    daily_metrics = []
    
    for label, y_pred in zip(labels, test_preds):
        for date in unique_dates:
            # Find indices for this date
            day_indices = [i for i, d in enumerate(dates) if d == date]
            
            if len(day_indices) > 0:
                y_true_day = y_test_48h[day_indices]
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

daily_metrics_df = calculate_daily_metrics_48h(y_test_plot, all_test_preds, labels)

print("Daily Metrics Table (48h Recursive vs Regular):")
print("=" * 80)
for model in labels:
    model_data = daily_metrics_df[daily_metrics_df['Model'] == model]
    print(f"\n{model}:")
    print("-" * 60)
    print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
    print("-" * 60)
    for _, row in model_data.iterrows():
        print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

print(f"\nRecursive Prediction Summary:")
print(f"- Predicted 48 hours recursively from 2024-01-01")
print(f"- Used actual lag values when available")
print(f"- Used own predictions for recent lags")
print(f"- Compared against regular (non-recursive) prediction")
print(f"- Recursive MAE: {metrics_i['Test_Recursive'][1]:.3f}")
print(f"- Regular MAE: {metrics_i['Test_Regular'][1]:.3f}")
print("\n" + "=" * 80)
# %%
