# %%
"""
SARIMA Baseline Model for PV Forecasting
========================================

This notebook implements and evaluates a SARIMA baseline model.
- Grid search for optimal parameters
- Rolling origin forecasts
- Model diagnostics and evaluation

Run as: python 02_baseline_sarima.py
"""

# %%
import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_io import load_config, load_and_process_data, create_time_splits
from models.arima import SARIMAForecaster
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')

# %%
def setup_directories():
    """Create necessary directories."""
    os.makedirs('../reports', exist_ok=True)
    os.makedirs('../reports/figures', exist_ok=True)
    os.makedirs('../models', exist_ok=True)

setup_directories()

# %%
"""
DATA LOADING AND PREPARATION
============================
"""

# Load configuration and data
config = load_config('../src/config.yaml')
df = load_and_process_data(config)

# Create time splits
train_df, val_df, test_df = create_time_splits(df, config)

# Extract target variable (electricity)
target_col = config['data']['target_column']
train_target = train_df[target_col]
val_target = val_df[target_col]
test_target = test_df[target_col]

print(f"SARIMA Baseline Model Training")
print(f"Training data: {len(train_target)} samples")
print(f"Validation data: {len(val_target)} samples")
print(f"Test data: {len(test_target)} samples")

# %%
"""
INITIAL DATA EXPLORATION FOR SARIMA
====================================
"""

# Plot target variable
plt.figure(figsize=(15, 6))
plt.plot(train_target.index, train_target, alpha=0.7, linewidth=0.5, label='Training')
plt.plot(val_target.index, val_target, alpha=0.7, linewidth=0.5, label='Validation')
plt.plot(test_target.index, test_target, alpha=0.7, linewidth=0.5, label='Test')
plt.title('PV Electricity Generation Time Series')
plt.xlabel('Date')
plt.ylabel('Electricity (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/sarima_target_series.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Analyze stationarity and seasonality
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries, title):
    """Check stationarity using Augmented Dickey-Fuller test."""
    result = adfuller(timeseries.dropna())
    print(f'\n{title}')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("Result: Stationary")
    else:
        print("Result: Non-stationary")

# Check stationarity of original series
check_stationarity(train_target, "Original Series Stationarity Test")

# Check stationarity of first difference
train_diff = train_target.diff().dropna()
check_stationarity(train_diff, "First Difference Stationarity Test")

# %%
"""
SARIMA MODEL FITTING AND PARAMETER TUNING
==========================================
"""

# Initialize SARIMA forecaster
sarima_forecaster = SARIMAForecaster(config)

# Fit model with grid search on validation set
print("\nFitting SARIMA model with grid search...")
sarima_forecaster.fit(train_target, val_target)

# Get best parameters
best_params = sarima_forecaster.best_params
print(f"\nBest SARIMA parameters:")
print(f"Order (p,d,q): {best_params[0]}")
print(f"Seasonal order (P,D,Q,s): {best_params[1]}")

# %%
"""
MODEL DIAGNOSTICS
=================
"""

# Model summary
print("\nModel Summary:")
print(sarima_forecaster.get_model_summary())

# %%
# Residual diagnostics
residual_diagnostics = sarima_forecaster.diagnose_residuals(train_target)
print(f"\nResidual Diagnostics:")
for key, value in residual_diagnostics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

# %%
# Plot residuals
residuals = sarima_forecaster.best_model.resid

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals over time
axes[0, 0].plot(residuals.index, residuals, alpha=0.7, linewidth=0.5)
axes[0, 0].set_title('Residuals over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True, alpha=0.3)

# Residuals histogram
axes[0, 1].hist(residuals, bins=50, alpha=0.7, density=True)
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(True, alpha=0.3)

# ACF of residuals
from statsmodels.tsa.stattools import acf
residual_acf = acf(residuals, nlags=40)
axes[1, 1].plot(range(len(residual_acf)), residual_acf, 'bo-', markersize=3)
axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_title('ACF of Residuals')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/sarima_residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
VALIDATION SET EVALUATION
=========================
"""

# Generate forecast for validation period
val_forecast = sarima_forecaster.forecast(len(val_target))
val_results = pd.DataFrame({
    'actual': val_target.values,
    'forecast': val_forecast
}, index=val_target.index)

# Evaluate validation performance
val_metrics = sarima_forecaster.evaluate_forecast(val_results)
print(f"\nValidation Set Performance:")
for key, value in val_metrics.items():
    if isinstance(value, float):
        print(f"{key.upper()}: {value:.4f}")
    else:
        print(f"{key.upper()}: {value}")

# %%
# Plot validation results
plt.figure(figsize=(15, 6))
plt.plot(val_results.index, val_results['actual'], label='Actual', alpha=0.8)
plt.plot(val_results.index, val_results['forecast'], label='SARIMA Forecast', alpha=0.8)
plt.title('SARIMA Validation Set Performance')
plt.xlabel('Date')
plt.ylabel('Electricity (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/sarima_validation_results.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
ROLLING ORIGIN FORECASTS
=========================
"""

# Combine training and validation for rolling forecasts
train_val_data = pd.concat([train_target, val_target])

# Perform rolling forecasts (start from validation period)
print("\nPerforming rolling origin forecasts...")
rolling_results = sarima_forecaster.rolling_forecast(
    data=train_val_data,
    forecast_horizon=24,  # 24-hour ahead forecasts
    start_date=val_target.index[0].strftime('%Y-%m-%d')
)

print(f"Generated {len(rolling_results)} rolling forecasts")

# %%
# Evaluate rolling forecast performance
rolling_metrics = sarima_forecaster.evaluate_forecast(rolling_results)
print(f"\nRolling Forecast Performance:")
for key, value in rolling_metrics.items():
    if isinstance(value, float):
        print(f"{key.upper()}: {value:.4f}")
    else:
        print(f"{key.upper()}: {value}")

# %%
# Plot rolling forecast results (sample period)
sample_period = rolling_results.iloc[:24*7]  # First week

plt.figure(figsize=(15, 6))
plt.plot(sample_period.index, sample_period['actual'], label='Actual', alpha=0.8, linewidth=2)
plt.plot(sample_period.index, sample_period['forecast'], label='SARIMA Forecast', alpha=0.8, linewidth=2)
plt.title('SARIMA Rolling Forecast Performance (First Week)')
plt.xlabel('Date')
plt.ylabel('Electricity (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/sarima_rolling_sample.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
FORECAST ACCURACY BY HOUR AND DAY
==================================
"""

# Add time features to rolling results
rolling_results_with_time = rolling_results.copy()
rolling_results_with_time['hour'] = rolling_results_with_time.index.hour
rolling_results_with_time['day_of_week'] = rolling_results_with_time.index.dayofweek
rolling_results_with_time['error'] = rolling_results_with_time['actual'] - rolling_results_with_time['forecast']
rolling_results_with_time['abs_error'] = np.abs(rolling_results_with_time['error'])

# %%
# MAE by hour of day
hourly_mae = rolling_results_with_time.groupby('hour')['abs_error'].mean()

plt.figure(figsize=(12, 6))
hourly_mae.plot(kind='bar', color='skyblue')
plt.title('SARIMA Forecast MAE by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/sarima_hourly_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# MAE by day of week
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_mae = rolling_results_with_time.groupby('day_of_week')['abs_error'].mean()
daily_mae.index = dow_names

plt.figure(figsize=(10, 6))
daily_mae.plot(kind='bar', color='lightcoral')
plt.title('SARIMA Forecast MAE by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/sarima_daily_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
FORECAST HORIZON ANALYSIS
=========================
"""

# Analyze forecast accuracy degradation over different horizons
print("\nAnalyzing forecast accuracy over different horizons...")

horizons = [24, 24*7, 24*30]  # 1 day, 1 week, 1 month
horizon_results = {}

for horizon in horizons:
    print(f"\nEvaluating {horizon}-hour ahead forecasts...")
    
    # Perform rolling forecasts with different horizons
    horizon_forecasts = sarima_forecaster.rolling_forecast(
        data=train_val_data,
        forecast_horizon=horizon,
        start_date=val_target.index[0].strftime('%Y-%m-%d')
    )
    
    # Evaluate performance
    horizon_metrics = sarima_forecaster.evaluate_forecast(horizon_forecasts)
    horizon_results[f"{horizon}h"] = horizon_metrics
    
    print(f"MAE: {horizon_metrics['mae']:.4f}")
    print(f"RMSE: {horizon_metrics['rmse']:.4f}")

# %%
# Plot forecast accuracy degradation
horizon_labels = list(horizon_results.keys())
mae_values = [horizon_results[h]['mae'] for h in horizon_labels]
rmse_values = [horizon_results[h]['rmse'] for h in horizon_labels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(horizon_labels, mae_values, 'bo-', linewidth=2, markersize=8)
ax1.set_title('MAE vs Forecast Horizon')
ax1.set_xlabel('Forecast Horizon')
ax1.set_ylabel('Mean Absolute Error')
ax1.grid(True, alpha=0.3)

ax2.plot(horizon_labels, rmse_values, 'ro-', linewidth=2, markersize=8)
ax2.set_title('RMSE vs Forecast Horizon')
ax2.set_xlabel('Forecast Horizon')
ax2.set_ylabel('Root Mean Square Error')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/sarima_horizon_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
SAVE RESULTS
============
"""

# Save model results
model_results = {
    'model_type': 'SARIMA',
    'best_params': {
        'order': best_params[0],
        'seasonal_order': best_params[1]
    },
    'validation_metrics': val_metrics,
    'rolling_metrics': rolling_metrics,
    'horizon_analysis': horizon_results,
    'residual_diagnostics': residual_diagnostics
}

# Save to JSON
import json
with open('../reports/sarima_results.json', 'w') as f:
    json.dump(model_results, f, indent=2, default=str)

print(f"\nSARIMA Baseline Model Complete!")
print(f"Results saved to:")
print(f"- ../reports/figures/ (visualization plots)")
print(f"- ../reports/sarima_results.json (model results)")

# Summary
print(f"\n" + "="*50)
print(f"SARIMA MODEL SUMMARY")
print(f"="*50)
print(f"Best Parameters: SARIMA{best_params[0]}x{best_params[1]}")
print(f"Validation MAE: {val_metrics['mae']:.4f}")
print(f"Rolling Forecast MAE: {rolling_metrics['mae']:.4f}")
print(f"Residual Test: {residual_diagnostics.get('autocorrelation_test', 'N/A')}")

# %% 