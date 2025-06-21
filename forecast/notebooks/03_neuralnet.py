# %%
"""
Neural Network Model for PV Forecasting
=======================================

This notebook implements and evaluates a neural network model.
- Feature engineering with weather data
- Hyperparameter search with Keras Tuner
- Manual training mode
- Model evaluation and comparison

Run as: python 03_neural_net.py
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
from features import FeatureEngineer
from models.nn import train_nn, forecast_nn, evaluate_nn_forecast
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
DATA LOADING AND FEATURE ENGINEERING
====================================
"""

# Load configuration and data
config = load_config('../src/config.yaml')
df = load_and_process_data(config)

# Create time splits
train_df, val_df, test_df = create_time_splits(df, config)

print(f"Neural Network Model Training")
print(f"Training data: {len(train_df)} samples")
print(f"Validation data: {len(val_df)} samples") 
print(f"Test data: {len(test_df)} samples")

# %%
# Feature engineering
feature_engineer = FeatureEngineer(config)

# Create features for training (with weather)
print("\nCreating features with weather data...")
X_train, y_train = feature_engineer.make_features(train_df, use_weather=True)

# Create features for validation (using fitted scalers)
X_val, y_val = feature_engineer.make_features(val_df, use_weather=True)

# Create features for test (using fitted scalers)
X_test, y_test = feature_engineer.make_features(test_df, use_weather=True)

print(f"Feature matrices created:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# %%
# Display feature information
print(f"\nFeature columns ({len(X_train.columns)}):")
feature_types = {
    'lag': [col for col in X_train.columns if 'lag' in col],
    'cyclical': [col for col in X_train.columns if any(x in col for x in ['sin', 'cos'])],
    'weather': [col for col in X_train.columns if any(x in col for x in ['irradiance', 'temperature'])]
}

for ftype, cols in feature_types.items():
    print(f"- {ftype.title()}: {len(cols)} features")

# %%
"""
NEURAL NETWORK TRAINING - MANUAL MODE
=====================================
"""

print("\n" + "="*60)
print("TRAINING NEURAL NETWORK - MANUAL MODE")
print("="*60)

# Train with manual hyperparameters (faster for testing)
manual_results = train_nn(
    config=config,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    search=False  # Manual mode
)

manual_model = manual_results['model']
manual_history = manual_results['history']
manual_params = manual_results['best_params']

print(f"\nManual training completed!")
print(f"Best parameters: {manual_params}")

# %%
# Plot manual training history
if manual_history:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(manual_history['loss'], label='Training Loss')
    ax1.plot(manual_history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss (Manual)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(manual_history['mae'], label='Training MAE')
    ax2.plot(manual_history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE (Manual)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/figures/nn_manual_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
"""
NEURAL NETWORK TRAINING - HYPERPARAMETER SEARCH
===============================================
"""

print("\n" + "="*60)
print("TRAINING NEURAL NETWORK - HYPERPARAMETER SEARCH")
print("="*60)

# Train with hyperparameter search
search_results = train_nn(
    config=config,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    search=True  # Search mode
)

search_model = search_results['model']
search_params = search_results['best_params']
tuner = search_results['tuner']

print(f"\nHyperparameter search completed!")
print(f"Best parameters: {search_params}")

# %%
# Display tuner results
print(f"\nTop 5 hyperparameter combinations:")
tuner.results_summary(num_trials=5)

# %%
"""
MODEL EVALUATION
================
"""

# Helper function to create evaluation sequences
def create_eval_sequences(X, y, forecast_length=24):
    """Create sequences for evaluation."""
    X_seq, y_seq = [], []
    for i in range(len(X) - forecast_length + 1):
        X_seq.append(X.iloc[i].values)
        y_seq.append(y.iloc[i:i + forecast_length].values)
        if len(y_seq[-1]) != forecast_length:
            X_seq.pop()
            y_seq.pop()
    return np.array(X_seq), np.array(y_seq)

# %%
# Evaluate on validation set
print("\n" + "="*60)
print("VALIDATION SET EVALUATION")
print("="*60)

X_val_seq, y_val_seq = create_eval_sequences(X_val, y_val)

# Manual model predictions
manual_pred = forecast_nn(manual_model, X_val_seq)
manual_metrics = evaluate_nn_forecast(y_val_seq, manual_pred)

# Search model predictions
search_pred = forecast_nn(search_model, X_val_seq)
search_metrics = evaluate_nn_forecast(y_val_seq, search_pred)

print(f"\nValidation Results:")
print(f"Manual Model:")
for key, value in manual_metrics.items():
    if isinstance(value, float):
        print(f"  {key.upper()}: {value:.4f}")

print(f"\nSearch Model:")
for key, value in search_metrics.items():
    if isinstance(value, float):
        print(f"  {key.upper()}: {value:.4f}")

# %%
# Plot validation predictions (first week)
sample_size = min(7 * 24, len(y_val_seq))  # First week
sample_indices = range(sample_size)

# Flatten for plotting
y_val_flat = y_val_seq[:sample_size].flatten()
manual_pred_flat = manual_pred[:sample_size].flatten()
search_pred_flat = search_pred[:sample_size].flatten()

plt.figure(figsize=(15, 8))
plt.plot(y_val_flat, label='Actual', alpha=0.8, linewidth=2)
plt.plot(manual_pred_flat, label='Manual NN', alpha=0.8, linewidth=2)
plt.plot(search_pred_flat, label='Tuned NN', alpha=0.8, linewidth=2)
plt.title('Neural Network Validation Predictions (First Week)')
plt.xlabel('Hours')
plt.ylabel('Electricity (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/nn_validation_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
FORECAST ACCURACY ANALYSIS
==========================
"""

# Use the better performing model for detailed analysis
better_model = search_model if search_metrics['mae'] < manual_metrics['mae'] else manual_model
better_pred = search_pred if search_metrics['mae'] < manual_metrics['mae'] else manual_pred
better_name = 'Tuned NN' if search_metrics['mae'] < manual_metrics['mae'] else 'Manual NN'

print(f"\nUsing {better_name} for detailed analysis")

# %%
# MAE by forecast horizon (1-24 hours ahead)
horizon_mae = []
for h in range(24):
    h_true = y_val_seq[:, h]
    h_pred = better_pred[:, h]
    h_mae = np.mean(np.abs(h_true - h_pred))
    horizon_mae.append(h_mae)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 25), horizon_mae, 'bo-', linewidth=2, markersize=6)
plt.title(f'{better_name}: MAE by Forecast Horizon')
plt.xlabel('Hours Ahead')
plt.ylabel('Mean Absolute Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/nn_horizon_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Create time-based analysis
val_dates = X_val.index[:len(y_val_seq)]
val_results_df = pd.DataFrame({
    'hour': val_dates.hour,
    'day_of_week': val_dates.dayofweek,
    'actual': y_val_seq[:, 0],  # First hour of each forecast
    'predicted': better_pred[:, 0],
    'error': y_val_seq[:, 0] - better_pred[:, 0],
    'abs_error': np.abs(y_val_seq[:, 0] - better_pred[:, 0])
})

# %%
# MAE by hour of day
hourly_mae = val_results_df.groupby('hour')['abs_error'].mean()

plt.figure(figsize=(12, 6))
hourly_mae.plot(kind='bar', color='skyblue')
plt.title(f'{better_name}: MAE by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/nn_hourly_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# MAE by day of week
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_mae = val_results_df.groupby('day_of_week')['abs_error'].mean()
daily_mae.index = dow_names

plt.figure(figsize=(10, 6))
daily_mae.plot(kind='bar', color='lightcoral')
plt.title(f'{better_name}: MAE by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/nn_daily_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
FEATURE IMPORTANCE ANALYSIS
============================
"""

# Simple feature importance using permutation
def permutation_importance(model, X, y, n_repeats=5):
    """Calculate permutation importance for neural network."""
    baseline_score = evaluate_nn_forecast(y, forecast_nn(model, X))['mae']
    
    importances = {}
    for i, feature in enumerate(X_columns):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # Permute feature column
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            score = evaluate_nn_forecast(y, forecast_nn(model, X_perm))['mae']
            scores.append(score - baseline_score)
        importances[feature] = np.mean(scores)
    
    return importances

# %%
print(f"\nCalculating feature importance...")
X_columns = X_val.columns
importance_scores = permutation_importance(better_model, X_val_seq, y_val_seq)

# Sort by importance
sorted_importance = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\nTop 10 Most Important Features:")
for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score:.6f}")

# %%
# Plot feature importance
top_features = dict(sorted_importance[:15])
feature_names = list(top_features.keys())
importance_values = list(top_features.values())

plt.figure(figsize=(12, 8))
colors = ['red' if x > 0 else 'blue' for x in importance_values]
plt.barh(feature_names, importance_values, color=colors, alpha=0.7)
plt.title(f'{better_name}: Top 15 Feature Importances')
plt.xlabel('Permutation Importance (Δ MAE)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/nn_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
SAVE RESULTS
============
"""

# Save model results
nn_results = {
    'model_type': 'Neural Network',
    'manual_params': manual_params,
    'search_params': search_params,
    'manual_metrics': manual_metrics,
    'search_metrics': search_metrics,
    'best_model': better_name,
    'feature_importance': dict(sorted_importance[:20]),
    'horizon_analysis': {f'hour_{h+1}': mae for h, mae in enumerate(horizon_mae)}
}

# Save to JSON
import json
with open('../reports/nn_results.json', 'w') as f:
    json.dump(nn_results, f, indent=2, default=str)

print(f"\nNeural Network Model Complete!")
print(f"Results saved to:")
print(f"- ../reports/figures/ (visualization plots)")
print(f"- ../reports/nn_results.json (model results)")
print(f"- ../models/pv_forecasting_tuner/ (tuner results)")

# Summary
print(f"\n" + "="*60)
print(f"NEURAL NETWORK MODEL SUMMARY")
print(f"="*60)
print(f"Manual Model - MAE: {manual_metrics['mae']:.4f}, RMSE: {manual_metrics['rmse']:.4f}")
print(f"Tuned Model  - MAE: {search_metrics['mae']:.4f}, RMSE: {search_metrics['rmse']:.4f}")
print(f"Best Model: {better_name}")
print(f"Number of features: {len(X_train.columns)}")

# %% 