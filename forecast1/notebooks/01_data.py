# %%
"""
Data Exploratory Analysis for PV Forecasting System
====================================================

This notebook explores the PV generation time-series data.
Run as: python 01_data_eda.py
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

plt.style.use('seaborn-v0_8')

# %%
def setup_directories():
    """Create necessary directories."""
    os.makedirs('../reports', exist_ok=True)
    os.makedirs('../reports/figures', exist_ok=True)
    os.makedirs('../data', exist_ok=True)

setup_directories()

# %%
# Load configuration and data
config = load_config('../src/config.yaml')
print("Configuration loaded")

# %%
# Load and process data
df = load_and_process_data(config)
print(f"Loaded data: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# %%
# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# %%
# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# %%
# Create time splits
train_df, val_df, test_df = create_time_splits(df, config)

# %%
"""
VISUALIZATION
=============
"""

# Full time series
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].plot(df.index, df[col], alpha=0.7, linewidth=0.5)
    axes[i].axvline(pd.Timestamp('2022-01-01'), color='red', linestyle='--', alpha=0.7)
    axes[i].axvline(pd.Timestamp('2024-01-01'), color='orange', linestyle='--', alpha=0.7)
    axes[i].set_title(col.title())
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.savefig('../reports/figures/full_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Monthly patterns
monthly_stats = df.groupby(df.index.month).mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    monthly_stats[col].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Monthly Average: {col}')
    axes[i].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('../reports/figures/monthly_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Daily patterns
hourly_stats = df.groupby(df.index.hour).mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    hourly_stats[col].plot(ax=axes[i], marker='o')
    axes[i].set_title(f'Daily Pattern: {col}')
    axes[i].set_xlabel('Hour')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/daily_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Correlation matrix
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Variable Correlation Matrix')
plt.tight_layout()
plt.savefig('../reports/figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCorrelation with electricity:")
print(correlation_matrix['electricity'].sort_values(ascending=False))

# %%
# Feature engineering test
feature_engineer = FeatureEngineer(config)
sample_data = train_df.tail(24*7)  # Last week

X_sample, y_sample = feature_engineer.make_features(sample_data, use_weather=True)
print(f"\nFeature engineering test:")
print(f"Input shape: {sample_data.shape}")
print(f"Feature matrix shape: {X_sample.shape}")
print(f"Number of features: {len(X_sample.columns)}")

# %%
print("EDA Complete!")
print("Generated files:")
print("- reports/figures/ (visualization plots)")

# %% 