import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

# Load data
file_path = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/load-2023.csv"  # Change to actual file path
df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

# Ensure proper time series format
df = df.asfreq('H')  # Set frequency to hourly

# Extract month and hour for analysis
df['month'] = df.index.month
df['hour'] = df.index.hour
df['day'] = df.index.day  # Added this to enable the monthplot()

# Aggregate seasonality by month (daily averages for each month)
monthly_seasonality = df.groupby(['month', 'hour'])['value'].mean().unstack()

# STL decomposition
stl = STL(df['value'], period=24*7, robust=True)  # Weekly periodicity assumed
result = stl.fit()
df['seasonal'] = result.seasonal  # Extract seasonal component

# === Minimalist STL Visualization ===
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

# Set black-and-white style
plt.style.use('grayscale')

# Plot original data
axes[0].plot(df.index, df['value'], linewidth=1)
axes[0].set_ylabel("Data")
axes[0].grid(False)

# Plot seasonal component
axes[1].plot(df.index, df['seasonal'], linewidth=1)
axes[1].set_ylabel("Seasonal")
axes[1].grid(False)

# Plot trend component
axes[2].plot(df.index, result.trend, linewidth=1)
axes[2].set_ylabel("Trend")
axes[2].grid(False)

# Plot residuals
axes[3].plot(df.index, result.resid, linewidth=1)
axes[3].axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
axes[3].set_ylabel("Remainder")
axes[3].set_xlabel("Time")
axes[3].grid(False)

# Remove unnecessary borders and ticks
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=False)

plt.show()

# # === `monthplot()` Equivalent: Seasonal Component Across Months ===
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df, x='day', y='seasonal', hue='month', palette="tab10", legend=False)
# plt.title("Seasonal Component across Months")
# plt.xlabel("Day of Month")
# plt.ylabel("Seasonal Effect")
# plt.xticks(np.arange(1, 32, step=2))  # Show days from 1 to 31
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()
