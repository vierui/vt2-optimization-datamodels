import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. LOAD THE CSV AND PREPARE TIMESTAMP INDEX
df = pd.read_csv(
    '../../data/renewables/pv_with_weather_data.csv',
    comment='#',
    parse_dates=['time'],
    index_col='time'
)
df = df.asfreq('h')  # ensure a regular hourly index (fixed deprecation warning)

# Keep only the PV ("electricity") column as our target
df = df[['electricity']].rename(columns={'electricity': 'pv'})

# 2. OPTIMIZATION: Use a smaller subset for faster processing (last 6 months + test)
train_start = pd.Timestamp('2023-07-01 00:00:00')  # Start from July 2023
train_end = pd.Timestamp('2023-12-31 23:00:00')
test_start = train_end + pd.Timedelta(hours=1)

# Filter to subset for faster processing
df_subset = df.loc[train_start:]
df_train = df_subset.loc[train_start:train_end]
df_test = df_subset.loc[test_start:]

print(f"Using optimized dataset: {len(df_train)} training hours, {len(df_test)} test hours")

# 3. ENCODE "TIME" AS A 1D NUMERIC ARRAY
subset_indices = np.arange(len(df_subset))
train_indices = subset_indices[:len(df_train)]
test_indices = subset_indices[len(df_train):]

t_train = train_indices.reshape(-1, 1)
t_test = test_indices.reshape(-1, 1)

y_train = df_train['pv'].values
y_test = df_test['pv'].values

# 4. CHOOSE A KERNEL: (ExpSineSquared × RBF) + WhiteKernel
k_periodic = ExpSineSquared(length_scale=1.0, periodicity=24.0)
k_rbf = RBF(length_scale=24.0 * 30.0)  # one‐month‐scale drift
k_noise = WhiteKernel(noise_level=1e-2)

kernel = (k_periodic * k_rbf) + k_noise

# 5. INSTANTIATE & FIT THE GP REGRESSOR (OPTIMIZED)
print("Fitting Gaussian Process (optimized for faster execution)...")
gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=3,  # Reduced from 10 for faster execution
    random_state=0
)
gp.fit(t_train, y_train)
print("✓ GP fitting completed!")

# 6. FORECASTING: MAKE PREDICTIONS ON df_test
target_dates = ['2024-01-01', '2024-01-08', '2024-01-31']

# Pre‐allocate a DataFrame to hold MAE/RMSE results
results = []

for date_str in target_dates:
    print(f"Processing forecast for {date_str}...")
    
    # a) Build a 24‐hour range for that date, at hourly resolution
    day_start = pd.Timestamp(date_str + ' 00:00:00')
    day_end = pd.Timestamp(date_str + ' 23:00:00')
    idx_range = pd.date_range(day_start, day_end, freq='h')

    # b) Convert those timestamps to indices in our subset
    try:
        positions = [df_subset.index.get_loc(ts) for ts in idx_range]
    except KeyError:
        print(f"  → {date_str} missing from index, skipping.")
        continue

    t_day = np.array(positions).reshape(-1, 1)
    y_day_actual = df_subset.loc[idx_range, 'pv'].values

    # c) GP prediction: mean + standard deviation
    y_day_pred, y_std = gp.predict(t_day, return_std=True)

    # d) Compute daily metrics: MAE and RMSE for those 24 points
    mae_day = mean_absolute_error(y_day_actual, y_day_pred)
    rmse_day = np.sqrt(mean_squared_error(y_day_actual, y_day_pred))

    results.append({
        'date': date_str,
        'MAE': mae_day,
        'RMSE': rmse_day
    })

    # e) Plot that day's actual vs. predicted + ±1.96σ
    plt.figure(figsize=(10, 4))
    plt.plot(idx_range, y_day_actual, 'k-', linewidth=2, label='Actual PV')
    plt.plot(idx_range, y_day_pred, 'r-', linewidth=2, label='GP Mean')
    upper = y_day_pred + 1.96*y_std
    lower = y_day_pred - 1.96*y_std
    plt.fill_between(idx_range, lower, upper, color='r', alpha=0.2, label='95% CI')
    plt.title(f"GP Forecast on {date_str} (MAE: {mae_day:.3f}, RMSE: {rmse_day:.3f})")
    plt.xlabel('Time (Hourly)')
    plt.ylabel('PV Output')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 7. SUMMARY TABLE OF DAILY ERRORS
summary_df = pd.DataFrame(results)
print("\nDaily Forecast Errors on Key Dates (2024):")
print(summary_df.to_string(index=False))

# 8. FINAL TEST SET METRICS (OPTIMIZED)
print("\nComputing overall test metrics (using sample for speed)...")
# Sample subset if test set is large
if len(t_test) > 1000:
    print(f"Sampling 1000 points from {len(t_test)} test points for faster evaluation")
    sample_indices = np.random.choice(len(t_test), 1000, replace=False)
    t_test_sample = t_test[sample_indices]
    y_test_sample = y_test[sample_indices]
else:
    t_test_sample = t_test
    y_test_sample = y_test

y_test_pred, y_test_std = gp.predict(t_test_sample, return_std=True)
mae_overall = mean_absolute_error(y_test_sample, y_test_pred)
rmse_overall = np.sqrt(mean_squared_error(y_test_sample, y_test_pred))
print(f"\nOverall 2024 Test RMSE: {rmse_overall:.3f}, MAE: {mae_overall:.3f}")
print(f"(Based on {len(t_test_sample)} data points)")