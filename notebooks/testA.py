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
# print("\nData Description:\n", df.describe())
# cols = ['electricity', 't2m', 'temperature', 'swgdn', 'cldtot', 'prectotland', 'irradiance_direct', 'irradiance_diffuse']

# corr_matrix = df[cols].corr()
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()

# sns.pairplot(df[cols], diag_kind="kde", plot_kws={"alpha":0.3, "s":10})
# plt.suptitle('Pairplot: Electricity and Weather Features', y=1.02)
# plt.show()

# %%
# 3. Split: Train, Validation, Test - OPERATIONAL FORECASTING
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
# 4. Feature Engineering: Extract time features
for d in [train_df, valid_df, test_df]:
    d['hour'] = d['time'].dt.hour
    d['month'] = d['time'].dt.month
    d['dayofweek'] = d['time'].dt.dayofweek

# For test_df, identify forecast period (excluding lag buffer)
test_df['is_forecast_period'] = test_df['time'] >= forecast_start
forecast_mask = test_df['is_forecast_period'].values

print(f"\nForecast period analysis:")
print(f"Total test data: {test_df.shape[0]} hours")
print(f"Forecast period: {forecast_mask.sum()} hours (actual evaluation period)")
print(f"Lag buffer: {(~forecast_mask).sum()} hours (for potential lag features)")

# %%
# 5. Define Feature Sets and Labels
feature_sets = [
    (['hour', 'month'],
     "A: time")
    ]

target = 'electricity'

# %%
# 6. Utility Functions
def prepare_set(df, features, target):
    X = df[features]
    y = df[target].values
    return X, y

def prepare_forecast_set(df, features, target, forecast_mask):
    """Prepare data with forecast period extraction"""
    X = df[features]
    y = df[target].values
    X_forecast = X[forecast_mask]
    y_forecast = y[forecast_mask]
    return X, y, X_forecast, y_forecast

def evaluate_model(X_train, y_train, X_valid, y_valid, X_test_forecast, y_test_forecast, label):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test_forecast)
    metrics = {}
    for split, y, y_pred in [
        ("Valid", y_valid, y_valid_pred),
        ("Test",  y_test_forecast,  y_test_pred)
    ]:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics[split] = (rmse, mae, r2)
        print(f"{label} - {split}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  ")
    print("-" * 60)
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
    X_test, y_test, X_test_forecast, y_test_forecast = prepare_forecast_set(test_df, features, target, forecast_mask)
    model, y_valid_pred, y_test_pred, metrics = evaluate_model(
        X_train, y_train, X_valid, y_valid, X_test_forecast, y_test_forecast, label)
    all_valid_preds.append(y_valid_pred)
    all_test_preds.append(y_test_pred)
    labels.append(label)
    metrics_dict[label] = metrics

# Use forecast period targets for plotting
y_test_plot = y_test_forecast

# %%
# 8. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
for i, (y_pred, label) in enumerate(zip(all_test_preds, labels)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1]})
    
    # Left plot: Time series forecast (48 hours = 2 days) - rectangular
    n_hours = 48
    ax1.plot(y_test_plot[:n_hours], label='Actual', linewidth=2, color='black')
    ax1.plot(y_pred[:n_hours], label=f'Predicted', alpha=0.7, color='blue')
    ax1.set_title('Operational Forecast (First 2 Days from 2024-01-01)')
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
# 9. Daily Metrics Table (7-Day Forecast Period Only)
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
    print(f"\n{model}:")
    print("-" * 60)
    print(f"{'Date':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
    print("-" * 60)
    for _, row in model_data.iterrows():
        print(f"{row['Date']:<12} {row['MAE']:<8.3f} {row['RMSE']:<8.3f} {row['R2']:<8.3f}")

print("\n" + "=" * 80)
# %%