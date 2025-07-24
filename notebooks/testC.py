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
# 2. Split: Train, Validation, Test with exact dates (test starts 1 week earlier)
train_df = df[df['time'] <= "2020-12-31"].copy()
valid_df = df[(df['time'] >= "2021-01-01") & (df['time'] <= "2023-12-24")].copy()
test_df = df[(df['time'] >= "2023-12-25") & (df['time'] < "2024-03-01")].copy()

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
def add_lags(df, lag_features, lags):
    for col in lag_features:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

lag_features = ['electricity', 'irradiance_direct', 'irradiance_diffuse', 't2m', 'cldtot']
lags = [1, 2, 3, 6, 12, 24, 48, 168]

for d in [train_df, valid_df, test_df]:
    add_lags(d, lag_features, lags)
    d.dropna(inplace=True)

# %%
# 5. Define Feature Sets
target = 'electricity'

# All available weather features
weather_features = ['t2m', 'temperature', 'swgdn', 'cldtot', 'prectotland', 
                   'irradiance_direct', 'irradiance_diffuse']

# All time features
time_features = ['hour', 'month', 'dayofweek', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']

# All lag features
lag_cols = [col for col in train_df.columns if '_lag' in col]

# All features for optimization
all_features = weather_features + time_features + lag_cols

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
# 7. Prepare data for optimization
print("Preparing data for Bayesian optimization...")
X_train_all, y_train, available_features = prepare_set(train_df, all_features, target)
X_valid_all, y_valid, _ = prepare_set(valid_df, all_features, target)
X_test_all, y_test, _ = prepare_set(test_df, all_features, target)

print(f"Total available features: {len(available_features)}")
print(f"Training data shape: {X_train_all.shape}")

# %%
# 8. Hyperparameter Tuning (Set F)
print("\nRunning Bayesian optimization for hyperparameters...")

# Define search space for hyperparameter tuning only
dimensions_f = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 12, name='max_depth'),
    Real(0.1, 1.0, name='subsample'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf'),
]

@use_named_args(dimensions_f)
def objective_f(n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf):
    # Model training with all features
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
    
    model.fit(X_train_all, y_train)
    y_valid_pred = model.predict(X_valid_all)
    
    mae = mean_absolute_error(y_valid, y_valid_pred)
    return mae

result_f = gp_minimize(objective_f, dimensions_f, n_calls=50, random_state=42, verbose=False)
best_params_f = {
    'n_estimators': result_f.x[0],
    'learning_rate': result_f.x[1],
    'max_depth': result_f.x[2],
    'subsample': result_f.x[3],
    'min_samples_split': result_f.x[4],
    'min_samples_leaf': result_f.x[5]
}

print(f"Set F - Best parameters:")
for param, value in best_params_f.items():
    print(f"  {param}: {value}")
print(f"  Best MAE: {result_f.fun:.3f}")

# Train Set F model
model_f, y_valid_pred_f, y_test_pred_f, metrics_f = evaluate_model(
    X_train_all, y_train, X_valid_all, y_valid, X_test_all, y_test, 
    "F: Hyperparameter tuning", **best_params_f)

# %%
# 9. Hyperparameter Tuning with TimeSeriesSplit CV (Set G)
print("\nRunning Bayesian optimization with TimeSeriesSplit CV...")

# Combine train and validation for time series CV
X_train_valid = pd.concat([pd.DataFrame(X_train_all, columns=available_features), 
                          pd.DataFrame(X_valid_all, columns=available_features)]).values
y_train_valid = np.concatenate([y_train, y_valid])

@use_named_args(dimensions_f)
def objective_g(n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf):
    # Model with time series cross-validation
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

result_g = gp_minimize(objective_g, dimensions_f, n_calls=50, random_state=42, verbose=False)
best_params_g = {
    'n_estimators': result_g.x[0],
    'learning_rate': result_g.x[1],
    'max_depth': result_g.x[2],
    'subsample': result_g.x[3],
    'min_samples_split': result_g.x[4],
    'min_samples_leaf': result_g.x[5]
}

print(f"Set G - Best parameters:")
for param, value in best_params_g.items():
    print(f"  {param}: {value}")
print(f"  Best CV MAE: {result_g.fun:.3f}")

# Train Set G model on full train+valid data
model_g = GradientBoostingRegressor(random_state=42, **best_params_g)
model_g.fit(X_train_valid, y_train_valid)
y_test_pred_g = model_g.predict(X_test_all)

# Calculate metrics for Set G (using valid set for comparison)
y_valid_pred_g = model_g.predict(X_valid_all)
rmse_valid_g = np.sqrt(mean_squared_error(y_valid, y_valid_pred_g))
mae_valid_g = mean_absolute_error(y_valid, y_valid_pred_g)
r2_valid_g = r2_score(y_valid, y_valid_pred_g)

rmse_test_g = np.sqrt(mean_squared_error(y_test, y_test_pred_g))
mae_test_g = mean_absolute_error(y_test, y_test_pred_g)
r2_test_g = r2_score(y_test, y_test_pred_g)

metrics_g = {
    "Valid": (rmse_valid_g, mae_valid_g, r2_valid_g),
    "Test": (rmse_test_g, mae_test_g, r2_test_g)
}

print(f"G: TimeSeriesSplit CV - Valid: RMSE={rmse_valid_g:.3f}  MAE={mae_valid_g:.3f}  R2={r2_valid_g:.3f}")
print(f"G: TimeSeriesSplit CV - Test: RMSE={rmse_test_g:.3f}  MAE={mae_test_g:.3f}  R2={r2_test_g:.3f}")
print("-" * 60)

# %%
# 10. Collect Results
all_test_preds = [y_test_pred_f, y_test_pred_g]
all_valid_preds = [y_valid_pred_f, y_valid_pred_g]
labels = ["F: Hyperparameter tuning", "G: TimeSeriesSplit CV"]
metrics_dict = {"F: Hyperparameter tuning": metrics_f, "G: TimeSeriesSplit CV": metrics_g}

# %%
# 11. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
for i, (y_pred, label) in enumerate(zip(all_test_preds, labels)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1]})
    
    # Left plot: Time series forecast (48 hours = 2 days) - rectangular
    n_hours = 48
    ax1.plot(y_test[:n_hours], label='Actual', linewidth=2, color='black')
    ax1.plot(y_pred[:n_hours], label=f'Predicted', alpha=0.7, color='blue')
    ax1.set_title('Forecast (First 2 Days)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Electricity (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Predicted vs Actual scatter - square with same height
    ax2.scatter(y_test, y_pred, s=10, alpha=0.5, color='blue')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
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
# 12. Daily Metrics Table (First Week Only)
def calculate_daily_metrics(y_test, test_preds, labels, test_df):
    test_df_copy = test_df.copy()
    test_df_copy['date'] = test_df_copy['time'].dt.date
    unique_dates = sorted(test_df_copy['date'].unique())
    
    # Limit to first 7 days only
    first_week_dates = unique_dates[:7]
    
    daily_metrics = []
    
    for label, y_pred in zip(labels, test_preds):
        for date in first_week_dates:
            day_mask = test_df_copy['date'] == date
            day_indices = test_df_copy[day_mask].index - test_df_copy.index[0]
            
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

daily_metrics_df = calculate_daily_metrics(y_test, all_test_preds, labels, test_df)

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
