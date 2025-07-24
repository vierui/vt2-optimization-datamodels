# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
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

# Feature Set C: All weather + time
features_c = weather_features + time_features

# Feature Set D: All lags + engineering features  
features_d = lag_cols + time_features

# All features for Bayesian optimization
all_features = weather_features + time_features + lag_cols

feature_sets = [
    (features_c, "C: All weather + time"),
    (features_d, "D: All lags + engineering"),
]

# %%
# 6. Utility Functions
def prepare_set(df, features, target):
    # Only use features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target].values
    return X, y, available_features

def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test, label):
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
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
# 7. Bayesian Optimization for Feature Selection (Set E)
print("Preparing data for Bayesian optimization...")
X_train_all, y_train, available_features = prepare_set(train_df, all_features, target)
X_valid_all, y_valid, _ = prepare_set(valid_df, all_features, target)
X_test_all, y_test, _ = prepare_set(test_df, all_features, target)

print(f"Total available features: {len(available_features)}")
print(f"Training data shape: {X_train_all.shape}")

# Define search space for Bayesian optimization (feature selection only)
dimensions = [
    Integer(5, min(50, len(available_features)), name='n_features'),
]

@use_named_args(dimensions)
def objective(n_features):
    # Feature selection
    selector = SelectKBest(f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train_all, y_train)
    X_valid_selected = selector.transform(X_valid_all)
    
    # Model training with fixed hyperparameters
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    model.fit(X_train_selected, y_train)
    y_valid_pred = model.predict(X_valid_selected)
    
    # Return negative MAE (since we want to minimize)
    mae = mean_absolute_error(y_valid, y_valid_pred)
    return mae

print("Running Bayesian optimization...")
result = gp_minimize(objective, dimensions, n_calls=30, random_state=42, verbose=False)

# Extract best parameters
best_n_features = result.x[0]

print(f"Best parameters found:")
print(f"  n_features: {best_n_features}")
print(f"  Best MAE: {result.fun:.3f}")
print(f"Model hyperparameters (fixed):")
print(f"  n_estimators: 100")
print(f"  learning_rate: 0.1")
print(f"  max_depth: 6")

# %%
# 8. Train final model with best parameters (Set E)
print("\nTraining final optimized model...")
selector = SelectKBest(f_regression, k=best_n_features)
X_train_selected = selector.fit_transform(X_train_all, y_train)
X_valid_selected = selector.transform(X_valid_all)
X_test_selected = selector.transform(X_test_all)

selected_features = [available_features[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

best_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=10
)

best_model.fit(X_train_selected, y_train)
y_valid_pred_best = best_model.predict(X_valid_selected)
y_test_pred_best = best_model.predict(X_test_selected)

# Add to feature sets
feature_sets.append((selected_features, "E: Bayesian optimized"))

# %%
# 9. Run All Models
all_test_preds = []
all_valid_preds = []
labels = []
metrics_dict = {}

for features, label in feature_sets[:-1]:  # C and D
    X_train, y_train, _ = prepare_set(train_df, features, target)
    X_valid, y_valid, _ = prepare_set(valid_df, features, target)
    X_test, y_test, _ = prepare_set(test_df, features, target)
    
    model, y_valid_pred, y_test_pred, metrics = evaluate_model(
        X_train, y_train, X_valid, y_valid, X_test, y_test, label)
    
    all_valid_preds.append(y_valid_pred)
    all_test_preds.append(y_test_pred)
    labels.append(label)
    metrics_dict[label] = metrics

# Add Bayesian optimized results
all_valid_preds.append(y_valid_pred_best)
all_test_preds.append(y_test_pred_best)
labels.append("E: Bayesian optimized")

# Calculate metrics for Bayesian model
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred_best))
mae_valid = mean_absolute_error(y_valid, y_valid_pred_best)
r2_valid = r2_score(y_valid, y_valid_pred_best)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
mae_test = mean_absolute_error(y_test, y_test_pred_best)
r2_test = r2_score(y_test, y_test_pred_best)

metrics_dict["E: Bayesian optimized"] = {
    "Valid": (rmse_valid, mae_valid, r2_valid),
    "Test": (rmse_test, mae_test, r2_test)
}

print(f"E: Bayesian optimized - Valid: RMSE={rmse_valid:.3f}  MAE={mae_valid:.3f}  R2={r2_valid:.3f}")
print(f"E: Bayesian optimized - Test: RMSE={rmse_test:.3f}  MAE={mae_test:.3f}  R2={r2_test:.3f}")
print("-" * 60)

# %%
# 10. Individual Plots for Each Feature Set (Forecast + Scatter + Metrics)
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
# 11. Daily Metrics Table (First Week Only)
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
# 12. Feature Importance for Best Model
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance - Bayesian Optimized Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
# %%
