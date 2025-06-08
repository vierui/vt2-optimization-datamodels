# %% [markdown]
# ## Neural Network Hyperparameter & Architecture Tuning
# Systematic optimization following the implementation plan

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import keras_tuner as kt
import time
from datetime import datetime

# %%
# UPDATED PARAMETERS - Following Implementation Plan
LAGS        = list(range(1,25))   # past 24h
TEST_YEARS  = 1                   # reduced to 1 year for more training data
STEP        = 'H'                 # hourly frequency
MAX_EPOCHS  = 100
TUNER_PATIENCE = 5               # early stopping patience during tuning
DISPLAY_LEN = 24                 # hours to plot
DATES       = ['2023-01-01','2023-01-08','2023-02-01']
TUNER_TRIALS = 20                # number of hyperparameter combinations to try
EXECUTIONS_PER_TRIAL = 2         # multiple runs per trial for robustness

print(f"Starting hyperparameter tuning at {datetime.now()}")
print(f"Target: {TUNER_TRIALS} trials with {EXECUTIONS_PER_TRIAL} executions each")

# %%
# 1. LOAD DATA (skip comments)
df = pd.read_csv(
    'data/renewables/pv_with_weather_data.csv',
    comment='#', parse_dates=['time']
)
df.rename(columns={'time':'timestamp','electricity':'pv','irradiance_direct':'dir_irr','irradiance_diffuse':'dif_irr','temperature':'temp'}, inplace=True)
df.set_index('timestamp', inplace=True)
# Ensure regular hourly
df = df.asfreq(STEP)

print(f"Data loaded: {len(df)} samples from {df.index.min()} to {df.index.max()}")

# %%
# 2. FEATURE ENGINEERING
def make_features(data, lags, include_time=True, include_weather=False):
    X = pd.DataFrame(index=data.index)
    if lags:
        for lag in lags:
            X[f'lag_{lag}'] = data['pv'].shift(lag)
    if include_time:
        X['sin_hour'] = np.sin(2*np.pi*data.index.hour/24)
        X['cos_hour'] = np.cos(2*np.pi*data.index.hour/24)
        X['sin_doy']  = np.sin(2*np.pi*data.index.dayofyear/365)
        X['cos_doy']  = np.cos(2*np.pi*data.index.dayofyear/365)
    if include_weather:
        X['dir_irr'] = data['dir_irr']
        X['dif_irr'] = data['dif_irr']
        X['temp']    = data['temp']
    y = data['pv']
    return X.join(y.rename('target')).dropna()

# Build feature sets
feats_time      = make_features(df, lags=LAGS, include_time=True,  include_weather=False)
feats_all       = make_features(df, lags=LAGS, include_time=True,  include_weather=True)
feats_weather   = make_features(df, lags=[],   include_time=False, include_weather=True)

print(f"Feature sets created:")
print(f"- Time-only: {feats_time.shape[1]-1} features")
print(f"- All features: {feats_all.shape[1]-1} features") 
print(f"- Weather-only: {feats_weather.shape[1]-1} features")

# %%
# 3. IMPROVED DATA SPLITS - Following Implementation Plan
# Training: everything up to today – 1 year
# Test: the final 1-year window
split_date = feats_all.index.max() - pd.DateOffset(years=TEST_YEARS)

def split_feats_improved(feats):
    """Split with proper train/test separation"""
    train_data = feats[feats.index <= split_date]
    test_data = feats[feats.index > split_date]
    
    X_train = train_data.drop(columns='target').values
    y_train = train_data['target'].values
    X_test = test_data.drop(columns='target').values
    y_test = test_data['target'].values
    
    return X_train, y_train, X_test, y_test, train_data.index, test_data.index

# Split all feature sets
Xt_tr, yt_tr, Xt_te, yt_te, idx_tr_t, idx_te_t = split_feats_improved(feats_time)
Xa_tr, ya_tr, Xa_te, ya_te, idx_tr_a, idx_te_a = split_feats_improved(feats_all)
Xw_tr, yw_tr, Xw_te, yw_te, idx_tr_w, idx_te_w = split_feats_improved(feats_weather)

print(f"\nData splits (1-year test):")
print(f"Training: {len(Xa_tr)} samples ({idx_tr_a.min()} to {idx_tr_a.max()})")
print(f"Test: {len(Xa_te)} samples ({idx_te_a.min()} to {idx_te_a.max()})")

# %%
# 4. SCALE DATA
y_scaler = StandardScaler().fit(ya_tr.reshape(-1,1))
Xt_scaler = StandardScaler().fit(Xt_tr)
Xa_scaler = StandardScaler().fit(Xa_tr)
Xw_scaler = StandardScaler().fit(Xw_tr)

# Scale training and test sets
Xt_tr_s, Xt_te_s = Xt_scaler.transform(Xt_tr), Xt_scaler.transform(Xt_te)
Xa_tr_s, Xa_te_s = Xa_scaler.transform(Xa_tr), Xa_scaler.transform(Xa_te)
Xw_tr_s, Xw_te_s = Xw_scaler.transform(Xw_tr), Xw_scaler.transform(Xw_te)

# Scale targets
yt_tr_s = y_scaler.transform(yt_tr.reshape(-1,1)).ravel()
ya_tr_s = y_scaler.transform(ya_tr.reshape(-1,1)).ravel()
yw_tr_s = y_scaler.transform(yw_tr.reshape(-1,1)).ravel()

print("Data scaling completed")

# %%
# 5. BASELINE MODELS - Stable references for comparison
def build_baseline_model(n_feat, model_type='simple'):
    """Build baseline models for comparison"""
    if model_type == 'simple':
        model = models.Sequential([
            layers.Input(shape=(n_feat,)),
            layers.Dense(n_feat*2, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(n_feat, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
    model.compile(optimizer='adam', loss='mse')
    return model

def eval_model(model, X_test, y_test, scaler):
    """Evaluate model and return metrics"""
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse, y_pred

# Train baseline models
print("\n" + "="*50)
print("TRAINING BASELINE MODELS")
print("="*50)

es_baseline = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Time-only baseline
print("Training time-only baseline...")
baseline_time = build_baseline_model(Xt_tr_s.shape[1])
baseline_time.fit(Xt_tr_s, yt_tr_s, validation_split=0.2, epochs=MAX_EPOCHS, 
                  batch_size=64, callbacks=[es_baseline], verbose=0)

# Weather-only baseline  
print("Training weather-only baseline...")
baseline_weather = build_baseline_model(Xw_tr_s.shape[1])
baseline_weather.fit(Xw_tr_s, yw_tr_s, validation_split=0.2, epochs=MAX_EPOCHS,
                     batch_size=64, callbacks=[es_baseline], verbose=0)

# Evaluate baselines on test set
mae_time_base, rmse_time_base, _ = eval_model(baseline_time, Xt_te_s, yt_te, y_scaler)
mae_weather_base, rmse_weather_base, _ = eval_model(baseline_weather, Xw_te_s, yw_te, y_scaler)

print(f"\nBASELINE RESULTS (1-year test):")
print(f"Time-only    MAE={mae_time_base:.3f}, RMSE={rmse_time_base:.3f}")
print(f"Weather-only MAE={mae_weather_base:.3f}, RMSE={rmse_weather_base:.3f}")
print(f"\nTarget: Beat both baselines with tuned all-features model")

# %%
# 6. HYPERPARAMETER TUNING MODEL BUILDER
def build_tunable_model(hp):
    """
    Build model with hyperparameters following the search space:
    - Hidden layers: 1-3
    - Units per layer: {32, 64, 128}  
    - Dropout: 0-0.4 (step 0.1)
    - L2 regularization: 1e-5 to 1e-2 (log scale)
    - Learning rate: {1e-3, 3e-4, 1e-4}
    - Batch size: {32, 64, 128}
    """
    n_features = Xa_tr_s.shape[1]
    
    # Architecture hyperparameters
    n_layers = hp.Int('n_layers', 1, 3)
    units = hp.Choice('units', [32, 64, 128])
    dropout_rate = hp.Float('dropout', 0.0, 0.4, step=0.1)
    l2_reg = hp.Float('l2_regularization', 1e-5, 1e-2, sampling='log')
    learning_rate = hp.Choice('learning_rate', [1e-3, 3e-4, 1e-4])
    
    # Build model
    model = models.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    
    # Add hidden layers
    for i in range(n_layers):
        model.add(layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Compile with tuned learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# %%
# 7. SET UP AND RUN HYPERPARAMETER TUNING
print("\n" + "="*50)
print("STARTING HYPERPARAMETER TUNING")
print("="*50)

# Create tuner
tuner = kt.RandomSearch(
    build_tunable_model,
    objective='val_loss',
    max_trials=TUNER_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    directory='hyperparameter_tuning',
    project_name='pv_forecasting_nn'
)

print(f"Tuner created: {TUNER_TRIALS} trials, {EXECUTIONS_PER_TRIAL} executions each")
print(f"Search space summary:")
tuner.search_space_summary()

# Callback for early stopping during tuning
tuning_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=TUNER_PATIENCE, 
        restore_best_weights=True
    )
]

# Start timing
start_time = time.time()

# Run the hyperparameter search
print(f"\nStarting search at {datetime.now()}")
tuner.search(
    Xa_tr_s, ya_tr_s,
    epochs=MAX_EPOCHS,
    validation_split=0.2,
    callbacks=tuning_callbacks,
    verbose=1,
    batch_size=None  # Will be handled by hp.Choice if added to build_tunable_model
)

search_time = time.time() - start_time
print(f"\nHyperparameter search completed in {search_time/60:.1f} minutes")

# %%
# 8. ANALYZE TUNING RESULTS
print("\n" + "="*50)
print("TUNING RESULTS ANALYSIS")
print("="*50)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
for param, value in best_hps.values.items():
    print(f"  {param}: {value}")

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
print(f"\nBest model architecture:")
best_model.summary()

# Show top 3 trials
print(f"\nTop 3 trials:")
for i, trial in enumerate(tuner.oracle.get_best_trials(num_trials=3)):
    print(f"Trial {i+1}: Loss = {trial.score:.6f}")
    for param, value in trial.hyperparameters.values.items():
        print(f"  {param}: {value}")
    print()

# %%
# 9. RETRAIN BEST MODEL ON FULL TRAINING DATA
print("\n" + "="*50)
print("RETRAINING BEST MODEL ON FULL TRAINING DATA")
print("="*50)

# Build final model with best hyperparameters
final_model = models.Sequential()
final_model.add(layers.Input(shape=(Xa_tr_s.shape[1],)))

n_layers = best_hps.get('n_layers')
units = best_hps.get('units')
dropout_rate = best_hps.get('dropout')
l2_reg = best_hps.get('l2_regularization')
learning_rate = best_hps.get('learning_rate')

for i in range(n_layers):
    final_model.add(layers.Dense(
        units, 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    final_model.add(layers.Dropout(dropout_rate))

final_model.add(layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train final model on full training data (no validation split)
final_callbacks = [
    callbacks.EarlyStopping(
        monitor='loss', 
        patience=15, 
        restore_best_weights=True
    )
]

print("Training final model on full training dataset...")
final_history = final_model.fit(
    Xa_tr_s, ya_tr_s,
    epochs=MAX_EPOCHS,
    batch_size=64,  # Use reasonable default
    callbacks=final_callbacks,
    verbose=1
)

# %%
# 10. FINAL EVALUATION
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

# Evaluate final tuned model
mae_tuned, rmse_tuned, pred_tuned = eval_model(final_model, Xa_te_s, ya_te, y_scaler)

print(f"FINAL RESULTS (1-year test):")
print(f"Time-only baseline    MAE={mae_time_base:.3f}, RMSE={rmse_time_base:.3f}")
print(f"Weather-only baseline MAE={mae_weather_base:.3f}, RMSE={rmse_weather_base:.3f}")
print(f"Tuned all-features    MAE={mae_tuned:.3f}, RMSE={rmse_tuned:.3f}")

# Calculate improvements
mae_improvement_vs_time = (mae_time_base - mae_tuned) / mae_time_base * 100
mae_improvement_vs_weather = (mae_weather_base - mae_tuned) / mae_weather_base * 100
rmse_improvement_vs_time = (rmse_time_base - rmse_tuned) / rmse_time_base * 100
rmse_improvement_vs_weather = (rmse_weather_base - rmse_tuned) / rmse_weather_base * 100

print(f"\nIMPROVEMENTS:")
print(f"vs Time-only:    MAE {mae_improvement_vs_time:+.1f}%, RMSE {rmse_improvement_vs_time:+.1f}%")
print(f"vs Weather-only: MAE {mae_improvement_vs_weather:+.1f}%, RMSE {rmse_improvement_vs_weather:+.1f}%")

# %%
# 11. DIAGNOSTIC PLOTS
print("\n" + "="*50)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*50)

# Training history plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(final_history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Final Model Training History')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample week forecast comparison
plt.subplot(1, 2, 2)
sample_days = 7 * 24  # One week
plt.plot(idx_te_a[:sample_days], ya_te[:sample_days], 'k-', label='Actual', linewidth=2)
plt.plot(idx_te_a[:sample_days], pred_tuned[:sample_days], 'r-', label='Tuned Model', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('PV Output')
plt.title('Sample Week Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 12. DAILY MAE SUMMARY FOR KEY DATES
print("\n" + "="*50)
print("DAILY MAE ANALYSIS")
print("="*50)

print("Date       | Tuned Model | Time Baseline | Weather Baseline")
print("-" * 60)

# Get baseline predictions for comparison
_, _, pred_time_base = eval_model(baseline_time, Xt_te_s, yt_te, y_scaler)
_, _, pred_weather_base = eval_model(baseline_weather, Xw_te_s, yw_te, y_scaler)

for date_str in DATES:
    day = pd.to_datetime(date_str)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'actual': ya_te,
        'tuned': pred_tuned,
        'time_base': pred_time_base,
        'weather_base': pred_weather_base
    }, index=idx_te_a)
    
    # Filter for specific day
    day_data = comparison_df[comparison_df.index.normalize() == day]
    
    if not day_data.empty:
        mae_tuned_day = np.mean(np.abs(day_data['tuned'] - day_data['actual']))
        mae_time_day = np.mean(np.abs(day_data['time_base'] - day_data['actual']))
        mae_weather_day = np.mean(np.abs(day_data['weather_base'] - day_data['actual']))
        
        print(f"{date_str} | {mae_tuned_day:>10.3f} | {mae_time_day:>12.3f} | {mae_weather_day:>15.3f}")
    else:
        print(f"{date_str} | Not in test range")

# %%
# 13. SUMMARY REPORT
print("\n" + "="*60)
print("HYPERPARAMETER TUNING SUMMARY REPORT")
print("="*60)

print(f"Experiment completed at: {datetime.now()}")
print(f"Total tuning time: {search_time/60:.1f} minutes")
print(f"Trials evaluated: {TUNER_TRIALS}")
print(f"Test period: {idx_te_a.min()} to {idx_te_a.max()}")

print(f"\nBest hyperparameters:")
for param, value in best_hps.values.items():
    print(f"  {param}: {value}")

print(f"\nFinal performance:")
print(f"  MAE: {mae_tuned:.3f}")
print(f"  RMSE: {rmse_tuned:.3f}")

print(f"\nBaseline comparisons:")
print(f"  Time-only baseline beaten: {'✓' if mae_tuned < mae_time_base else '✗'}")
print(f"  Weather-only baseline beaten: {'✓' if mae_tuned < mae_weather_base else '✗'}")

print(f"\nKey improvements achieved:")
print(f"  vs Time-only: {mae_improvement_vs_time:+.1f}% MAE, {rmse_improvement_vs_time:+.1f}% RMSE")
print(f"  vs Weather-only: {mae_improvement_vs_weather:+.1f}% MAE, {rmse_improvement_vs_weather:+.1f}% RMSE")

print(f"\nRecommendations:")
if mae_tuned < min(mae_time_base, mae_weather_base):
    print("  ✓ Tuned model successfully beats both baselines")
    print("  ✓ Hyperparameter optimization was effective")
else:
    print("  ⚠ Consider expanding search space or more trials")
    print("  ⚠ May need different architecture approaches")

print("\n" + "="*60)
print("TUNING COMPLETE")
print("="*60) 