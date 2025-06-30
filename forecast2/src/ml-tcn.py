# forecast2/src/ml-tcn.py
import joblib
import optuna
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# TensorFlow imports with warning suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from .dataset import make_sets, make_anchor_training_set, CFG

MODEL_PATH = Path(__file__).parents[1] / "models" / "tcn.h5"
HORIZON = 24

def nmae(y_true, y_pred, mask=None):
    """Normalized Mean Absolute Error"""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)

def pinball_loss(tau: float = 0.5):
    """Pinball (quantile) loss for tau=0.5 (equivalent to MAE)"""
    def loss(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * err, (tau - 1) * err))
    return loss

def _tcn_block(x: tf.Tensor, filters: int, kernel_size: int, dilation: int) -> tf.Tensor:
    """A single causal TCN residual block"""
    conv1 = L.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation,
                     activation="relu")(x)
    conv2 = L.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation,
                     activation="relu")(conv1)
    
    # Match dimensions for residual connection
    if x.shape[-1] != filters:
        x = L.Conv1D(filters, 1, padding="same")(x)
    
    return L.Add()([x, conv2])

def build_tcn(input_timesteps: int, n_features: int, forecast_length: int,
              filters: int = 64, n_blocks: int = 4, kernel_size: int = 3) -> keras.Model:
    """Build TCN model for multistep forecasting"""
    inp = L.Input(shape=(input_timesteps, n_features))
    x = inp
    
    # Stacked residual blocks with exponentially increasing dilations
    for i in range(n_blocks):
        x = _tcn_block(x, filters, kernel_size, dilation=2 ** i)
    
    x = L.LayerNormalization()(x)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(forecast_length)(x)
    
    return keras.Model(inp, out, name="TCN")

def create_sequences(X: pd.DataFrame, y: pd.Series, input_timesteps: int,
                     forecast_length: int) -> tuple:
    """Convert tabular data to sequences for TCN"""
    X_seq, y_seq = [], []
    total_len = len(X)
    
    for end in range(input_timesteps, total_len - forecast_length + 1):
        start = end - input_timesteps
        X_slice = X.iloc[start:end].values
        y_slice = y.iloc[end:end + forecast_length].values
        X_seq.append(X_slice)
        y_seq.append(y_slice)
    
    return np.asarray(X_seq), np.asarray(y_seq)

def _anchor_row(df, day: str):
    """Return the last timestamp BEFORE 00:00 local time of <day>"""
    import pandas as pd
    tgt = pd.Timestamp(day).tz_localize("Europe/Zurich")
    return df[df.index < tgt].iloc[-1]

def train(cfg: dict = None, reload: bool = False):
    """Train TCN model"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists() and not reload:
        return keras.models.load_model(MODEL_PATH, 
                                     custom_objects={'loss': pinball_loss(0.5)})

    train_df, val_df, _ = make_sets()
    
    # If no config provided, reload from file to get latest parameters
    if cfg is None:
        config_path = Path(__file__).parents[1] / "config.yaml"
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        params = current_config.get("tcn", {})
    else:
        params = cfg
    
    print(f"Training TCN with parameters: {params}")
    
    # Get parameters
    input_timesteps = params.get("input_timesteps", 168)
    filters = params.get("filters", 64)
    n_blocks = params.get("n_blocks", 4)
    kernel_size = params.get("kernel_size", 3)
    batch_size = params.get("batch_size", 32)
    epochs = params.get("epochs", 100)
    patience = params.get("patience", 10)
    learning_rate = params.get("learning_rate", 0.001)
    loss_type = params.get("loss", "pinball")
    
    # Create sequences
    X_train, y_train = create_sequences(
        train_df.drop("electricity_pu", axis=1), 
        train_df["electricity_pu"], 
        input_timesteps, 
        HORIZON
    )
    X_val, y_val = create_sequences(
        val_df.drop("electricity_pu", axis=1), 
        val_df["electricity_pu"], 
        input_timesteps, 
        HORIZON
    )
    
    print(f"Training sequences: {X_train.shape}, {y_train.shape}")
    print(f"Validation sequences: {X_val.shape}, {y_val.shape}")
    
    # Build model
    model = build_tcn(input_timesteps, X_train.shape[2], HORIZON,
                      filters=filters, n_blocks=n_blocks, kernel_size=kernel_size)
    
    # Compile model
    loss_fn = pinball_loss(0.5) if loss_type.lower() == "pinball" else "mae"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), 
                  loss=loss_fn, metrics=["mae"])
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=patience // 2,
                                          factor=0.5, min_lr=1e-6)
    ]
    
    # Train model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    # Save model
    model.save(MODEL_PATH)
    return model

def predict(model, day: str, horizon: int = HORIZON):
    """Make day-ahead predictions for a specific day"""
    _, _, test_df = make_sets()
    
    # Get TCN parameters for sequence creation
    config_path = Path(__file__).parents[1] / "config.yaml"
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f)
    input_timesteps = current_config.get("tcn", {}).get("input_timesteps", 168)
    
    # Find anchor point
    anchor_idx = test_df.index.get_loc(_anchor_row(test_df, day).name)
    
    # Create sequence ending at anchor point
    if anchor_idx < input_timesteps:
        raise ValueError(f"Not enough history for day {day}")
    
    start_idx = anchor_idx - input_timesteps + 1
    end_idx = anchor_idx + 1
    
    X_features = test_df.drop("electricity_pu", axis=1).iloc[start_idx:end_idx]
    X_seq = X_features.values.reshape(1, input_timesteps, -1)
    
    # Make prediction
    y_pred = model.predict(X_seq, verbose=0)[0]
    return np.maximum(0, y_pred)  # Ensure non-negative

def tune(cfg: dict = None, n_trials: int = 20):
    """Tune TCN hyperparameters using Optuna"""
    train_df, val_df, _ = make_sets()
    
    def objective(trial):
        ranges = CFG["tcn_tuning_ranges"]
        
        # Sample hyperparameters
        input_timesteps = trial.suggest_int("input_timesteps", 
                                           ranges["input_timesteps"]["min"], 
                                           ranges["input_timesteps"]["max"])
        filters = trial.suggest_int("filters", 
                                   ranges["filters"]["min"], 
                                   ranges["filters"]["max"])
        n_blocks = trial.suggest_int("n_blocks", 
                                    ranges["n_blocks"]["min"], 
                                    ranges["n_blocks"]["max"])
        kernel_size = trial.suggest_int("kernel_size", 
                                       ranges["kernel_size"]["min"], 
                                       ranges["kernel_size"]["max"])
        batch_size = trial.suggest_int("batch_size", 
                                      ranges["batch_size"]["min"], 
                                      ranges["batch_size"]["max"])
        learning_rate = trial.suggest_float("learning_rate", 
                                           ranges["learning_rate"]["min"], 
                                           ranges["learning_rate"]["max"], 
                                           log=ranges["learning_rate"].get("log", False))
        patience = trial.suggest_int("patience", 
                                    ranges["patience"]["min"], 
                                    ranges["patience"]["max"])
        
        params = {
            "input_timesteps": input_timesteps,
            "filters": filters,
            "n_blocks": n_blocks,
            "kernel_size": kernel_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "epochs": 50,  # Reduced for tuning
            "loss": "pinball"
        }
        
        try:
            # Create sequences
            X_train, y_train = create_sequences(
                train_df.drop("electricity_pu", axis=1), 
                train_df["electricity_pu"], 
                input_timesteps, 
                HORIZON
            )
            X_val, y_val = create_sequences(
                val_df.drop("electricity_pu", axis=1), 
                val_df["electricity_pu"], 
                input_timesteps, 
                HORIZON
            )
            
            if len(X_train) < 10 or len(X_val) < 10:
                return float('inf')
            
            # Build and train model
            model = build_tcn(input_timesteps, X_train.shape[2], HORIZON,
                              filters=filters, n_blocks=n_blocks, kernel_size=kernel_size)
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate), 
                          loss=pinball_loss(0.5), metrics=["mae"])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=params["epochs"],
                         batch_size=batch_size,
                         callbacks=[keras.callbacks.EarlyStopping(patience=patience//2, 
                                                                 restore_best_weights=True)],
                         verbose=0)
            
            # Evaluate
            y_pred = model.predict(X_val, verbose=0)
            score = nmae(y_val.ravel(), y_pred.ravel())
            
            # Clean up
            del model
            keras.backend.clear_session()
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best params:", study.best_params, "score:", study.best_value)
    
    # Update config file with best parameters
    config_path = Path(__file__).parents[1] / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update tcn section with best parameters
    config['tcn'].update(study.best_params)
    # Add back the epochs for final training
    config['tcn']['epochs'] = 100
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config file with best parameters: {study.best_params}")
    
    return train(reload=True) 