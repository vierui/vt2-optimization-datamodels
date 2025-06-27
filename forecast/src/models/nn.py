"""
Temporal Convolutional Network (TCN) forecaster for PV generation.
Replaces the previous MLP‐based neural network model.

Key design choices
------------------
* 1-D dilated causal convolutions with residual blocks (no recurrence)
* Flexible sequence length (`input_timesteps`, defaults to max   lag in
  config)
* Multi-step direct forecasting – one forward pass predicts the next
  `forecast_length` steps (default 24 h)
* Pinball / quantile loss (τ = 0.5 ⇒ MAE) implemented as a drop-in
  replacement for Keras losses

Usage
-----
```
from models.nn import train_tcn, forecast_tcn, evaluate_tcn_forecast

results = train_tcn(config, X_train, y_train, X_val, y_val)
model   = results["model"]
…
y_pred  = forecast_tcn(model, X_test_seq)
metrics = evaluate_tcn_forecast(y_test_seq, y_pred)
```
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Utility: quantile / pinball loss (τ = 0.5 == MAE)
# ----------------------------------------------------------------------------

def pinball_loss(tau: float = 0.5):
    """Return the pinball (quantile) loss @ quantile *tau*.

    For *tau* = 0.5 this is equivalent to the MAE up to a factor of 2.
    """

    def loss(y_true, y_pred):  # pylint:disable=missing-function-docstring
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * err, (tau - 1) * err))

    return loss


# ----------------------------------------------------------------------------
# Temporal Convolutional Network architecture
# ----------------------------------------------------------------------------

def _tcn_block(x: tf.Tensor, filters: int, kernel_size: int, dilation: int) -> tf.Tensor:
    """A single causal TCN residual block."""
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
    """Build a simple yet effective TCN for multistep forecasting."""

    inp = L.Input(shape=(input_timesteps, n_features))
    x = inp

    # Stacked residual blocks with exponentially increasing dilations
    for i in range(n_blocks):
        x = _tcn_block(x, filters, kernel_size, dilation=2 ** i)

    x = L.LayerNormalization()(x)

    # Global average pooling preserves position-invariant information
    x = L.GlobalAveragePooling1D()(x)

    out = L.Dense(forecast_length)(x)
    return keras.Model(inp, out, name="PV_TCN")


# ----------------------------------------------------------------------------
# Data helpers – sequence generator
# ----------------------------------------------------------------------------

def create_sequences(X: pd.DataFrame, y: pd.Series, input_timesteps: int,
                     forecast_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tabular features to sliding window sequences.

    *X* contains engineered features at 1-h resolution.
    """
    X_seq, y_seq = [], []
    total_len = len(X)

    for end in range(input_timesteps, total_len - forecast_length + 1):
        start = end - input_timesteps
        X_slice = X.iloc[start:end].values          # shape (timesteps, features)
        y_slice = y.iloc[end:end + forecast_length].values  # next 24 h
        X_seq.append(X_slice)
        y_seq.append(y_slice)

    return np.asarray(X_seq), np.asarray(y_seq)


# ----------------------------------------------------------------------------
# Public training / inference API
# ----------------------------------------------------------------------------


def train_tcn(config: Dict, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
    """Train a TCN with parameters defined in *config['tcn']*.

    The function keeps the signature similar to the old *train_nn* helper
    for drop-in replacement in existing notebooks.
    """
    tcn_cfg = config.setdefault("tcn", {})
    forecast_len = config["forecast"]["forecast_length"]

    # Hyper-parameters with sensible defaults
    input_steps = tcn_cfg.get("input_timesteps", max(config["features"]["target_lags"]))
    filters = tcn_cfg.get("filters", 64)
    n_blocks = tcn_cfg.get("n_blocks", 4)
    kernel_size = tcn_cfg.get("kernel_size", 3)
    batch_size = tcn_cfg.get("batch_size", 64)
    epochs = tcn_cfg.get("epochs", 100)
    patience = tcn_cfg.get("patience", 8)
    learning_rate = tcn_cfg.get("learning_rate", 1e-3)
    loss_type = tcn_cfg.get("loss", "pinball")  # 'pinball' or 'mae'

    # Prepare sequences
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, input_steps, forecast_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, input_steps, forecast_len)

    logger.info("Training sequences: %s", X_tr_seq.shape)
    logger.info("Validation sequences: %s", X_val_seq.shape)

    model = build_tcn(input_steps, X_train.shape[1], forecast_len,
                      filters=filters, n_blocks=n_blocks, kernel_size=kernel_size)

    loss_fn = pinball_loss(0.5) if loss_type.lower() == "pinball" else "mae"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss_fn,
                  metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=patience // 2,
                                          factor=0.5, min_lr=1e-6)
    ]

    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return {
        "model": model,
        "history": history.history,
        "params": {
            "input_timesteps": input_steps,
            "filters": filters,
            "n_blocks": n_blocks,
            "kernel_size": kernel_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss": loss_type,
            "learning_rate": learning_rate,
        },
    }


def forecast_tcn(model: keras.Model, X_seq: np.ndarray) -> np.ndarray:
    """Generate forecasts with a trained TCN.

    *X_seq* must have shape (samples, input_timesteps, n_features).
    """
    pred = model.predict(X_seq, verbose=0)
    return np.maximum(0, pred)  # electricity generation cannot be negative


def evaluate_tcn_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    # Flatten to compute aggregate metrics
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    mape = (np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-8)).mean()) * 100
    bias = (y_pred_f - y_true_f).mean()
    r2 = np.corrcoef(y_true_f, y_pred_f)[0, 1] ** 2 if len(y_true_f) > 1 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": bias,
        "r2": r2,
        "n_samples": len(y_true_f),
    }


# Backward-compatibility aliases ------------------------------------------------
PVNeuralNet = None               # the old class is deprecated
train_nn = train_tcn             # notebooks can keep their function calls
forecast_nn = forecast_tcn
evaluate_nn_forecast = evaluate_tcn_forecast
