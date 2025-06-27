# %%
"""
Neural Network (TCN) Model for PV Forecasting
============================================

This revamped notebook trains and evaluates a **Temporal Convolutional
Network (TCN)**.  It is a drop-in upgrade over the previous MLP-based
notebook – only a few lines changed:

* imports now pull helpers from `models.nn` (TCN backend)
* sequence creation uses the shared `create_sequences` util
* evaluation sections read `input_timesteps` from the trained model
  params

Run as: `python 03_neuralnet.py`
"""
# %%
import sys
import os
from pathlib import Path

# Add the forecast/src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_io import load_config, load_and_process_data, create_time_splits
from features import FeatureEngineer
from models.nn import (
    train_nn,          # now trains a TCN
    forecast_nn,
    evaluate_nn_forecast,
    create_sequences,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8")

# %%
# ---------------------------------------------------------------------------
# 1. Setup folders
# ---------------------------------------------------------------------------
for p in ["../reports", "../reports/figures", "../models"]:
    Path(p).mkdir(parents=True, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# 2. Load data & create splits
# ---------------------------------------------------------------------------
config = load_config(str(src_dir / "config.yaml"))

df = load_and_process_data(config)
train_df, val_df, test_df = create_time_splits(df, config)

print(
    f"Training samples: {len(train_df)}  | Validation: {len(val_df)}  | Test: {len(test_df)}"
)

# %%
# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
fe = FeatureEngineer(config)

X_train, y_train = fe.make_features(train_df, use_weather=True)
X_val, y_val = fe.make_features(val_df, use_weather=True)
X_test, y_test = fe.make_features(test_df, use_weather=True)

print(f"Feature matrix shape (train): {X_train.shape}")

# %%
# ---------------------------------------------------------------------------
# 4. Train TCN – manual hyper-params (search=False) or basic random search
# ---------------------------------------------------------------------------
print("\n=== TRAINING TCN (manual) ===\n")
train_results = train_nn(
    config=config,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)

model = train_results["model"]
params = train_results["params"]
print("Best params:", params)

# %%
# ---------------------------------------------------------------------------
# 5. Plot training history
# ---------------------------------------------------------------------------
if train_results.get("history"):
    hist = train_results["history"]
    plt.figure(figsize=(10, 4))
    plt.plot(hist["loss"], label="Train")
    plt.plot(hist["val_loss"], label="Val")
    plt.title("Training history (loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../reports/figures/tcn_training_history.png", dpi=150)
    plt.show()

# %%
# ---------------------------------------------------------------------------
# 6. Validation evaluation (multi-step)
# ---------------------------------------------------------------------------
forecast_len = config["forecast"]["forecast_length"]
input_steps  = params["input_timesteps"]

X_val_seq, y_val_seq = create_sequences(X_val, y_val, input_steps, forecast_len)

val_pred = forecast_nn(model, X_val_seq)
val_metrics = evaluate_nn_forecast(y_val_seq, val_pred)
print("\nValidation metrics:")
for k, v in val_metrics.items():
    if isinstance(v, float):
        print(f"  {k.upper()}: {v:.4f}")

# %%
# ---------------------------------------------------------------------------
# 7. Horizon MAE curve (24-h)
# ---------------------------------------------------------------------------
h_mae = np.mean(np.abs(y_val_seq - val_pred), axis=0)
plt.figure(figsize=(8, 4))
plt.plot(range(1, forecast_len + 1), h_mae, "bo-")
plt.title("MAE by forecast horizon")
plt.xlabel("Hours ahead")
plt.ylabel("MAE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/figures/tcn_horizon_mae.png", dpi=150)
plt.show()

# %%
# ---------------------------------------------------------------------------
# 8. Save results & model
# ---------------------------------------------------------------------------
model.save("../models/pv_tcn.h5")

result_payload = {
    "model_type": "TCN",
    "params": params,
    "metrics": val_metrics,
}
with open("../reports/tcn_results.json", "w") as fp:
    json.dump(result_payload, fp, indent=2)

print("\nTCN training complete – results saved.")

# %%