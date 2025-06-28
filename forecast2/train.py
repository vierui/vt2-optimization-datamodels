# forecast2/train.py
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .dataset import make_sets, split, CFG
from .model import build_model, save_model, load_model

HORIZON = 24  # hours

def nmae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)

def fit():
    train_df, val_df, _ = make_sets()
    X_train, y_train = split(train_df, HORIZON)
    X_val, y_val = split(val_df, HORIZON)

    model = build_model()
    model.fit(X_train, y_train)
    save_model(model)

def predict(day: str):
    """Predict the 24 h starting at 00:00 for given day (YYYY-MM-DD)."""
    _, _, test_df = make_sets()
    day_start = pd.Timestamp(day).tz_localize("Europe/Zurich")
    X_pred = test_df.loc[[day_start]].drop(columns=["electricity_pu"])
    model = load_model()
    y_hat = model.predict(X_pred)[0]
    return y_hat

def evaluate():
    _, _, test_df = make_sets()
    X_test, y_test = split(test_df, HORIZON)
    model = load_model()
    y_hat = model.predict(X_test)
    score = nmae(y_test.values.ravel(), y_hat.ravel())
    print(f"Test nMAE (daylight+night): {score:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", type=str, metavar="YYYY-MM-DD")
    p.add_argument("--eval", action="store_true")
    args = p.parse_args()

    if args.train:
        fit()
    elif args.predict:
        print(predict(args.predict))
    elif args.eval:
        evaluate()
    else:
        p.print_help()

if __name__ == "__main__":
    main()