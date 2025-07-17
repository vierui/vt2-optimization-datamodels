"""
Forecast pipeline – fixed UnboundLocal error & explicit basic / rich
test-frames creation.  Still prints concise progress logs.

python run.py --config config.yaml
"""
from pathlib import Path
import argparse, json, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features import add_time_features, add_lags, add_rolling_means
from src.eda      import correlation_heatmap, pairplot
from src.evaluation import regression_metrics, to_frame
from src.models.gbdt   import build_feature_sets, train_gbdt
from src.models.sarima import train_sarima


# ------------------------------------------------------------------ #
def make_dirs(root: Path):
    for sub in ("eda", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------- util: drop highly-correlated (>thresh) ------------------
def drop_high_corr(df, thresh=0.9, protect=()):
    """
    Iteratively removes a column from every highly-correlated pair.

    * `protect` columns are never removed; if one side of the pair is protected
      we drop the other one.  If both are protected we keep both.
    """
    numeric = df.select_dtypes(include=[np.number])
    # exclude protected cols (time index is not numeric anyway)
    corr = numeric.corr().abs()
    to_drop = set()

    for col in corr.columns:
        if col in to_drop:
            continue
        # columns highly-correlated with `col`
        high = corr.index[(corr[col] > thresh) & (corr.index != col)]
        for h in high:
            if h in to_drop:
                continue
            # decide which side to drop
            if col in protect and h in protect:
                # keep both (explicitly protected)
                continue
            elif col in protect:
                to_drop.add(h)
            elif h in protect:
                to_drop.add(col)
            else:
                # neither protected → arbitrarily drop `h`
                to_drop.add(h)

    df_out = df.drop(columns=list(to_drop))
    return df_out, sorted(to_drop)


# ------------------------------------------------------------------ #
def run_pipeline(cfg):
    out_root = Path(cfg["output_dir"])
    make_dirs(out_root)
    out_eda, out_rep = out_root / "eda", out_root / "reports"

    # 1. LOAD
    print("[1/7] Loading data …")
    df = pd.read_csv(cfg["dataset"], comment="#")
    df[cfg["time_column"]] = pd.to_datetime(df[cfg["time_column"]])
    df.sort_values(cfg["time_column"], inplace=True)
    df.rename(columns={cfg["target_column"]: "y"}, inplace=True)

    test_mask = df[cfg["time_column"]] >= cfg["test_start"]
    train_df_orig, test_df_orig = df[~test_mask].copy(), df[test_mask].copy()

    # 2. QUICK EDA
    print("[2/7] Generating EDA plots …")
    cols_eda = ["y", "t2m", "temperature", "swgdn",
                "cldtot", "prectotland",
                "irradiance_direct", "irradiance_diffuse"]
    correlation_heatmap(train_df_orig, cols_eda, out_eda)
    pairplot(train_df_orig.sample(frac=0.15, random_state=0), cols_eda, out_eda)

    # 3. DROP HIGH CORR
    print("[3/7] Dropping highly-correlated (>0.9) columns …")
    manual_feats = [
        "irradiance_direct", "irradiance_diffuse", "t2m", "cldtot",
        "hour", "month",
        "y_lag1", "y_lag24",
        "irradiance_direct_lag1", "irradiance_direct_lag24",
    ]
    train_df, dropped = drop_high_corr(
        train_df_orig, 0.9,
        protect=manual_feats + [cfg["time_column"], "y"]
    )
    test_df = test_df_orig.drop(columns=[c for c in dropped if c in test_df_orig.columns])
    print(f"      Removed {len(dropped)} col(s): {dropped}")

    # 4. FEATURE ENGINEERING TRACKS
    print("[4/7] Building feature tracks …")
    lag_basic, lag_rich = [1, 24], [1, 24, 48, 168]
    lag_cols = ["y", "irradiance_direct"]

    basic_tr = add_lags(add_time_features(train_df), lag_cols, lag_basic).dropna()
    rich_tr  = add_rolling_means(
        add_lags(add_time_features(train_df), lag_cols, lag_rich),
        lag_cols, (24, 168)).dropna()

    basic_te = add_lags(add_time_features(test_df), lag_cols, lag_basic).dropna()
    rich_te  = add_rolling_means(
        add_lags(add_time_features(test_df), lag_cols, lag_rich),
        lag_cols, (24, 168)).dropna()

    # 5. BUILD THREE GBT FEATURE SETS
    print("[5/7] Building Manual / BSFS / FwdSFS sets …")
    sets_dict, cv_mae = build_feature_sets(
        manual_feats,
        rich_tr,                              # always use rich DF for selection
        cfg["feature_selection"]["n_keep"],
        cfg["feature_selection"]["backward_keep"],  # backward_keep (now from config)
        cfg["feature_selection"]["forward_pool"],
        cfg["cv"],
        cfg["random_state"]
    )

    # map each set to its proper (train, test) frames
    sets_train = {lbl: (basic_tr if lbl == "Manual" else rich_tr)
                  for lbl in sets_dict}
    sets_test  = {lbl: (basic_te if lbl == "Manual" else rich_te)
                  for lbl in sets_dict}

    # 6. TRAIN / EVAL ALL SETS
    print("[6/7] Training & evaluating …")
    gbt_metrics, gbt_preds = {}, {}
    for lbl, feats in sets_dict.items():
        print(f"    → {lbl} ({len(feats)} feats)")
        preds, _ = train_gbdt(sets_train[lbl], sets_test[lbl],
                              feats, cfg["random_state"])
        gbt_preds[lbl] = preds
        gbt_metrics[lbl] = regression_metrics(sets_test[lbl]["y"].values, preds)

    best_lbl = min(gbt_metrics, key=lambda k: gbt_metrics[k]["MAE"])
    print(f"      Best GBT set: {best_lbl} (MAE {gbt_metrics[best_lbl]['MAE']:.3f})")

    # 7. SARIMA  (unchanged)
    print("[7/7] SARIMA benchmark …")
    sar_pred, sar_cfg = train_sarima(rich_tr, rich_te, cfg["sarima"], cfg["time_column"])
    sar_metrics = regression_metrics(rich_te["y"].values, sar_pred.values)

    # -------- PLOTS ------------------------------------------------
    n_hours = cfg["plots_first_n_hours"]
    start_t = rich_te[cfg["time_column"]].iloc[0]
    end_t   = start_t + pd.Timedelta(hours=n_hours)
    mask    = (rich_te[cfg["time_column"]] >= start_t) & (rich_te[cfg["time_column"]] < end_t)
    x_ax    = rich_te[cfg["time_column"]][mask]

    plt.figure(figsize=(11, 5))
    plt.plot(x_ax, rich_te["y"].values[mask], label="True", color="black")
    for lbl in gbt_preds:
        plt.plot(x_ax, gbt_preds[lbl][mask], label=f"GBT-{lbl}")
    plt.plot(x_ax, sar_pred.values[mask], label="SARIMA")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_rep / "pred_vs_true_first3days.png"); plt.close()

    # -------- SUMMARY ---------------------------------------------
    summary = {
        "cv_mae": cv_mae,
        "gbt_metrics": gbt_metrics,
        "sarima_metrics": sar_metrics,
        "sarima_best_order": sar_cfg["order"],
        "sarima_aic": sar_cfg["aic"],
        "top10_gain":  gain_series.head(10).to_dict(),
        "bottom10_gain": gain_series.tail(10).to_dict()
    }
    with open(out_rep / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== HOLD-OUT 2024 METRICS =====")
    print(to_frame({**gbt_metrics, "SARIMA": sar_metrics}))
    print(f"\n✓ Artefacts saved in {out_root.resolve()}\n")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    run_pipeline(load_cfg(ap.parse_args().config))