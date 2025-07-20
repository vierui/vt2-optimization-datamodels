# forecast/run.py
"""
Forecast pipeline  ▸  3-week test set, daily metrics, CV traces, residual
diagnostics, forecast uncertainty, and tuned-parameter persistence.

Usage
-----
python run.py --config config.yaml            # interactive (default)
python run.py --config config.yaml --force    # always re-train
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, yaml, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# ── local imports ─────────────────────────────────────────────────────────
from src.features       import add_time_features, add_lags, add_rolling_means
from src.eda            import correlation_heatmap, pairplot
from src.evaluation     import regression_metrics, to_frame
from src.models.gbdt    import build_feature_sets, train_gbdt, _cv_mae_folds
from src.models.sarima  import train_sarima


# -------------------------------------------------------------------------#
def make_dirs(root: Path):
    for sub in ("eda", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_builtin(obj):
    """
    Recursively convert numpy / pandas scalars (and timestamps) to builtin
    Python types so PyYAML can dump them.
    """
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat(timespec="seconds")
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


def save_cfg(path: str, cfg: dict):
    with open(path, "w") as f:
        yaml.safe_dump(_to_builtin(cfg), f, sort_keys=False)


# -------------------------------------------------------------------------#
def prompt_retrain(ts: str) -> bool:
    try:
        ans = input(f"Found tuned parameters from {ts} — re-train? [y/N] ").strip().lower()
        return ans == "y"
    except EOFError:             # non-interactive, e.g. CI
        return False


# -------------------------------------------------------------------------#
def drop_high_corr(df, thresh=0.9, protect=()):
    numeric = df.select_dtypes(include=[np.number])
    corr    = numeric.corr().abs()
    to_drop = set()

    for col in corr.columns:
        if col in to_drop:
            continue
        high = corr.index[(corr[col] > thresh) & (corr.index != col)]
        for h in high:
            if h in to_drop:                 continue
            if col in protect and h in protect:   continue
            elif col in protect:             to_drop.add(h)
            elif h  in protect:              to_drop.add(col)
            else:                            to_drop.add(h)
    return df.drop(columns=list(to_drop)), sorted(to_drop)


# -------------------------------------------------------------------------#
def run_pipeline(cfg_path: str, force: bool):

    cfg = load_cfg(cfg_path)
    out_root = Path(cfg["output_dir"])
    make_dirs(out_root)
    out_eda, out_rep = out_root / "eda", out_root / "reports"

    # ── 1. LOAD ────────────────────────────────────────────────────
    print("[1/9] Loading data …")
    df = pd.read_csv(cfg["dataset"], comment="#")
    df[cfg["time_column"]] = pd.to_datetime(df[cfg["time_column"]])
    df.sort_values(cfg["time_column"], inplace=True)
    df.rename(columns={cfg["target_column"]: "y"}, inplace=True)

    test_start = pd.to_datetime(cfg["test_start"])
    test_end   = test_start + pd.Timedelta(weeks=cfg.get("test_weeks", 3))
    mask_test  = (df[cfg["time_column"]] >= test_start) & (df[cfg["time_column"]] < test_end)
    train_df_orig = df[~mask_test].copy()
    test_df_orig  = df[mask_test ].copy()

    # ── 2. QUICK EDA ───────────────────────────────────────────────
    print("[2/9] Generating EDA plots …")
    cols_eda = ["y", "t2m", "temperature", "swgdn",
                "cldtot", "prectotland",
                "irradiance_direct", "irradiance_diffuse"]
    correlation_heatmap(train_df_orig, cols_eda, out_eda)
    pairplot(train_df_orig.sample(frac=0.15, random_state=0), cols_eda, out_eda)

    # ── 3. DROP HIGH CORR ──────────────────────────────────────────
    print("[3/9] Dropping highly-correlated (>0.9) columns …")
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

    # ── 4. FEATURE ENGINEERING ─────────────────────────────────────
    print("[4/9] Building feature tracks …")
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

    # ── 5. BUILD / LOAD FEATURE SETS ───────────────────────────────
    tuned = cfg.get("tuned", {}) or {}

    feats_ready = all(
        isinstance(tuned.get(k), list) and tuned[k]
        for k in ("bsfs_feats", "fwdsfs_feats", "bayesopt_feats")
    )
    retrain = (
        force or
        (not feats_ready) or
        (feats_ready and not force and prompt_retrain(tuned.get("timestamp", "-")))
    )

    if retrain:
        print("[5/9] Searching feature subsets …")
        sets_dict, cv_mae, gain_series, bo_trace, cv_scores = build_feature_sets(
            manual_feats, rich_tr,
            cfg["feature_selection"]["n_keep"],
            cfg["feature_selection"]["forward_pool"],
            cfg["feature_selection"]["k"],
            cfg["cv"], cfg["random_state"],
        )

        cfg["tuned"] = {
            "bsfs_feats":     sets_dict["BSFS"],
            "fwdsfs_feats":   sets_dict["FwdSFS"],
            "bayesopt_feats": sets_dict["BayesOpt"],
            "bo_trace":       bo_trace,
            "timestamp":      pd.Timestamp.utcnow(),
            # sarima_order filled later
        }
        save_cfg(cfg_path, cfg)
    else:
        print("[5/9] Using tuned feature lists from config …")
        sets_dict = {
            "Manual":   manual_feats,
            "BSFS":     tuned["bsfs_feats"],
            "FwdSFS":   tuned["fwdsfs_feats"],
            "BayesOpt": tuned["bayesopt_feats"],
        }
        cv_mae, cv_scores, bo_trace = {}, {}, tuned.get("bo_trace", [])
        for lbl, feats in sets_dict.items():
            folds = _cv_mae_folds(rich_tr[feats], rich_tr["y"].values, cfg["cv"], cfg["random_state"])
            cv_mae[lbl]   = float(np.mean(folds))
            cv_scores[lbl] = folds
        gain_series = pd.Series(dtype=float)

    # ── 6. TRAIN & EVAL ────────────────────────────────────────────
    print("[6/9] Training & evaluating …")
    gbt_metrics, gbt_preds = {}, {}
    for lbl, feats in sets_dict.items():
        print(f"    → {lbl} ({len(feats)} feats)")
        preds, _ = train_gbdt(rich_tr, rich_te, feats, cfg["random_state"], label=lbl)
        gbt_preds[lbl]   = preds
        gbt_metrics[lbl] = regression_metrics(rich_te["y"].values, preds)

    # ── 7. SARIMA ─────────────────────────────────────────────────
    print("[7/9] SARIMA benchmark …")
    if retrain or (not tuned.get("sarima_order")):
        sar_pred, sar_cfg = train_sarima(rich_tr, rich_te, cfg["sarima"], cfg["time_column"])
        cfg["tuned"]["sarima_order"] = sar_cfg["order"]
        save_cfg(cfg_path, cfg)
    else:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = tuple(tuned["sarima_order"])
        train_ts = rich_tr.set_index(cfg["time_column"])["y"]
        sarima = SARIMAX(
            train_ts,
            order=(order[0], 0, order[1]),
            seasonal_order=(order[2], 0, order[3], cfg["sarima"]["seasonal_period"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        sar_pred = sarima.forecast(steps=len(rich_te))
        sar_pred.index = rich_te[cfg["time_column"]].values
        sar_cfg = {"order": order, "aic": sarima.aic}

    sar_metrics = regression_metrics(rich_te["y"].values, sar_pred.values)

    if retrain:
        save_cfg(cfg_path, cfg)      # ensure SARIMA order persisted

    # ── 8. DIAGNOSTIC & EXTRA PLOTS ───────────────────────────────
    print("[8/9] Generating diagnostic plots …")

    # A. CV trace ---------------------------------------------------
    cv_df = (
        pd.DataFrame(cv_scores)
        .reset_index(names="Fold")
        .melt(id_vars="Fold", var_name="Model", value_name="MAE")
    )
    cv_df.to_csv(out_rep / "cv_scores.csv", index=False)
    plt.figure(figsize=(7, 4))
    for m, g in cv_df.groupby("Model"):
        plt.plot(g["Fold"] + 1, g["MAE"], marker="o", label=m)
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("Time-series CV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_rep / "cv_scores.png")
    plt.close()

    # B. daily metrics ---------------------------------------------
    daily_rows, true_y, dates = [], rich_te["y"].values, rich_te[cfg["time_column"]].dt.normalize()
    for lbl, preds in {**gbt_preds, "SARIMA": sar_pred.values}.items():
        for d in pd.date_range(test_start, test_end, freq="D")[:-1]:
            m = dates == d
            if not m.any():
                continue
            daily_rows.append({
                "date":  d.date(),
                "model": lbl,
                "MAE":   mean_absolute_error(true_y[m], preds[m]),
                "RMSE":  np.sqrt(mean_squared_error(true_y[m], preds[m])),
                "R2":    r2_score(true_y[m], preds[m]),
            })
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(out_rep / "daily_metrics.csv", index=False)
    daily_df.to_json(out_rep / "daily_metrics.json", orient="records", indent=2)

    for metric in ["MAE", "RMSE", "R2"]:
        daily_df.pivot(index="date", columns="model", values=metric).plot(
            kind="bar", figsize=(12, 4)
        )
        plt.ylabel(metric)
        plt.title(f"{metric} per day (3-week test)")
        plt.tight_layout()
        plt.savefig(out_rep / f"daily_{metric.lower()}.png")
        plt.close()

    # C. residual panels (first 3 days) -----------------------------
    first_days = [test_start + pd.Timedelta(weeks=i) for i in range(3)]
    panel_models = ["Manual", "FwdSFS", "BayesOpt"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=False)
    for r, d in enumerate(first_days):
        m = (rich_te[cfg["time_column"]] >= d) & (rich_te[cfg["time_column"]] < d + pd.Timedelta(days=1))
        x = rich_te[cfg["time_column"]][m]
        for c, mod in enumerate(panel_models):
            res = (gbt_preds[mod] if mod in gbt_preds else sar_pred.values)[m] - rich_te["y"].values[m]
            ax = axes[r, c]
            ax.plot(x, res)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(f"{mod} – {d.date()}")
    fig.tight_layout()
    fig.savefig(out_rep / "residuals_firstday_panels.png")
    plt.close()

    # D. SARIMA residual ACF ---------------------------------------
    sar_resid = sar_pred.values - rich_te["y"].values
    plt.figure(figsize=(6, 4))
    plot_acf(sar_resid, lags=48, alpha=0.05)
    plt.title("SARIMA residual ACF (48 lags)")
    plt.tight_layout()
    plt.savefig(out_rep / "acf_sarima.png")
    plt.close()

    # E. uncertainty ribbon (best day-1 model) ---------------------
    day1_mask  = dates == test_start
    day1_truth = rich_te["y"].values[day1_mask]
    models_day1 = {lbl: preds[day1_mask] for lbl, preds in gbt_preds.items()}
    models_day1["SARIMA"] = sar_pred.values[day1_mask]
    best_lbl = min(models_day1, key=lambda m: mean_absolute_error(day1_truth, models_day1[m]))
    print(f"      Uncertainty ribbon generated for best model on day-1 → {best_lbl}")

    feats_best = sets_dict[best_lbl] if best_lbl != "SARIMA" else None
    if feats_best:
        X_tr, y_tr = rich_tr[feats_best], rich_tr["y"].values
        X_te       = rich_te[feats_best][day1_mask]
        boot_pred  = np.zeros((100, len(X_te)))
        rs_base    = cfg["random_state"]
        for i in range(100):
            idx = np.random.choice(len(X_tr), size=len(X_tr), replace=True)
            mdl = GradientBoostingRegressor(random_state=rs_base+i).fit(X_tr.iloc[idx], y_tr[idx])
            boot_pred[i] = mdl.predict(X_te)
        lower  = np.percentile(boot_pred,  5, axis=0)
        upper  = np.percentile(boot_pred, 95, axis=0)
        median = np.median   (boot_pred, axis=0)
        x_ax   = rich_te[cfg["time_column"]][day1_mask]
        plt.figure(figsize=(11, 4))
        plt.fill_between(x_ax, lower, upper, alpha=0.3, label="90 % PI")
        plt.plot(x_ax, median, label="Median pred")
        plt.plot(x_ax, day1_truth, color="k", label="True")
        plt.title(f"Uncertainty – {best_lbl} (day-1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_rep / "best_model_uncertainty_day1.png")
        plt.close()

    # F. first-3-days prediction trace -----------------------------
    n_hours = cfg["plots_first_n_hours"]
    mask_3d = rich_te[cfg["time_column"]] < test_start + pd.Timedelta(hours=n_hours)
    x_ax    = rich_te[cfg["time_column"]][mask_3d]
    plt.figure(figsize=(11, 5))
    plt.plot(x_ax, rich_te["y"].loc[x_ax.index], label="True", color="black")
    for lbl, preds in gbt_preds.items():
        plt.plot(x_ax, preds[x_ax.index], label=f"GBT-{lbl}")
    plt.plot(x_ax, sar_pred.values[x_ax.index], label="SARIMA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_rep / "pred_vs_true_first3days.png")
    plt.close()

    # ── 9. SUMMARY JSON + console ─────────────────────────────────
    print("[9/9] Writing summary …")
    summary = {
        "cv_mae":           cv_mae,
        "gbt_metrics":      gbt_metrics,
        "sarima_metrics":   sar_metrics,
        "sarima_best_order": sar_cfg["order"],
        "sarima_aic":        sar_cfg["aic"],
        "bayesopt_trace":   {str(k): float(v) for k, v in bo_trace},
        "bayesopt_best_k":  int(min(bo_trace, key=lambda t: t[1])[0]) if bo_trace else None,
    }
    with open(out_rep / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== HOLD-OUT METRICS (3-week) =====")
    print(to_frame({**gbt_metrics, "SARIMA": sar_metrics}))
    print(f"\n✓ Artefacts saved in {out_root.resolve()}\n")


# -------------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force",  action="store_true", help="force re-training; ignore tuned parameters")
    args = ap.parse_args()
    run_pipeline(args.config, force=args.force)