#!/usr/bin/env python3
# forecast2/compare.py
"""
Single CLI entry-point for training, tuning, predicting and plotting forecast models.
"""
import argparse
import warnings
from pathlib import Path
import yaml
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .src.dataset import make_sets, load_raw
import importlib

# Import modules with hyphens using importlib
ml_xgb = importlib.import_module('forecast2.src.ml-xgb')
ml_tcn = importlib.import_module('forecast2.src.ml-tcn')
stat_sarima = importlib.import_module('forecast2.src.stat-sarima')
stat_prophet = importlib.import_module('forecast2.src.stat-prophet')

# Model Registry
ALL_REGISTRY = {
    "xgb": (ml_xgb.train, ml_xgb.predict, ml_xgb.tune),
    "tcn": (ml_tcn.train, ml_tcn.predict, ml_tcn.tune),
    "sarima": (stat_sarima.train, stat_sarima.predict, stat_sarima.tune),
    "prophet": (stat_prophet.train, stat_prophet.predict, stat_prophet.tune),
}
HORIZON = 24

def nmae(y_true, y_pred, mask=None):
    """Normalized Mean Absolute Error"""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if np.mean(y_true) == 0:
        return float('inf')
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)

def load_config():
    """Load configuration file"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_actual_values(day: str, horizon: int = HORIZON):
    """Get actual values for a specific day"""
    _, _, test_df = make_sets()
    day_start = pd.Timestamp(day).tz_localize("Europe/Zurich")
    
    if day_start not in test_df.index:
        available_days = test_df.index.date
        print(f"❌ Date {day} not found in test set.")
        print(f"📅 Available test dates: {available_days.min()} to {available_days.max()}")
        return None
    
    # Get actual values for the day
    day_data = test_df.loc[day_start:day_start + pd.Timedelta(hours=horizon-1)]
    return day_data["electricity_pu"].values

def evaluate_models(models: list, day: str, config: dict = None):
    """Evaluate multiple models on a specific day"""
    results = {}
    predictions = {}
    
    print(f"🔮 EVALUATING MODELS FOR {day}")
    print("=" * 50)
    
    # Get actual values
    y_true = get_actual_values(day)
    if y_true is None:
        return results, predictions
    
    # Also get day mask for daylight-only evaluation
    _, _, test_df = make_sets()
    day_start = pd.Timestamp(day).tz_localize("Europe/Zurich")
    day_data = test_df.loc[day_start:day_start + pd.Timedelta(hours=HORIZON-1)]
    day_mask = day_data["is_day"].values.astype(bool)
    
    for model_name in models:
        if model_name not in ALL_REGISTRY:
            print(f"❌ Unknown model: {model_name}")
            continue
            
        train_fn, predict_fn, tune_fn = ALL_REGISTRY[model_name]
        
        try:
            print(f"\n🤖 Processing {model_name.upper()}...")
            
            # Get model config
            model_cfg = config.get(model_name, {}) if config else {}
            
            # Load existing model (assumes it's already trained)
            print(f"📁 Loading {model_name}...")
            model = train_fn(model_cfg, reload=False)  # Don't retrain, just load
            
            if model is None:
                print(f"⚠️  No trained model found for {model_name}")
                continue
            
            # Make predictions
            print(f"🔮 Predicting with {model_name}...")
            y_pred = predict_fn(model, day)
            
            # Calculate metrics
            mae_all = nmae(y_true, y_pred)
            mae_day = nmae(y_true, y_pred, mask=day_mask) if day_mask.any() else float('inf')
            
            results[model_name] = {
                'nmae_24h': mae_all,
                'nmae_daylight': mae_day,
                'predictions': y_pred
            }
            predictions[model_name] = y_pred
            
            print(f"✅ {model_name.upper():>8} nMAE-24h={mae_all:.3f}   nMAE-daylight={mae_day:.3f}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    
    return results, predictions

def plot_comparison(day: str, results: dict, predictions: dict, save_path: str = None):
    """Plot comparison of model predictions"""
    y_true = get_actual_values(day)
    if y_true is None:
        return
    
    plt.figure(figsize=(12, 6))
    hours = range(len(y_true))
    
    # Plot actual values
    plt.plot(hours, y_true, 'k-', linewidth=2, label='Actual', alpha=0.8)
    
    # Plot predictions for each model
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        nmae_score = results[model_name]['nmae_24h']
        plt.plot(hours, y_pred, color=color, linestyle='--', 
                label=f'{model_name.upper()} (nMAE={nmae_score:.3f})', alpha=0.7)
    
    plt.title(f'Day-ahead PV Forecast Comparison • {day}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Per-unit Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()

def tune_models(models: list, config: dict = None):
    """Tune hyperparameters for specified models"""
    print("🔧 HYPERPARAMETER TUNING")
    print("=" * 50)
    
    for model_name in models:
        if model_name not in ALL_REGISTRY:
            print(f"❌ Unknown model: {model_name}")
            continue
            
        train_fn, predict_fn, tune_fn = ALL_REGISTRY[model_name]
        
        try:
            print(f"\n🔧 Tuning {model_name.upper()}...")
            model_cfg = config.get(model_name, {}) if config else {}
            model = tune_fn(model_cfg)
            
            if model is not None:
                print(f"✅ {model_name.upper()} tuning completed successfully")
            else:
                print(f"⚠️  {model_name.upper()} tuning failed")
                
        except Exception as e:
            print(f"❌ Error tuning {model_name}: {e}")
            continue

def train_models(models: list, config: dict = None):
    """Train specified models"""
    print("🏋️  MODEL TRAINING")
    print("=" * 50)
    
    for model_name in models:
        if model_name not in ALL_REGISTRY:
            print(f"❌ Unknown model: {model_name}")
            continue
            
        train_fn, predict_fn, tune_fn = ALL_REGISTRY[model_name]
        
        try:
            print(f"\n🏋️  Training {model_name.upper()}...")
            model_cfg = config.get(model_name, {}) if config else {}
            model = train_fn(model_cfg)
            
            if model is not None:
                print(f"✅ {model_name.upper()} training completed successfully")
            else:
                print(f"⚠️  {model_name.upper()} training failed")
                
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue

def run_diagnostics(models: list, config: dict = None):
    """Run pre-modeling diagnostics for specified models"""
    print("🔬 PRE-MODELING DIAGNOSTICS")
    print("=" * 50)
    
    for model_name in models:
        if model_name not in ALL_REGISTRY:
            print(f"❌ Unknown model: {model_name}")
            continue
        
        if model_name == "sarima":
            try:
                print(f"\n🔬 Running diagnostics for {model_name.upper()}...")
                
                # Load data
                data = load_raw()
                y = data['electricity_pu']
                print(f"📊 Dataset loaded: {y.shape[0]} observations")
                print(f"📅 Date range: {y.index.min()} to {y.index.max()}")
                
                # Run SARIMA diagnostics
                print("🧪 Running SARIMA pre-modeling diagnostics...")
                diagnostics_results = stat_sarima.run_diagnostics(y)
                
                # Save diagnostics results
                diagnostics_path = Path("forecast2/models/sarima_diagnostics.pkl")
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(diagnostics_results, diagnostics_path)
                
                print("✅ Diagnostics completed and saved!")
                print("\n" + "="*50)
                print("🎯 FINAL RECOMMENDATIONS FOR CONFIG:")
                print("="*50)
                
                rec = diagnostics_results['recommendations']
                print(f"📈 d (trend differencing): {rec['d']}")
                print(f"🔄 D (seasonal differencing): {rec['D']}")
                print(f"📊 p_range (AR orders): {rec['p_range']}")
                print(f"📊 q_range (MA orders): {rec['q_range']}")
                print(f"🌊 P_range (seasonal AR): {rec['P_range']}")
                print(f"🌊 Q_range (seasonal MA): {rec['Q_range']}")
                
                # Update config file with diagnostic results
                updated_config = update_sarima_config(config, diagnostics_results)
                
                print("\n✅ Config file updated with diagnostic recommendations!")
                
            except Exception as e:
                print(f"❌ Error running diagnostics for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"ℹ️  Diagnostics not implemented for {model_name}")

def update_sarima_config(config: dict, diagnostics_results: dict) -> dict:
    """Update config file with SARIMA diagnostic results"""
    
    rec = diagnostics_results['recommendations']
    
    # Create updated SARIMA config section
    sarima_config = {
        'd': rec['d'],
        'D': rec['D'],
        'max_p': max(rec['p_range']),
        'max_q': max(rec['q_range']),
        'max_P': max(rec['P_range']),
        'max_Q': max(rec['Q_range']),
        'seasonal_period': 24,
        'information_criterion': 'aicc',
        'stepwise': True,
        'trace': True,
        'n_fits': 50,
        'enforce_stationarity': False,
        'enforce_invertibility': False,
        # Store diagnostic metadata
        'diagnostics': {
            'trend_strength': diagnostics_results['stl']['trend_strength'],
            'seasonal_strength': diagnostics_results['stl']['seasonal_strength'],
            'unit_root_tests_passed': True,
            'diagnostics_date': pd.Timestamp.now().isoformat()
        }
    }
    
    print("\n📝 SARIMA config section to be saved:")
    for key, value in sarima_config.items():
        if key != 'diagnostics':
            print(f"  {key}: {value}")
    
    # Update the config dictionary
    config = config.copy()
    config['sarima'] = sarima_config
    
    # Write back to config file
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"💾 Config saved to: {config_path}")
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Forecast model comparison tool")
    parser.add_argument("--models", nargs='+', required=True,
                       choices=list(ALL_REGISTRY.keys()),
                       help="Models to work with")
    
    # Action group - only one action allowed
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--tune", action="store_true",
                             help="Tune hyperparameters for specified models")
    action_group.add_argument("--train", action="store_true",
                             help="Train specified models")
    action_group.add_argument("--diagnostic", action="store_true",
                             help="Run pre-modeling diagnostics (currently supports SARIMA)")
    action_group.add_argument("--day", type=str,
                             help="Day to evaluate models (YYYY-MM-DD)")
    
    # Optional arguments for evaluation mode
    parser.add_argument("--plot", action="store_true", default=True,
                       help="Generate comparison plot (only for --day mode)")
    parser.add_argument("--save-plot", type=str,
                       help="Save plot to specified path (only for --day mode)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    if args.tune:
        # Pure tuning mode - no day required
        tune_models(args.models, config)
        
    elif args.train:
        # Pure training mode - no day required
        train_models(args.models, config)
        
    elif args.diagnostic:
        # Diagnostic mode - run pre-modeling diagnostics
        run_diagnostics(args.models, config)
        
    elif args.day:
        # Evaluation mode - requires day
        results, predictions = evaluate_models(
            args.models, args.day, config=config
        )
        
        if not results:
            print("❌ No successful model evaluations")
            return
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 COMPARISON SUMMARY")
        print("=" * 50)
        for model_name, metrics in results.items():
            print(f"{model_name.upper():>8} nMAE-24h={metrics['nmae_24h']:.3f}   "
                  f"nMAE-daylight={metrics['nmae_daylight']:.3f}")
        
        # Plot comparison
        if args.plot and predictions:
            save_path = args.save_plot or f"results/{args.day.replace('-', '_')}_compare.png"
            
            # Ensure results directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            plot_comparison(args.day, results, predictions, save_path)

if __name__ == "__main__":
    main() 