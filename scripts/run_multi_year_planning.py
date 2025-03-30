#!/usr/bin/env python3
"""
Multi-year Planning Optimization Script

This script uses the analysis.json configuration to run multi-year planning optimization:
1. Loads the multi-year planning configuration
2. Modifies grid data and profiles based on year-specific parameters
3. Runs optimization for each planning year and season
4. Calculates present value of system costs
"""
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np

# Add the parent directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
from scripts.main import main as run_single_year
from scripts.pre import process_data_for_optimization, SEASON_WEIGHTS
from scripts.network import Network
from scripts.post import calculate_annual_cost, save_annual_cost_report, save_detailed_cost_report

def load_planning_config(config_file):
    """
    Load the multi-year planning configuration from a JSON file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary with planning configuration
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required keys
        required_keys = ['planning_horizon', 'load_growth']
        for key in required_keys:
            if key not in config:
                print(f"Error: Missing required key '{key}' in planning configuration")
                return None
        
        # Ensure years are in planning_horizon
        if 'years' not in config['planning_horizon']:
            print(f"Error: Missing 'years' in planning_horizon configuration")
            return None
            
        return config
    except Exception as e:
        print(f"Error loading planning configuration: {e}")
        return None

def adjust_grid_data_for_year(grid_data, profiles, year, config):
    """
    Adjust grid data and profiles for a specific planning year
    
    Args:
        grid_data: Dictionary with grid component data
        profiles: Dictionary with season profiles
        year: Planning year
        config: Planning configuration
        
    Returns:
        Tuple of (adjusted_grid_data, adjusted_profiles)
    """
    # Make deep copies to avoid modifying the original data
    import copy
    adjusted_grid_data = copy.deepcopy(grid_data)
    adjusted_profiles = copy.deepcopy(profiles)
    
    # 1. Apply load growth
    load_growth_factor = config['load_growth'].get(str(year), 1.0)
    
    # Adjust load nominal values
    if 'loads' in adjusted_grid_data:
        adjusted_grid_data['loads']['p_mw'] *= load_growth_factor
    
    # Adjust load profiles for each season
    for season in adjusted_profiles:
        if 'loads' in adjusted_profiles[season]:
            # Scale load profiles by growth factor
            adjusted_profiles[season]['loads']['p_pu'] *= load_growth_factor
    
    # 2. Apply renewable availability factors
    if 'renewable_availability_factors' in config:
        # For wind generators
        if 'wind' in config['renewable_availability_factors']:
            wind_factor = config['renewable_availability_factors']['wind'].get(str(year), 1.0)
            
            # Identify wind generators
            if 'generators' in adjusted_grid_data:
                wind_gen_ids = adjusted_grid_data['generators'].loc[
                    adjusted_grid_data['generators']['type'] == 'wind', 'id'
                ].tolist()
                
                # Adjust profiles for each season
                for season in adjusted_profiles:
                    if 'generators' in adjusted_profiles[season]:
                        for gen_id in wind_gen_ids:
                            try:
                                mask = adjusted_profiles[season]['generators'].index.get_level_values('generator_id') == gen_id
                                adjusted_profiles[season]['generators'].loc[mask, 'p_max_pu'] *= wind_factor
                                # Cap at 1.0
                                adjusted_profiles[season]['generators'].loc[mask, 'p_max_pu'] = \
                                    adjusted_profiles[season]['generators'].loc[mask, 'p_max_pu'].clip(upper=1.0)
                            except Exception as e:
                                print(f"Warning: Could not adjust profile for wind generator {gen_id}: {e}")
        
        # For solar generators
        if 'solar' in config['renewable_availability_factors']:
            solar_factor = config['renewable_availability_factors']['solar'].get(str(year), 1.0)
            
            # Identify solar generators
            if 'generators' in adjusted_grid_data:
                solar_gen_ids = adjusted_grid_data['generators'].loc[
                    adjusted_grid_data['generators']['type'] == 'solar', 'id'
                ].tolist()
                
                # Adjust profiles for each season
                for season in adjusted_profiles:
                    if 'generators' in adjusted_profiles[season]:
                        for gen_id in solar_gen_ids:
                            try:
                                mask = adjusted_profiles[season]['generators'].index.get_level_values('generator_id') == gen_id
                                adjusted_profiles[season]['generators'].loc[mask, 'p_max_pu'] *= solar_factor
                                # Cap at 1.0
                                adjusted_profiles[season]['generators'].loc[mask, 'p_max_pu'] = \
                                    adjusted_profiles[season]['generators'].loc[mask, 'p_max_pu'].clip(upper=1.0)
                            except Exception as e:
                                print(f"Warning: Could not adjust profile for solar generator {gen_id}: {e}")
    
    # 3. Apply cost learning curves
    if 'cost_learning_curves' in config:
        # For wind generators
        if 'wind' in config['cost_learning_curves']:
            wind_cost_factor = config['cost_learning_curves']['wind'].get(str(year), 1.0)
            
            # Adjust generator CAPEX
            if 'generators' in adjusted_grid_data:
                wind_gen_mask = adjusted_grid_data['generators']['type'] == 'wind'
                if 'capex_per_mw' in adjusted_grid_data['generators'].columns and any(wind_gen_mask):
                    adjusted_grid_data['generators'].loc[wind_gen_mask, 'capex_per_mw'] *= wind_cost_factor
        
        # For solar generators
        if 'solar' in config['cost_learning_curves']:
            solar_cost_factor = config['cost_learning_curves']['solar'].get(str(year), 1.0)
            
            # Adjust generator CAPEX
            if 'generators' in adjusted_grid_data:
                solar_gen_mask = adjusted_grid_data['generators']['type'] == 'solar'
                if 'capex_per_mw' in adjusted_grid_data['generators'].columns and any(solar_gen_mask):
                    adjusted_grid_data['generators'].loc[solar_gen_mask, 'capex_per_mw'] *= solar_cost_factor
        
        # For storage units
        if 'storage' in config['cost_learning_curves']:
            storage_cost_factor = config['cost_learning_curves']['storage'].get(str(year), 1.0)
            
            # Adjust storage CAPEX
            if 'storage_units' in adjusted_grid_data:
                if 'capex_per_mw' in adjusted_grid_data['storage_units'].columns:
                    adjusted_grid_data['storage_units']['capex_per_mw'] *= storage_cost_factor
    
    # 4. Apply lifetime extensions if applicable
    if 'asset_lifetime_extensions' in config and 'installation_year_thresholds' in config['asset_lifetime_extensions']:
        # Find the applicable threshold year (the most recent threshold year that is <= the current year)
        applicable_threshold = None
        for threshold_year in sorted([int(y) for y in config['asset_lifetime_extensions']['installation_year_thresholds'].keys()]):
            if threshold_year <= year:
                applicable_threshold = str(threshold_year)
        
        if applicable_threshold:
            thresholds = config['asset_lifetime_extensions']['installation_year_thresholds'][applicable_threshold]
            
            # Apply to wind generators
            if 'wind' in thresholds and 'generators' in adjusted_grid_data:
                wind_gen_mask = adjusted_grid_data['generators']['type'] == 'wind'
                if 'lifetime_years' in adjusted_grid_data['generators'].columns and any(wind_gen_mask):
                    adjusted_grid_data['generators'].loc[wind_gen_mask, 'lifetime_years'] = thresholds['wind']
            
            # Apply to solar generators
            if 'solar' in thresholds and 'generators' in adjusted_grid_data:
                solar_gen_mask = adjusted_grid_data['generators']['type'] == 'solar'
                if 'lifetime_years' in adjusted_grid_data['generators'].columns and any(solar_gen_mask):
                    adjusted_grid_data['generators'].loc[solar_gen_mask, 'lifetime_years'] = thresholds['solar']
            
            # Apply to storage units
            if 'storage' in thresholds and 'storage_units' in adjusted_grid_data:
                if 'lifetime_years' in adjusted_grid_data['storage_units'].columns:
                    adjusted_grid_data['storage_units']['lifetime_years'] = thresholds['storage']
    
    return adjusted_grid_data, adjusted_profiles

def calculate_present_value(costs_by_year, discount_rate, base_year):
    """
    Calculate the present value of costs
    
    Args:
        costs_by_year: Dictionary mapping years to costs
        discount_rate: Annual discount rate
        base_year: Base year for present value calculation
        
    Returns:
        Present value of all costs
    """
    present_value = 0.0
    
    for year, cost in costs_by_year.items():
        # Calculate discount factor
        year_diff = int(year) - base_year
        discount_factor = (1 + discount_rate) ** (-year_diff)
        
        # Add discounted cost to present value
        present_value += cost * discount_factor
    
    return present_value

def run_multi_year_planning():
    """
    Run the multi-year planning optimization
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run multi-year planning optimization")
    parser.add_argument("--grid-dir", type=str, default=os.path.join(project_root, "data/grid"), 
                        help="Directory containing grid data")
    parser.add_argument("--processed-dir", type=str, default=os.path.join(project_root, "data/processed"),
                        help="Directory containing processed time series data")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "results/multi_year"),
                        help="Directory to store results")
    parser.add_argument("--config-file", type=str, default=os.path.join(project_root, "data/grid/analysis.json"),
                        help="Path to multi-year planning configuration file")
    args = parser.parse_args()
    
    # Validate paths
    for path in [args.grid_dir, args.processed_dir]:
        if not os.path.exists(path):
            print(f"Error: Directory not found: {path}")
            return False
    
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file not found: {args.config_file}")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load planning configuration
    print(f"Loading planning configuration from {args.config_file}...")
    config = load_planning_config(args.config_file)
    if not config:
        return False
    
    # Extract planning years and discount rate
    years = config['planning_horizon']['years']
    system_discount_rate = config['planning_horizon'].get('system_discount_rate', 0.05)
    base_year = years[0]
    
    print(f"Running multi-year planning for years: {years}")
    print(f"System discount rate: {system_discount_rate}")
    
    # Process base data for optimization
    print("\nProcessing base data for optimization...")
    try:
        base_data = process_data_for_optimization(args.grid_dir, args.processed_dir)
        if not base_data or 'grid_data' not in base_data or 'seasons_profiles' not in base_data:
            print("Error: Failed to process base data. Ensure all required files exist.")
            return False
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Dictionary to store yearly results
    year_results = {}
    annual_costs = {}
    
    # Run optimization for each year
    for year in years:
        print(f"\n{'='*80}")
        print(f"PLANNING YEAR: {year}")
        print(f"{'='*80}")
        
        # Create year output directory
        year_output_dir = os.path.join(args.output_dir, f"year_{year}")
        os.makedirs(year_output_dir, exist_ok=True)
        
        # Adjust grid data and profiles for this year
        print(f"Adjusting data for year {year}...")
        adjusted_grid_data, adjusted_profiles = adjust_grid_data_for_year(
            base_data['grid_data'], 
            base_data['seasons_profiles'],
            year,
            config
        )
        
        # Prepare adjusted data
        adjusted_data = {
            'grid_data': adjusted_grid_data,
            'seasons_profiles': adjusted_profiles
        }
        
        # Run optimization for all seasons in this year
        print(f"Running optimization for all seasons in year {year}...")
        try:
            from scripts.main import run_optimization_for_all_seasons
            season_results = run_optimization_for_all_seasons(adjusted_data, year_output_dir)
            if not season_results:
                print(f"Error: No season optimizations completed successfully for year {year}.")
                continue
        except Exception as e:
            print(f"Error during year {year} optimization: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Calculate annual cost for this year
        print(f"\nCalculating annual cost for year {year}...")
        try:
            from scripts.main import calculate_and_report_annual_cost
            annual_cost = calculate_and_report_annual_cost(season_results, year_output_dir)
            annual_costs[year] = annual_cost
        except Exception as e:
            print(f"Error calculating annual cost for year {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Store the results for this year
        year_results[year] = {
            'season_results': season_results,
            'annual_cost': annual_cost
        }
        
        print(f"Year {year} optimization completed successfully!")
    
    # Calculate present value of system costs
    if annual_costs:
        present_value = calculate_present_value(annual_costs, system_discount_rate, base_year)
        
        print("\n" + "="*80)
        print(f"MULTI-YEAR PLANNING RESULTS")
        print("="*80)
        print(f"Base year: {base_year}")
        print(f"System discount rate: {system_discount_rate}")
        print("\nAnnual costs by year:")
        for year in sorted(annual_costs.keys()):
            print(f"  Year {year}: {annual_costs[year]:.2f}")
        print(f"\nNet present value of system costs: {present_value:.2f}")
        
        # Save multi-year planning summary
        summary_path = os.path.join(args.output_dir, "multi_year_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'base_year': base_year,
                'planning_years': years,
                'system_discount_rate': system_discount_rate,
                'annual_costs': {str(year): annual_costs[year] for year in annual_costs},
                'present_value': present_value,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nMulti-year planning summary saved to: {summary_path}")
        print(f"Results saved to: {args.output_dir}")
        
        return True
    else:
        print("Error: No successful optimizations to calculate present value.")
        return False

if __name__ == "__main__":
    success = run_multi_year_planning()
    exit(0 if success else 1) 