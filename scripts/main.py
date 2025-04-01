#!/usr/bin/env python3
"""
Main script for running the power grid optimization for a full year

This script orchestrates the entire optimization workflow:
1. Uses pre.py to load and process data for the three representative seasons
2. Creates network models for each season
3. Runs DC optimal power flow for each season
4. Calculates the annual cost using weighted season costs
"""
import os
import argparse
import json
from datetime import datetime
import pandas as pd

# Import local modules
from network import Network
from pre import process_data_for_optimization, SEASON_WEEKS, SEASON_WEIGHTS
from post import calculate_annual_cost, save_annual_cost_report, save_detailed_cost_report, generate_implementation_plan

def create_network_for_season(grid_data, season_profiles):
    """
    Create a network model for a specific season
    
    Args:
        grid_data: Dictionary with grid component data
        season_profiles: Dictionary with season-specific profiles
        
    Returns:
        Network object configured for the season
    """
    # Create an empty network
    network = Network()
    
    # Debug the grid_data state
    if 'generators' in grid_data:
        print("\nDebugging generator data:")
        print(f"Generators DataFrame columns: {grid_data['generators'].columns.tolist()}")
        if 'lifetime_years' in grid_data['generators'].columns:
            print(f"Generator lifetime_years values: {grid_data['generators']['lifetime_years'].tolist()}")
        else:
            print("WARNING: 'lifetime_years' column not found in generators DataFrame")
    
    if 'storage_units' in grid_data:
        print("\nDebugging storage data:")
        print(f"Storage DataFrame columns: {grid_data['storage_units'].columns.tolist()}")
        if 'lifetime_years' in grid_data['storage_units'].columns:
            print(f"Storage lifetime_years values: {grid_data['storage_units']['lifetime_years'].tolist()}")
        else:
            print("WARNING: 'lifetime_years' column not found in storage_units DataFrame")
    
    # Add buses
    if 'buses' in grid_data:
        for _, bus in grid_data['buses'].iterrows():
            network.add_bus(bus['id'], bus['name'])
    
    # Add generators
    if 'generators' in grid_data:
        for _, gen in grid_data['generators'].iterrows():
            # Check for required fields
            required_fields = ['id', 'name', 'bus_id', 'capacity_mw', 'cost_mwh', 'type', 'capex_per_mw', 'lifetime_years']
            missing_fields = [field for field in required_fields if field not in gen or pd.isna(gen[field])]
            
            if missing_fields:
                print(f"Generator {gen.get('id', 'unknown')} is missing fields: {missing_fields}")
                print(f"Generator row data: {gen.to_dict()}")
                # CHANGED: Now raising an error for missing fields instead of fallbacks
                raise ValueError(f"Generator {gen.get('id', 'unknown')} is missing required fields: {missing_fields}. Please provide all required values.")
            
            # Debug the generator data before adding it
            print(f"Adding generator {gen['id']} with lifetime: {gen['lifetime_years']} (type: {type(gen['lifetime_years'])})")
                
            network.add_generator(
                gen['id'], 
                gen['name'], 
                gen['bus_id'], 
                gen['capacity_mw'], 
                gen['cost_mwh'], 
                gen_type=gen['type'],
                capex_per_mw=gen['capex_per_mw'],
                lifetime_years=float(gen['lifetime_years'])  # Ensure it's a float
            )
    
    # Add loads
    if 'loads' in grid_data:
        for _, load in grid_data['loads'].iterrows():
            # Check for required fields
            required_fields = ['id', 'name', 'bus_id', 'p_mw']
            missing_fields = [field for field in required_fields if field not in load or pd.isna(load[field])]
            
            if missing_fields:
                print(f"WARNING: Load {load.get('id', 'unknown')} is missing required fields: {missing_fields}")
                continue
                
            network.add_load(
                load['id'],
                load['name'],
                load['bus_id'],
                load['p_mw']
            )
    
    # Add storage units
    if 'storage_units' in grid_data:
        for _, storage in grid_data['storage_units'].iterrows():
            # Check for required fields
            required_fields = ['id', 'name', 'bus_id', 'p_mw', 'energy_mwh', 'efficiency_store', 
                              'efficiency_dispatch', 'capex_per_mw', 'lifetime_years']
            missing_fields = [field for field in required_fields if field not in storage or pd.isna(storage[field])]
            
            if missing_fields:
                print(f"Storage {storage.get('id', 'unknown')} is missing fields: {missing_fields}")
                print(f"Storage row data: {storage.to_dict()}")
                # CHANGED: Now raising an error for missing fields instead of fallbacks
                raise ValueError(f"Storage {storage.get('id', 'unknown')} is missing required fields: {missing_fields}. Please provide all required values.")
            
            # Debug the storage data before adding it
            print(f"Adding storage {storage['id']} with lifetime: {storage['lifetime_years']} (type: {type(storage['lifetime_years'])})")
                
            network.add_storage(
                storage['id'],
                storage['name'],
                storage['bus_id'],
                storage['p_mw'],
                storage['energy_mwh'],
                storage['efficiency_store'],
                storage['efficiency_dispatch'],
                capex_per_mw=storage['capex_per_mw'],
                lifetime_years=float(storage['lifetime_years'])  # Ensure it's a float
            )
    
    # Add lines
    if 'lines' in grid_data:
        for _, line in grid_data['lines'].iterrows():
            network.add_line(
                line['id'],
                line['name'],
                line['bus_from'],
                line['bus_to'],
                line['susceptance'],
                line['capacity_mw']
            )
    
    # Set time horizon
    network.set_snapshots(season_profiles['hours'])
    
    # Set planning horizon from analysis.json if available
    if 'analysis' in grid_data and 'planning_horizon' in grid_data['analysis']:
        try:
            years = grid_data['analysis']['planning_horizon'].get('years', [])
            absolute_years = grid_data['analysis']['planning_horizon'].get('absolute_years', [])
            discount_rate = grid_data['analysis']['planning_horizon'].get('system_discount_rate', 0.05)
            
            if years:
                print(f"Setting planning horizon with relative years: {years}")
                if absolute_years:
                    print(f"Corresponding to absolute years: {absolute_years}")
                network.set_planning_horizon(years, discount_rate)
                
                # Store year mappings in the network for reference
                network.year_mapping = grid_data['analysis']['planning_horizon'].get('year_mapping', {})
                network.inverse_mapping = grid_data['analysis']['planning_horizon'].get('inverse_mapping', {})
                network.base_year = grid_data['analysis']['planning_horizon'].get('base_year', 2023)
                
                # Apply load growth factors based on year position
                if 'load_growth' in grid_data['analysis']:
                    load_growth = grid_data['analysis']['load_growth']
                    print(f"Applying load growth factors: {load_growth}")
                    
                    # Create a mapping of year to load factor
                    year_to_load_factor = {}
                    for year_str, factor in load_growth.items():
                        if year_str.isdigit():
                            year_to_load_factor[int(year_str)] = factor
                    
                    # Load growth adjustments will happen in the optimization module
                    network.year_to_load_factor = year_to_load_factor
        except Exception as e:
            print(f"Warning: Failed to set planning horizon from analysis.json: {e}")
            import traceback
            traceback.print_exc()
    
    # Set generator availability profiles
    if not season_profiles['generators'].empty:
        for gen_id in grid_data['generators']['id']:
            try:
                # Get profile for this generator
                profile = season_profiles['generators'].xs(gen_id, level='generator_id')['p_max_pu'].values
                if len(profile) > 0:
                    network.gen_p_max_pu[gen_id] = profile
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not set profile for generator {gen_id}: {e}")
                # Use default constant value
                network.gen_p_max_pu[gen_id] = [1.0] * network.T
    
    # Set load profiles
    if not season_profiles['loads'].empty:
        for load_id in grid_data['loads']['id']:
            try:
                # Get profile for this load
                profile = season_profiles['loads'].xs(load_id, level='load_id')['p_pu'].values
                if len(profile) > 0:
                    # Get load nominal power
                    load_p_mw = float(grid_data['loads'].loc[grid_data['loads']['id'] == load_id, 'p_mw'].values[0])
                    # Set load profile
                    network.loads_t[load_id] = profile * load_p_mw
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not set profile for load {load_id}: {e}")
                # Use constant load
                if load_id in network.loads.index:
                    load_p_mw = network.loads.loc[load_id, 'p_mw']
                    network.loads_t[load_id] = [load_p_mw] * network.T
    
    return network

def run_optimization_for_all_seasons(data, output_dir=None):
    """
    Run optimization for all three representative seasons
    
    Args:
        data: Dictionary with grid data and season profiles
        output_dir: Directory to save results (optional)
        
    Returns:
        Dictionary with optimization results for each season
    """
    results = {}
    
    for season in SEASON_WEEKS.keys():
        print(f"\nOptimizing {season.upper()} season...")
        
        # Get season-specific profiles
        season_profiles = data['seasons_profiles'][season]
        
        # Create network for this season
        network = create_network_for_season(data['grid_data'], season_profiles)
        
        # Run DC optimal power flow
        print(f"Running DC optimal power flow...")
        success = network.dcopf()
        
        if success:
            # Print season results summary
            print(f"Optimization successful!")
            print(f"Total cost for {season} season: {network.objective_value:.2f}")
            
            # Store the result
            results[season] = {
                'network': network,
                'cost': network.objective_value
            }
            
            # Save result to file if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                result_file = os.path.join(output_dir, f"{season}_result.json")
                
                # Save the result
                with open(result_file, 'w') as f:
                    json.dump({
                        'season': season,
                        'objective_value': network.objective_value,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                
                # Pickle the network for later analysis
                network_file = os.path.join(output_dir, f"{season}_network.pkl")
                network.save_to_pickle(network_file)
                
                print(f"Results saved to: {result_file}")
                print(f"Network saved to: {network_file}")
        else:
            print(f"Optimization failed for {season} season!")
    
    return results

def calculate_and_report_annual_cost(season_results, output_dir=None):
    """
    Calculate and report the annual cost based on season results
    
    Args:
        season_results: Dictionary with optimization results for each season
        output_dir: Directory to save the report (optional)
        
    Returns:
        Total annual cost
    """
    # Extract costs from results
    costs = {season: result['cost'] for season, result in season_results.items()}
    
    # Calculate annual cost
    annual_cost = calculate_annual_cost(costs)
    
    # Print annual cost calculation
    print("\nAnnual cost calculation:")
    for season, cost in costs.items():
        weeks = SEASON_WEIGHTS.get(season, 0)
        print(f"{season.capitalize()} cost: {cost:.2f} × {weeks} weeks = {cost * weeks:.2f}")
    print(f"Total annual cost: {annual_cost:.2f}")
    
    # Save annual cost report if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "annual_cost_report.json")
        save_annual_cost_report(costs, report_path)
        print(f"Annual cost report saved to: {report_path}")
        
        # Generate detailed cost reports for each season
        for season, result in season_results.items():
            detailed_report_path = os.path.join(output_dir, f"{season}_detailed_cost.json")
            save_detailed_cost_report(result['network'], detailed_report_path)
            
            # Generate implementation plan for each season
            implementation_plan_path = os.path.join(output_dir, f"{season}_implementation_plan.json")
            generate_implementation_plan(result['network'], implementation_plan_path)
        
        # Also generate a combined implementation plan using the winter season's network
        # (since all seasons should have the same installation decisions)
        if 'winter' in season_results:
            combined_plan_path = os.path.join(output_dir, "implementation_plan.json")
            generate_implementation_plan(season_results['winter']['network'], combined_plan_path)
            print(f"Combined implementation plan saved to: {combined_plan_path}")
    
    return annual_cost

def main():
    """
    Main function to run the optimization
    """
    try:
        # Get the project root directory (parent of scripts directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run annual power grid optimization using representative seasons")
        parser.add_argument("--grid-dir", type=str, default=os.path.join(project_root, "data/grid"), 
                            help="Directory containing grid data")
        parser.add_argument("--processed-dir", type=str, default=os.path.join(project_root, "data/processed"),
                            help="Directory containing processed time series data")
        parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "results/annual"),
                            help="Directory to store results")
        parser.add_argument("--planning-years", type=int, default=None,
                            help="Number of years in the planning horizon (default: use value from analysis.json)")
        args = parser.parse_args()
        
        # Validate directories
        for directory in [args.grid_dir, args.processed_dir]:
            if not os.path.exists(directory):
                print(f"Error: Directory not found: {directory}")
                return False
        
        # Check if analysis.json exists and read planning years if not specified
        if args.planning_years is None:
            analysis_path = os.path.join(args.grid_dir, 'analysis.json')
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, 'r') as f:
                        analysis_data = json.load(f)
                    if 'planning_horizon' in analysis_data and 'years' in analysis_data['planning_horizon']:
                        args.planning_years = len(analysis_data['planning_horizon']['years'])
                        print(f"Using planning horizon of {args.planning_years} years from analysis.json")
                except Exception as e:
                    print(f"Error reading analysis.json: {e}")
            
            # If still None, default to 10
            if args.planning_years is None:
                args.planning_years = 10
                print(f"Defaulting to planning horizon of {args.planning_years} years")
                
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Process data for optimization
        print("Processing data for optimization...")
        try:
            data = process_data_for_optimization(args.grid_dir, args.processed_dir, planning_years=args.planning_years)
            if not data or 'grid_data' not in data or 'seasons_profiles' not in data:
                print("Error: Failed to process data. Ensure all required files exist.")
                return False
        except Exception as e:
            print(f"Error during data processing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 2: Run optimization for all seasons
        print("\nRunning optimization for all seasons...")
        try:
            season_results = run_optimization_for_all_seasons(data, args.output_dir)
            if not season_results:
                print("Error: No season optimizations completed successfully.")
                return False
        except Exception as e:
            print(f"Error during season optimization: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 3: Calculate annual cost
        print("\nCalculating annual cost...")
        try:
            annual_cost = calculate_and_report_annual_cost(season_results, args.output_dir)
        except Exception as e:
            print(f"Error calculating annual cost: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\nOptimization completed successfully!")
        print(f"Total annual cost: {annual_cost:.2f}")
        print(f"Results saved to: {args.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 