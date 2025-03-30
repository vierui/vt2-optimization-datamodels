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

# Import local modules
from network import Network
from pre import process_data_for_optimization, SEASON_WEEKS, SEASON_WEIGHTS
from post import calculate_annual_cost, save_annual_cost_report, save_detailed_cost_report

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
    
    # Add buses
    if 'buses' in grid_data:
        for _, bus in grid_data['buses'].iterrows():
            network.add_bus(bus['id'], bus['name'])
    
    # Add generators
    if 'generators' in grid_data:
        for _, gen in grid_data['generators'].iterrows():
            network.add_generator(
                gen['id'], 
                gen['name'], 
                gen['bus_id'], 
                gen['capacity_mw'], 
                gen['cost_mwh'], 
                gen_type=gen['type'],
                capex_per_mw=gen.get('capex_per_mw', 0),
                lifetime_years=gen.get('lifetime_years', 25)
            )
    
    # Add loads
    if 'loads' in grid_data:
        for _, load in grid_data['loads'].iterrows():
            network.add_load(
                load['id'],
                load['name'],
                load['bus_id'],
                load['p_mw']
            )
    
    # Add storage units
    if 'storage_units' in grid_data:
        for _, storage in grid_data['storage_units'].iterrows():
            network.add_storage(
                storage['id'],
                storage['name'],
                storage['bus_id'],
                storage['p_mw'],
                storage['energy_mwh'],
                storage.get('efficiency_store', 0.95),
                storage.get('efficiency_dispatch', 0.95),
                capex_per_mw=storage.get('capex_per_mw', 0),
                lifetime_years=storage.get('lifetime_years', 15)
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
        args = parser.parse_args()
        
        # Validate directories
        for directory in [args.grid_dir, args.processed_dir]:
            if not os.path.exists(directory):
                print(f"Error: Directory not found: {directory}")
                return False
                
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Process data for optimization
        print("Processing data for optimization...")
        try:
            data = process_data_for_optimization(args.grid_dir, args.processed_dir)
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