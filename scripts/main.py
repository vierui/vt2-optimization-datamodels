#!/usr/bin/env python3
"""
Main entry point for the power grid optimization tool.

This script runs the integrated multi-year, multi-season optimization model
to determine the optimal grid expansion plan.
"""
import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

from pre import process_data_for_optimization
from optimization import optimize_integrated_network, optimize_seasonal_network
from post import generate_implementation_plan, calculate_annual_cost
from network import IntegratedNetwork, Network
from components import Bus, Generator, Load, Storage, Branch

def main():
    """Run the power grid optimization tool."""
    parser = argparse.ArgumentParser(
        description='Power Grid Optimization Tool with Integrated Multi-year Planning'
    )
    
    # Input files
    parser.add_argument('--grid-file', required=True,
                      help='Path to grid data CSV file')
    parser.add_argument('--profiles-dir', required=True,
                      help='Directory containing time series profile data')
    parser.add_argument('--analysis-file', required=False, default=None,
                      help='Path to analysis configuration JSON file')
    
    # Output settings
    parser.add_argument('--output-dir', required=False, default='results',
                      help='Directory to save output files (default: results)')
    parser.add_argument('--save-network', action='store_true',
                      help='Save the optimized network to a pickle file')
    
    # Optimization approach
    parser.add_argument('--integrated', action='store_true',
                      help='Use integrated optimization approach (default: False)')
    parser.add_argument('--season', required=False, choices=['winter', 'summer', 'spri_autu'],
                      help='Optimize only a specific season (for testing)')
    parser.add_argument('--year', type=int, required=False,
                      help='Optimize only a specific year (for testing)')
    
    # Advanced options
    parser.add_argument('--solver-options', type=json.loads, default={},
                      help='JSON string with solver options')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    try:
        processed_data = process_data_for_optimization(
            args.grid_file,
            args.profiles_dir,
            args.analysis_file
        )
        print("Data preprocessing completed.")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return 1
    
    # Step 2: Create and optimize integrated network
    print("\nStep 2: Creating and optimizing integrated network...")
    try:
        # Filter seasons if a specific season is requested
        all_seasons = list(processed_data['seasons_profiles'].keys())
        if args.season and args.season in all_seasons:
            print(f"Focusing on a single season: {args.season}")
            # Filter to only include the specified season
            seasons_to_use = [args.season]
        else:
            seasons_to_use = all_seasons
        
        # Filter years if a specific year is requested
        all_years = processed_data['grid_data']['analysis']['planning_horizon']['years']
        if args.year and args.year in all_years:
            print(f"Focusing on a single year: {args.year}")
            # Filter to only include the specified year
            years_to_use = [args.year]
        else:
            years_to_use = all_years
            
        # Create integrated network from preprocessed data
        integrated_network = IntegratedNetwork(
            seasons=seasons_to_use,
            years=years_to_use,
            discount_rate=processed_data['grid_data']['analysis']['planning_horizon']['system_discount_rate']
        )
        
        # Add seasonal networks
        for season, season_data in processed_data['seasons_profiles'].items():
            # Skip seasons that are not in our list
            if season not in seasons_to_use:
                continue
                
            # Create network for this season
            network = Network(name=season)
            
            # Add components from grid data
            # Add buses
            for idx, bus in processed_data['grid_data']['buses'].iterrows():
                bus_obj = Bus(
                    index=bus['id'], 
                    name=bus['name'], 
                    v_nom=bus.get('v_nom', 1.0)
                )
                network.buses = pd.concat([network.buses, pd.Series(bus_obj.__dict__, name=bus['id']).to_frame().T])
            
            # Add generators
            for idx, gen in processed_data['grid_data']['generators'].iterrows():
                gen_data = pd.Series({
                    'index': gen['id'],
                    'name': gen['name'],
                    'bus': gen['bus_id'],
                    'p_nom': gen['capacity_mw'],
                    'marginal_cost': gen['cost_mwh'],
                    'type': gen['type'],
                    'capex_per_mw': gen['capex_per_mw'],
                    'lifetime_years': gen['lifetime_years']
                }, name=gen['id'])
                network.generators = pd.concat([network.generators, gen_data.to_frame().T])
            
            # Add loads
            for idx, load in processed_data['grid_data']['loads'].iterrows():
                load_data = pd.Series({
                    'index': load['id'],
                    'name': load['name'],
                    'bus': load['bus_id'],
                    'p_set': load['p_mw']
                }, name=load['id'])
                network.loads = pd.concat([network.loads, load_data.to_frame().T])
            
            # Add storage
            if 'storage_units' in processed_data['grid_data']:
                for idx, storage in processed_data['grid_data']['storage_units'].iterrows():
                    storage_data = pd.Series({
                        'index': storage['id'],
                        'name': storage['name'],
                        'bus': storage['bus_id'],
                        'p_nom': storage['p_mw'],
                        'efficiency_store': storage['efficiency_store'],
                        'efficiency_dispatch': storage['efficiency_dispatch'],
                        'max_hours': storage['energy_mwh'] / storage['p_mw'] if storage['p_mw'] > 0 else 0,
                        'capex_per_mw': storage['capex_per_mw'],
                        'lifetime_years': storage['lifetime_years']
                    }, name=storage['id'])
                    network.storage_units = pd.concat([network.storage_units, storage_data.to_frame().T])
            
            # Add branches (lines)
            for idx, line in processed_data['grid_data']['lines'].iterrows():
                branch_data = pd.Series({
                    'index': line['id'],
                    'name': line['name'],
                    'from_bus': line['bus_from'],
                    'to_bus': line['bus_to'],
                    'x': 1.0 / line['susceptance'] if line['susceptance'] != 0 else 0,
                    's_nom': line['capacity_mw']
                }, name=line['id'])
                network.branches = pd.concat([network.branches, branch_data.to_frame().T])
            
            # Set up time series data
            # Create snapshots
            network.create_snapshots(
                start_time='2023-01-01', 
                periods=season_data['hours'], 
                freq='h'
            )
            
            # Add load time series
            for load_id in network.loads.index:
                try:
                    # Get profile for this load
                    profile = season_data['loads'].xs(load_id, level='load_id')['p_pu'].values
                    if len(profile) > 0:
                        # Get load nominal power
                        load_p_mw = network.loads.loc[load_id, 'p_set']
                        # Set load profile
                        network.add_load_time_series(load_id, profile * load_p_mw)
                except (KeyError, ValueError) as e:
                    # Use constant load
                    load_p_mw = network.loads.loc[load_id, 'p_set']
                    network.add_load_time_series(load_id, [load_p_mw] * len(network.snapshots))
            
            # Add to the integrated network
            integrated_network.add_season_network(season, network)
        
        # Run the optimization - either integrated or seasonal approach
        print("Running optimization...")
        if args.integrated:
            # Use the integrated optimization approach
            print("Using integrated optimization approach...")
            success = optimize_integrated_network(integrated_network, args.solver_options)
        else:
            # Use the seasonal optimization approach
            print("Using seasonal optimization approach...")
            success = optimize_seasonal_network(integrated_network, args.solver_options)
        
        if not success:
            print("Optimization failed!")
            return 1
        
        print("Optimization completed successfully.")
        
        # Save the optimized network if requested
        if args.save_network:
            network_file = os.path.join(args.output_dir, 'integrated_network.pkl')
            integrated_network.save_to_pickle(network_file)
            print(f"Optimized network saved to {network_file}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Generate implementation plan and calculate costs
    print("\nStep 3: Generating implementation plan...")
    try:
        # Generate implementation plan if we have installation data
        if hasattr(integrated_network, 'asset_installation_history') or hasattr(integrated_network, 'generators_installed_by_year'):
            # Maps generators_installed_by_year to asset_installation if needed
            if not hasattr(integrated_network, 'asset_installation') and hasattr(integrated_network, 'generators_installed_by_year'):
                integrated_network.asset_installation = {
                    'generators': integrated_network.generators_installed_by_year,
                    'storage': integrated_network.storage_installed_by_year if hasattr(integrated_network, 'storage_installed_by_year') else {}
                }
            
            implementation_plan = generate_implementation_plan(integrated_network)
            implementation_plan_file = os.path.join(args.output_dir, 'implementation_plan.json')
            with open(implementation_plan_file, 'w') as f:
                json.dump(implementation_plan, f, indent=2)
            print(f"Implementation plan saved to {implementation_plan_file}")
        else:
            print("Warning: No installation data found. Skipping implementation plan generation.")
        
        # Calculate annual costs
        if hasattr(integrated_network, 'seasons_total_cost'):
            annual_costs = calculate_annual_cost(integrated_network.seasons_total_cost)
            annual_costs_file = os.path.join(args.output_dir, 'annual_costs.json')
            with open(annual_costs_file, 'w') as f:
                json.dump(annual_costs, f, indent=2)
            print(f"Annual costs saved to {annual_costs_file}")
        elif hasattr(integrated_network, 'annual_cost'):
            # If annual cost is already calculated, just save it
            annual_costs_file = os.path.join(args.output_dir, 'annual_costs.json')
            with open(annual_costs_file, 'w') as f:
                json.dump(integrated_network.annual_cost, f, indent=2)
            print(f"Annual costs saved to {annual_costs_file}")
        else:
            print("Warning: No season costs found in the integrated network. Skipping annual cost calculation.")
        
    except Exception as e:
        print(f"Error during post-processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nPower grid optimization completed successfully.")
    return 0

def optimize_seasonal_network(integrated_network, solver_options=None):
    """
    Run optimization for each season separately
    
    Args:
        integrated_network: IntegratedNetwork object
        solver_options: Dictionary with solver options
    
    Returns:
        True if all optimizations succeeded, False otherwise
    """
    from optimization import optimize_seasonal_network
    # Set default solver options if None
    solver_options = solver_options or {}
    
    # Run optimization for each season
    success = optimize_seasonal_network(integrated_network, solver_options)
    
    return success

if __name__ == "__main__":
    sys.exit(main()) 