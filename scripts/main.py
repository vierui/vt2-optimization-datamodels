#!/usr/bin/env python3
"""
Main entry point for the simplified multi-year power grid optimization tool.

Key changes compared to older versions:
  - Single binary variable for each asset per year (no re-install or replacement).
  - No slack variables (no load shedding).
  - Annualized CAPEX cost (capex_per_mw * p_nom / lifetime).
  - No discounting in the cost function (simple sum across years).
  - Storage SoC forced to zero at season start/end, removing cross-season coupling.
"""

import os
import sys
# Ensure the script can find modules in the scripts directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
import logging
import traceback

# Import our revised pre-processing, optimization, and post modules
from pre import process_data_for_optimization, SEASON_WEIGHTS
from optimization import solve_multi_year_investment  # Uses the new streamlined approach
from post import generate_implementation_plan, plot_seasonal_profiles, NumpyEncoder

# Import your integrated network class, or a similar structure
from network import IntegratedNetwork, Network  # Adjust if your code differs
from components import Bus, Generator, Load, Storage, Branch  # optional if needed

# Paths and constants
SEASON_WEIGHTS = {'winter': 13, 'summer': 13, 'spri_autu': 26}

def setup_logging(output_dir):
    """Set up logging to both console and file."""
    log_file = os.path.join(output_dir, 'optimization.log')
    
    # Create a logger
    logger = logging.getLogger('optimization')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    """Run the simplified multi-year power grid optimization tool."""
    parser = argparse.ArgumentParser(
        description='Simplified Multi-Year Power Grid Optimization'
    )

    # Input files
    parser.add_argument('--grid-file', required=True,
                        help='Path to grid data directory or CSV file')
    parser.add_argument('--profiles-dir', required=True,
                        help='Directory containing time series profile data')
    parser.add_argument('--analysis-file', required=False, default=None,
                        help='Path to analysis configuration JSON file')

    # Output settings
    parser.add_argument('--output-dir', required=False, default='results',
                        help='Directory to save output files (default: results)')
    parser.add_argument('--save-network', action='store_true',
                        help='Save the optimized network to a pickle file')

    # Optional solver options
    parser.add_argument('--solver-options', type=json.loads, default={},
                        help='JSON string with solver options (e.g. \'{"timelimit":3600}\')')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting optimization run")

    #------------------------------------------------------------------
    # 1) PRE-PROCESSING
    #    Load data and create the dictionary required for optimization
    #------------------------------------------------------------------
    logger.info("Step 1: Preprocessing data...")
    try:
        processed_data = process_data_for_optimization(
            grid_dir=args.grid_file,
            processed_dir=args.profiles_dir,
            planning_years=None  # or override if you want
        )
        logger.info("Data preprocessing completed.")
        
        # Log the data summary
        logger.info(f"Processed data summary:")
        logger.info(f"  Grid data contains: {list(processed_data['grid_data'].keys())}")
        logger.info(f"  Seasons: {list(processed_data['seasons_profiles'].keys())}")
        
        buses_df = processed_data['grid_data']['buses']
        gens_df = processed_data['grid_data']['generators']
        loads_df = processed_data['grid_data']['loads']
        
        logger.info(f"  Buses: {len(buses_df)}")
        logger.info(f"  Generators: {len(gens_df)}")
        logger.info(f"  Generator types: {gens_df['type'].unique().tolist()}")
        logger.info(f"  Total generation capacity: {gens_df['capacity_mw'].sum()} MW")
        logger.info(f"  Loads: {len(loads_df)}")
        logger.info(f"  Total load: {loads_df['p_mw'].sum()} MW")
        
        # Print loads per bus
        loads_by_bus = loads_df.groupby('bus')['p_mw'].sum()
        logger.info("  Loads by bus:")
        for bus, load in loads_by_bus.items():
            logger.info(f"    Bus {bus}: {load} MW")
            
        # Print generators per bus
        gens_by_bus = gens_df.groupby('bus')['capacity_mw'].sum()
        logger.info("  Generation by bus:")
        for bus, cap in gens_by_bus.items():
            logger.info(f"    Bus {bus}: {cap} MW")
            
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return 1

    # Check that we have the necessary data
    if not processed_data or 'grid_data' not in processed_data or 'seasons_profiles' not in processed_data:
        logger.error("Error: Incomplete processed data.")
        return 1

    #------------------------------------------------------------------
    # 2) CREATE THE INTEGRATED NETWORK
    #    We'll build an IntegratedNetwork from the preprocessed data.
    #------------------------------------------------------------------
    logger.info("Step 2: Building integrated network object...")

    # Extract analysis info (years, discount, etc.)
    analysis = processed_data['grid_data'].get('analysis', {})
    planning_horizon = analysis.get('planning_horizon', {})
    # We assume 'years' is now a list of relative years [1,2,3,...]
    years = planning_horizon.get('years', [1])  # fallback if missing
    system_discount_rate = planning_horizon.get('system_discount_rate', 0.0)
    
    # Extract load growth factors from analysis.json
    load_growth = processed_data['grid_data'].get('analysis', {}).get('load_growth', {})
    load_growth_factors = {}
    
    # Convert string keys to integers for years
    for year_str, factor in load_growth.items():
        if year_str != 'description':  # Skip the description field
            try:
                year = int(year_str)
                load_growth_factors[year] = float(factor)
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid load growth entry: {year_str}={factor}")
    
    logger.info(f"Load growth factors: {load_growth_factors}")

    # Create an IntegratedNetwork instance
    integrated_network = IntegratedNetwork(
        seasons=list(processed_data['seasons_profiles'].keys()),
        years=years,
        discount_rate=system_discount_rate,
        season_weights=SEASON_WEIGHTS
    )
    
    # Add load growth factors to the integrated network
    integrated_network.load_growth = load_growth_factors
    
    logger.info(f"Created IntegratedNetwork with:")
    logger.info(f"  Years: {years}")
    logger.info(f"  Seasons: {integrated_network.seasons}")
    logger.info(f"  Discount rate: {system_discount_rate}")
    logger.info(f"  Season weights: {SEASON_WEIGHTS}")

    # For each season, build a sub-network
    for season, season_data in processed_data['seasons_profiles'].items():
        logger.info(f"Building network for season: {season}")
        network = Network(name=season)

        # Add buses from grid data
        buses_df = processed_data['grid_data']['buses']
        for idx, row in buses_df.iterrows():
            network.buses.loc[row['id']] = {
                'name': row['name'],
                'v_nom': row.get('v_nom', 1.0)
            }

        # Add lines
        lines_df = processed_data['grid_data']['lines']
        for idx, row in lines_df.iterrows():
            network.lines.loc[row['id']] = {
                'name': row['name'],
                'from_bus': row['bus_from'],
                'to_bus': row['bus_to'],
                'susceptance': row['susceptance'],
                's_nom': row['capacity_mw']
            }

        # Add generators
        gens_df = processed_data['grid_data']['generators']
        for idx, row in gens_df.iterrows():
            network.generators.loc[row['id']] = {
                'name': row['name'],
                'bus': row['bus'],
                'p_nom': row['capacity_mw'],
                'marginal_cost': row['cost_mwh'],
                'type': row['type'],
                'capex_per_mw': row['capex_per_mw'],
                'lifetime_years': row['lifetime_years']
            }

        # Add loads
        loads_df = processed_data['grid_data']['loads']
        for idx, row in loads_df.iterrows():
            network.loads.loc[row['id']] = {
                'name': row['name'],
                'bus': row['bus'],  # Using 'bus' to match the loads.csv column name
                'p_mw': row['p_mw']
            }

        # Add storage - Note: The file is named 'storages.csv' in the data dir
        storage_df = processed_data['grid_data'].get('storage_units', processed_data['grid_data'].get('storages', pd.DataFrame()))
        for idx, row in storage_df.iterrows():
            network.storage_units.loc[row['id']] = {
                'name': row['name'],
                'bus': row['bus'],
                'p_nom': row['p_mw'],
                'efficiency_store': row['efficiency_store'],
                'efficiency_dispatch': row['efficiency_dispatch'],
                'max_hours': (row['energy_mwh'] / row['p_mw']) if row['p_mw'] else 0,
                'capex_per_mw': row['capex_per_mw'],
                'lifetime_years': row['lifetime_years']
            }

        # Set up time snapshots for this season
        T_hours = season_data['hours']  # e.g. 168 for one week
        network.create_snapshots(start_time="2023-01-01", periods=T_hours, freq='h')
        logger.info(f"Created {T_hours} snapshots for season {season}")

        # If there's a generator or load time series, incorporate them
        if 'loads' in season_data:
            loads_ts = season_data['loads']  # multi-index: (time, load_id)
            # Get unique load IDs from the timeseries
            unique_load_ids = loads_ts.index.levels[1]
            logger.info(f"Adding load timeseries for {len(unique_load_ids)} loads in season {season}")
            for load_id in unique_load_ids:
                # Grab the timeseries for this load
                series_mask = loads_ts.xs(load_id, level='load_id')['p_pu']
                # Convert p_pu * nominal
                # We'll get the nominal from network.loads[load_id]['p_mw']
                try:
                    p_nominal = network.loads.loc[load_id, 'p_mw']
                    p_series = series_mask.values * p_nominal
                    network.add_load_time_series(load_id, p_series)
                except KeyError:
                    logger.warning(f"Failed to add load timeseries for load {load_id} - not found in network loads")
                    pass  # if mismatch

        # Process generator profiles for wind and solar
        if 'generators' in season_data:
            gens_ts = season_data['generators']
            if not gens_ts.empty:
                # Get unique generator IDs from the timeseries
                unique_gen_ids = gens_ts.index.levels[1]
                
                # Count how many generators of each type we're adding profiles for
                gen_types = {}
                for gen_id in unique_gen_ids:
                    if gen_id in network.generators.index:
                        gen_type = network.generators.at[gen_id, 'type']
                        gen_types[gen_type] = gen_types.get(gen_type, 0) + 1
                
                logger.info(f"Adding generator profiles for {len(unique_gen_ids)} generators in season {season}")
                logger.info(f"Generator types with profiles: {gen_types}")
                
                for gen_id in unique_gen_ids:
                    try:
                        # Check if this generator exists in the network
                        if gen_id not in network.generators.index:
                            logger.warning(f"Generator {gen_id} not found in network generators, skipping")
                            continue
                            
                        # Get the generator type
                        gen_type = network.generators.at[gen_id, 'type']
                        
                        # Only add profiles for wind and solar generators
                        if gen_type in ['wind', 'solar']:
                            # Grab the availability profile (already in MW)
                            p_max_values = gens_ts.xs(gen_id, level='gen_id')['p_max_pu'].values
                            
                            # Add the profile to the network
                            network.add_generator_time_series(gen_id, p_max_values)
                            logger.info(f"Added profile for {gen_type} generator {gen_id} in season {season}")
                        else:
                            logger.info(f"Skipping profile for thermal generator {gen_id}")
                    except Exception as e:
                        logger.warning(f"Failed to add generator profile for {gen_id}: {e}")
                        logger.warning(traceback.format_exc())

        # Add the sub-network to the integrated structure
        integrated_network.add_season_network(season, network)

    #------------------------------------------------------------------
    # 3) OPTIMIZATION
    #    Solve the multi-year investment problem (new approach).
    #------------------------------------------------------------------
    logger.info("\nStep 3: Solving the multi-year model...")
    try:
        # Log solver options
        logger.info(f"Solver options: {args.solver_options}")
        
        result = solve_multi_year_investment(integrated_network, solver_options=args.solver_options)
        if not result or result.get('status') not in ('optimal', 'optimal_inaccurate'):
            logger.error(f"Optimization failed or not optimal. Status: {result.get('status', 'unknown')}")
            return 1
        else:
            logger.info(f"Optimization completed successfully. Objective value: {result.get('value', 0):.2f}")
            
            # Print a summary of installation decisions
            if 'variables' in result:
                gen_installed = result['variables'].get('gen_installed', {})
                storage_installed = result['variables'].get('storage_installed', {})
                
                # Check if any generators or storage are selected
                gen_selected = [(g, y) for (g, y), val in gen_installed.items() if val > 0.5]
                storage_selected = [(s, y) for (s, y), val in storage_installed.items() if val > 0.5]
                
                logger.info(f"Generators installed: {len(gen_selected)}")
                logger.info(f"Storage units installed: {len(storage_selected)}")
                
                # In a realistic model, some things should be selected. If nothing is selected,
                # the model might need tuning.
                if not gen_selected and not storage_selected:
                    logger.warning("Warning: No generators or storage selected for installation.")
                    logger.warning("This may indicate that existing capacity is sufficient,")
                    logger.warning("or it may be a sign that the model needs adjustments.")
                    
                    # Log all the generation and storage variables to debug
                    logger.debug("Generator installation variables:")
                    for (g, y), val in gen_installed.items():
                        logger.debug(f"  Generator {g}, Year {y}: {val}")
                    
                    logger.debug("Storage installation variables:")
                    for (s, y), val in storage_installed.items():
                        logger.debug(f"  Storage {s}, Year {y}: {val}")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        logger.error(traceback.format_exc())
        return 1

    #------------------------------------------------------------------
    # 4) POST-PROCESSING
    #    e.g. Create an implementation plan and store results
    #------------------------------------------------------------------
    logger.info("\nStep 4: Post-processing results...")

    try:
        # Generate the implementation plan and save it
        implementation_plan = generate_implementation_plan(integrated_network, args.output_dir)
        if implementation_plan:
            logger.info("Implementation plan generated.")
            
            # Check contents of the plan
            generators_planned = len(implementation_plan.get('generators', {}))
            storage_planned = len(implementation_plan.get('storage', {}))
            
            if generators_planned > 0 or storage_planned > 0:
                logger.info(f"Plan includes {generators_planned} generators and {storage_planned} storage units")
            else:
                logger.info("Plan includes no new installations")
        else:
            logger.warning("No implementation plan returned.")
            
        # Generate and save profile plots
        logger.info("Generating seasonal resource profiles...")
        plot_files = plot_seasonal_profiles(integrated_network, args.output_dir)
        if plot_files:
            logger.info("Created profile plots:")
            for season, plot_file in plot_files.items():
                logger.info(f"  - {season}: {os.path.basename(plot_file)}")
        else:
            logger.warning("No profile plots were generated.")
            
    except Exception as e:
        logger.error(f"Error in post-processing: {e}")
        logger.error(traceback.format_exc())
        return 1

    # Optionally save the network
    if args.save_network:
        integrated_network.save_to_pickle(os.path.join(args.output_dir, 'integrated_network.pkl'))
        logger.info("Optimized integrated network saved to integrated_network.pkl")

    logger.info("\nAll steps completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())