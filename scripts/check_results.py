#!/usr/bin/env python3
"""
Check optimization results for each season
Verify generator dispatch and load consumption
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from network import Network

def check_seasonal_results(season, results_dir="results/annual"):
    """
    Check optimization results for a specific season
    
    Args:
        season: Season name ('winter', 'summer', or 'spri_autu')
        results_dir: Directory containing results
        
    Returns:
        Dictionary with statistics
    """
    # Path to pickle file
    network_file = os.path.join(results_dir, f"{season}_network.pkl")
    
    # Load network from pickle
    network = Network.load_from_pickle(network_file)
    if network is None:
        print(f"Error: Could not load network for {season} season from {network_file}")
        return None
    
    # Print objective value
    print(f"\n--- {season.upper()} SEASON RESULTS ---")
    print(f"Objective value: {network.objective_value:.2f}")
    
    # Generator dispatch statistics
    total_dispatch = 0
    print("\nGenerator dispatch (MWh):")
    for gen_id in network.generators.index:
        gen_sum = network.generators_t['p'][gen_id].sum()
        total_dispatch += gen_sum
        print(f"Generator {gen_id} ({network.generators.loc[gen_id, 'name']}): {gen_sum:.2f} MWh")
    
    print(f"\nTotal generator dispatch: {total_dispatch:.2f} MWh")
    
    # Storage statistics
    if hasattr(network, 'storage_units_t') and network.storage_units_t:
        print("\nStorage operation:")
        for s_id in network.storage_units.index:
            charge_sum = network.storage_units_t['p_charge'][s_id].sum()
            discharge_sum = network.storage_units_t['p_discharge'][s_id].sum()
            print(f"Storage {s_id} ({network.storage_units.loc[s_id, 'name']}): "
                  f"Charged {charge_sum:.2f} MWh, Discharged {discharge_sum:.2f} MWh")
    
    # Load consumption
    total_load = 0
    print("\nLoad consumption (MWh):")
    for load_id in network.loads.index:
        if load_id in network.loads_t.columns:
            load_sum = network.loads_t[load_id].sum()
            total_load += load_sum
            print(f"Load {load_id} ({network.loads.loc[load_id, 'name']}): "
                  f"Nominal power {network.loads.loc[load_id, 'p_mw']:.2f} MW, "
                  f"Total consumption {load_sum:.2f} MWh")
    
    print(f"\nTotal load consumption: {total_load:.2f} MWh")
    
    # Installation decisions
    if hasattr(network, 'generators_installed'):
        print("\nGenerator installation decisions:")
        for gen_id in network.generators.index:
            installed = network.generators_installed[gen_id] > 0.5
            capacity = network.generators.loc[gen_id, 'capacity_mw']
            gen_type = network.generators.loc[gen_id].get('type', 'thermal')
            print(f"Generator {gen_id} ({network.generators.loc[gen_id, 'name']}): "
                  f"Installed = {installed}, Capacity = {capacity:.2f} MW, Type = {gen_type}")
    
    if hasattr(network, 'storage_installed'):
        print("\nStorage installation decisions:")
        for s_id in network.storage_units.index:
            installed = network.storage_installed[s_id] > 0.5
            capacity = network.storage_units.loc[s_id, 'p_mw']
            energy = network.storage_units.loc[s_id, 'energy_mwh']
            print(f"Storage {s_id} ({network.storage_units.loc[s_id, 'name']}): "
                  f"Installed = {installed}, Power capacity = {capacity:.2f} MW, "
                  f"Energy capacity = {energy:.2f} MWh")
    
    # Line flows
    if hasattr(network, 'lines_t') and network.lines_t and 'p' in network.lines_t:
        print("\nAverage line flows (MW):")
        for line_id in network.lines.index:
            flows = network.lines_t['p'][line_id]
            avg_flow = flows.mean()
            max_flow = flows.max()
            capacity = network.lines.loc[line_id, 'capacity_mw']
            utilization = max_flow / capacity * 100 if capacity > 0 else 0
            print(f"Line {line_id}: Avg flow = {avg_flow:.2f} MW, "
                  f"Max flow = {max_flow:.2f} MW, "
                  f"Capacity = {capacity:.2f} MW, "
                  f"Max utilization = {utilization:.1f}%")
    
    return {
        'objective_value': network.objective_value,
        'total_dispatch': total_dispatch,
        'total_load': total_load
    }

def main():
    """
    Main function to check results for all seasons
    """
    # Path to results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results/annual")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return False
    
    # Check results for each season
    seasons = ['winter', 'summer', 'spri_autu']
    season_stats = {}
    
    for season in seasons:
        stats = check_seasonal_results(season, results_dir)
        if stats is not None:
            season_stats[season] = stats
    
    # Print summary
    if season_stats:
        print("\n--- SUMMARY ---")
        total_cost = 0
        for season, stats in season_stats.items():
            weight = 13 if season in ['winter', 'summer'] else 26
            weighted_cost = stats['objective_value'] * weight
            total_cost += weighted_cost
            print(f"{season.capitalize()}: Objective = {stats['objective_value']:.2f}, "
                  f"Generation = {stats['total_dispatch']:.2f} MWh, "
                  f"Load = {stats['total_load']:.2f} MWh, "
                  f"Weight = {weight} weeks, "
                  f"Weighted cost = {weighted_cost:.2f}")
        
        print(f"\nTotal annual cost: {total_cost:.2f}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 