#!/usr/bin/env python3
"""
Script to check the optimization results
"""
import os
import sys
import numpy as np
import pandas as pd

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.network import Network

def check_network_results(season):
    """
    Check the optimization results for a specific season
    
    Args:
        season: Season name (winter, summer, spri_autu)
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results/annual")
    
    # Load the network
    network_file = os.path.join(results_dir, f"{season}_network.pkl")
    network = Network.load_from_pickle(network_file)
    
    if network is None:
        print(f"Error: Could not load network for {season} season")
        return
    
    # Print basic information
    print(f"\n=== {season.upper()} SEASON RESULTS ===")
    print(f"Objective value: {network.objective_value}")
    print(f"Number of hours: {network.T}")
    
    # Print generator data columns for debugging
    print("\nGenerator data columns:", list(network.generators.columns))
    
    # Check generator dispatch
    if hasattr(network, 'generators_t') and 'p' in network.generators_t:
        gen_dispatch = network.generators_t['p']
        
        print("\nGenerator dispatch summary:")
        for gen_id in gen_dispatch.columns:
            # Get generator info
            gen_data = network.generators.loc[gen_id]
            
            # Get generator type safely
            gen_type = gen_data.get('type', 'unknown') if 'type' in network.generators.columns else 'unknown'
            
            capacity = gen_data['capacity_mw']
            cost = gen_data['cost_mwh']
            
            # Calculate statistics
            total_dispatch = gen_dispatch[gen_id].sum()
            max_dispatch = gen_dispatch[gen_id].max()
            avg_dispatch = gen_dispatch[gen_id].mean()
            utilization = avg_dispatch / capacity if capacity > 0 else 0
            total_cost = cost * total_dispatch
            
            # Print summary
            print(f"\nGenerator {gen_id} ({gen_type}):")
            print(f"  Capacity: {capacity} MW")
            print(f"  Cost: {cost} EUR/MWh")
            print(f"  Total dispatch: {total_dispatch:.2f} MWh")
            print(f"  Max hourly dispatch: {max_dispatch:.2f} MW")
            print(f"  Avg hourly dispatch: {avg_dispatch:.2f} MW")
            print(f"  Capacity factor: {utilization:.2%}")
            print(f"  Total cost: {total_cost:.2f} EUR")
            
            # Print hourly dispatch for first 24 hours
            print("\n  Hourly dispatch (first 24 hours):")
            for t in range(min(24, network.T)):
                print(f"    Hour {t}: {gen_dispatch[gen_id].iloc[t]:.4f} MW")
    
    # Check load consumption
    print("\nLoad consumption:")
    total_load = 0
    for load_id in network.loads.index:
        load_data = network.loads.loc[load_id]
        load_p = load_data['p_mw']
        
        # Get hourly load values
        if hasattr(network, 'loads_t') and not network.loads_t.empty:
            if load_id in network.loads_t.columns:
                total_load_mwh = network.loads_t[load_id].sum()
                avg_load = network.loads_t[load_id].mean()
                peak_load = network.loads_t[load_id].max()
                
                print(f"\nLoad {load_id}:")
                print(f"  Nominal power: {load_p} MW")
                print(f"  Total consumption: {total_load_mwh:.2f} MWh")
                print(f"  Average consumption: {avg_load:.2f} MW")
                print(f"  Peak consumption: {peak_load:.2f} MW")
                
                # Add to total load
                total_load += total_load_mwh
    
    print(f"\nTotal system load: {total_load:.2f} MWh")

def main():
    """Main function"""
    for season in ['winter', 'summer', 'spri_autu']:
        check_network_results(season)

if __name__ == "__main__":
    main() 