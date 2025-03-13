"""
Functions for creating scenario-specific visualizations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_scenario_results(results, demand_time_series, branch, bus, scenario_folder, 
                         season_key, id_to_type, id_to_gencost, id_to_pmax):
    """Plot and analyze scenario results"""
    # Get unique time points
    time_points = sorted(demand_time_series['time'].unique())
    
    # Create scenario folder
    os.makedirs(scenario_folder, exist_ok=True)
    
    # Initialize data structures
    total_gen_per_asset = {}
    total_gen_cost_per_asset = {}
    remaining_capacity_series = {}
    winter_gen = {}
    summer_gen = {}
    
    # Process generation and demand data
    for time in time_points:
        # Get generation for this time point
        gen_at_t = results['generation'][results['generation']['time'] == time]
        
        # Debug print
        print(f"\nTime {time}:")
        print("Generation data:", gen_at_t[['id', 'gen']].to_string())
        
        # Aggregate generation by type
        for _, gen_row in gen_at_t.iterrows():
            gen_type = id_to_type.get(gen_row['id'])
            if gen_type:
                # For storage types, gen is already net output (discharge - charge)
                total_gen_per_asset[gen_type] = total_gen_per_asset.get(gen_type, 0) + gen_row['gen']
                
                # Calculate generation cost
                if gen_row['id'] in id_to_gencost:
                    cost = gen_row['gen'] * id_to_gencost[gen_row['id']]
                    total_gen_cost_per_asset[gen_type] = total_gen_cost_per_asset.get(gen_type, 0) + cost
    
    # # Debug print
    # print("\nTotal generation per asset type:")
    # for asset_type, total_gen in total_gen_per_asset.items():
    #     print(f"{asset_type}: {total_gen}")

    # Calculate remaining capacity
    for gen_type in set(id_to_type.values()):
        gen_ids = [id for id, typ in id_to_type.items() if typ == gen_type]
        total_pmax = sum(id_to_pmax.get(id, 0) for id in gen_ids)
        used_capacity = total_gen_per_asset.get(gen_type, 0)
        remaining = total_pmax - used_capacity if total_pmax > used_capacity else 0
        remaining_capacity_series[gen_type] = remaining

    # Store seasonal data
    if season_key == 'winter':
        winter_gen = total_gen_per_asset.copy()
    elif season_key == 'summer':
        summer_gen = total_gen_per_asset.copy()

    # Create figure directory
    figure_dir = os.path.join(scenario_folder, "figure")
    os.makedirs(figure_dir, exist_ok=True)

    # Get all unique asset types
    all_types = sorted(set(total_gen_per_asset.keys()))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot generation and remaining capacity
    x = np.arange(len(all_types))
    total_gen_values = [total_gen_per_asset.get(asset, 0) for asset in all_types]
    remaining_values = [remaining_capacity_series.get(asset, 0) for asset in all_types]
    
    plt.bar(x, total_gen_values, label='Generation', color='skyblue')
    plt.bar(x, remaining_values, bottom=total_gen_values, 
            label='Remaining Capacity', color='lightgray', alpha=0.7)
    
    # Customize plot
    plt.xlabel('Asset Type')
    plt.ylabel('Power (MW)')
    plt.title(f'Generation and Remaining Capacity - {season_key.capitalize()}')
    plt.xticks(x, all_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.8)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(figure_dir, f"generation_capacity_{season_key}.png"))
    plt.close()

    return (total_gen_per_asset, 
            total_gen_cost_per_asset,
            remaining_capacity_series,
            {})  # Empty - no need of gen_by_type 