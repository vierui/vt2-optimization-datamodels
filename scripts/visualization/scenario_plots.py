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
    
    # Initialize data structures for both seasons
    total_gen_per_asset = {}
    total_gen_cost_per_asset = {}
    remaining_capacity_series = {}
    
    # Store seasonal data
    if season_key == 'winter':
        winter_gen = {}
        winter_remaining = {}
    else:  # summer
        summer_gen = {}
        summer_remaining = {}
    
    # Process generation and demand data
    for time in time_points:
        # Get generation for this time point
        gen_at_t = results['generation'][results['generation']['time'] == time]
        
        # Aggregate generation by type
        for _, gen_row in gen_at_t.iterrows():
            gen_type = id_to_type.get(gen_row['id'])
            if gen_type:
                total_gen_per_asset[gen_type] = total_gen_per_asset.get(gen_type, 0) + gen_row['gen']
                
                # Calculate generation cost
                if gen_row['id'] in id_to_gencost:
                    cost = gen_row['gen'] * id_to_gencost[gen_row['id']]
                    total_gen_cost_per_asset[gen_type] = total_gen_cost_per_asset.get(gen_type, 0) + cost
    
    # Calculate remaining capacity
    for gen_type in set(id_to_type.values()):
        gen_ids = [id for id, typ in id_to_type.items() if typ == gen_type]
        total_pmax = sum(id_to_pmax.get(id, 0) for id in gen_ids)
        used_capacity = total_gen_per_asset.get(gen_type, 0)
        remaining = total_pmax - used_capacity if total_pmax > used_capacity else 0
        remaining_capacity_series[gen_type] = remaining

    # Store data for the current season
    if season_key == 'winter':
        winter_gen = total_gen_per_asset.copy()
        winter_remaining = remaining_capacity_series.copy()
    else:  # summer
        summer_gen = total_gen_per_asset.copy()
        summer_remaining = remaining_capacity_series.copy()

    # Create figure directory
    figure_dir = os.path.join(scenario_folder, "figure")
    os.makedirs(figure_dir, exist_ok=True)

    # Only create the plot when processing summer data (to have both seasons available)
    if season_key == 'summer':
        # Get all unique asset types
        all_types = sorted(set(winter_gen.keys()) | set(summer_gen.keys()))
        
        # Create arrays for plotting
        x = np.arange(len(all_types))
        width = 0.35

        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot winter data
        winter_gen_values = [winter_gen.get(asset, 0) for asset in all_types]
        winter_remaining_values = [winter_remaining.get(asset, 0) for asset in all_types]
        
        # Plot summer data
        summer_gen_values = [summer_gen.get(asset, 0) for asset in all_types]
        summer_remaining_values = [summer_remaining.get(asset, 0) for asset in all_types]

        # Create bars
        plt.bar(x - width/2, winter_gen_values, width, label='Winter Generation', color='lightblue')
        plt.bar(x - width/2, winter_remaining_values, width, bottom=winter_gen_values,
                label='Winter Remaining Capacity', color='lightgray', alpha=0.7)
        
        plt.bar(x + width/2, summer_gen_values, width, label='Summer Generation', color='orange')
        plt.bar(x + width/2, summer_remaining_values, width, bottom=summer_gen_values,
                label='Summer Remaining Capacity', color='lightyellow', alpha=0.7)

        # Customize plot
        plt.xlabel('Asset Type')
        plt.ylabel('Power (MW)')
        plt.title('Generation and Remaining Capacity by Season')
        plt.xticks(x, all_types, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(figure_dir, "generation_capacity_comparison.png"))
        plt.close()
    
    return (total_gen_per_asset, 
            total_gen_cost_per_asset,
            remaining_capacity_series,
            {}) # Empty dict since we don't need gen_by_type anymore 