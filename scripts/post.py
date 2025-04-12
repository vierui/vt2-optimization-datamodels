#!/usr/bin/env python3
"""
Post-processing module for the simplified multi-year approach.

We demonstrate:
  - Implementation plan generation
  - Possibly cost breakdown if desired
  - Visualization of profiles for different generator types
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# A custom JSON encoder to handle e.g. numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_implementation_plan(integrated_network, output_dir="results"):
    """
    Build a simple plan describing in which year each generator or storage 
    is actually installed (binary=1).
    Then optionally we can parse any dispatch results if needed.

    integrated_network: 
      - Should have an 'integrated_results' dict with a 'variables' sub-dict 
        containing numeric values (not CVXPY variables).

    Returns: A dictionary summarizing the plan, also saved as JSON in output_dir.
    """

    if not hasattr(integrated_network, 'integrated_results') or integrated_network.integrated_results is None:
        print("[post.py] No integrated_results found on integrated_network.")
        return {}

    result = integrated_network.integrated_results
    if 'variables' not in result:
        print("[post.py] integrated_results has no 'variables'.")
        return {}

    var_dict = result['variables']

    gen_installed = var_dict.get('gen_installed', {})
    storage_installed = var_dict.get('storage_installed', {})

    # Initialize the plan with defaults
    plan = {
        'years': integrated_network.years,
        'generators': {},
        'storage': {},
        'objective_value': result.get('value', 0),
        'status': result.get('status', 'unknown')
    }

    # Extract generator installation
    # gen_installed keys look like (g, y) -> value
    # We'll see in which year it's "1"
    for (g, y), val in gen_installed.items():
        # That means generator g is installed in year y
        if val > 0.5:
            if g not in plan['generators']:
                plan['generators'][g] = {
                    'years_installed': []
                }
            plan['generators'][g]['years_installed'].append(y)

    # Similarly for storage
    for (s, y), val in storage_installed.items():
        if val > 0.5:
            if s not in plan['storage']:
                plan['storage'][s] = {
                    'years_installed': []
                }
            plan['storage'][s]['years_installed'].append(y)

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    plan_file = os.path.join(output_dir, "implementation_plan.json")
    try:
        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2, cls=NumpyEncoder)
        print(f"[post.py] Implementation plan saved to {plan_file}")
    except Exception as e:
        print(f"[post.py] Error saving plan JSON: {e}")

    return plan


def save_cost_breakdown(integrated_network, output_dir="results"):
    """
    Optional: If you store cost breakdown by year or season, you can produce 
    a separate JSON. Here is a skeleton that you can adapt.
    """

    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No integrated_results to derive cost breakdown.")
        return {}

    result = integrated_network.integrated_results
    # If you had separate operational/capital cost in your solution, 
    # you'd read them from the model or parse from the variables. 
    # Suppose you do it manually or it's not separated. Just store total.
    cost_info = {
        'objective': result.get('value', None),
        'status': result.get('status', 'unknown')
        # Could add more details if your solver tracked them
    }

    cost_file = os.path.join(output_dir, "cost_breakdown.json")
    try:
        with open(cost_file, "w") as f:
            json.dump(cost_info, f, indent=2, cls=NumpyEncoder)
        print(f"[post.py] Cost breakdown saved to {cost_file}")
    except Exception as e:
        print(f"[post.py] Error saving cost breakdown: {e}")

    return cost_info


def plot_seasonal_profiles(integrated_network, output_dir="results"):
    """
    Create and save plots showing the total profiles for solar, wind, thermal, and loads
    for each season (particularly summer and winter).
    
    Args:
        integrated_network: IntegratedNetwork object with season networks
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with plot filenames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Seasons to plot (focus on summer and winter)
    seasons_to_plot = ['summer', 'winter']
    if 'spri_autu' in integrated_network.seasons:
        seasons_to_plot.append('spri_autu')
    
    # For each season, collect and plot the profiles
    for season in seasons_to_plot:
        if season not in integrated_network.season_networks:
            print(f"[post.py] Season {season} not found in network, skipping")
            continue
            
        network = integrated_network.season_networks[season]
        hours = range(1, network.T + 1)
        
        # Create a new figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Resource Profiles for {season.capitalize()} Season", fontsize=16)
        
        # Plot 1: Generator availability by type
        # Collect data for wind, solar, and thermal generators
        wind_total = np.zeros(network.T)
        solar_total = np.zeros(network.T)
        thermal_cap = 0
        
        # Get total capacity by generator type
        for gen_id in network.generators.index:
            gen_type = network.generators.at[gen_id, 'type']
            gen_cap = network.generators.at[gen_id, 'p_nom']
            
            if gen_type == 'wind' and gen_id in network.generators_t['p_max_pu']:
                profile = network.generators_t['p_max_pu'][gen_id].values
                wind_total += profile
            elif gen_type == 'solar' and gen_id in network.generators_t['p_max_pu']:
                profile = network.generators_t['p_max_pu'][gen_id].values
                solar_total += profile
            elif gen_type == 'thermal':
                thermal_cap += gen_cap
        
        # Plot generator availability
        ax1.plot(hours, wind_total, label='Wind', color='blue', linewidth=2)
        ax1.plot(hours, solar_total, label='Solar', color='orange', linewidth=2)
        ax1.axhline(y=thermal_cap, label='Thermal', color='brown', linestyle='-', linewidth=2)
        
        ax1.set_ylabel('Available Capacity (MW)')
        ax1.set_title('Generator Availability Profiles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis to show integers
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Plot 2: Total load profile
        # Collect total load for each hour
        total_load = np.zeros(network.T)
        for bus in network.buses.index:
            # Get the load at this bus for each hour
            bus_load = np.zeros(network.T)
            for load_id in network.loads.index:
                if network.loads.at[load_id, 'bus'] == bus:
                    if load_id in network.loads_t['p']:
                        bus_load += network.loads_t['p'][load_id].values
                    else:
                        bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
            total_load += bus_load
        
        # Plot total load
        ax2.plot(hours, total_load, label='Total Load', color='red', linewidth=2)
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Load (MW)')
        ax2.set_title('Total System Load Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set common x-axis limits and labels
        ax1.set_xlim(1, network.T)
        ax2.set_xlim(1, network.T)
        
        # Finalize and save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        plot_file = os.path.join(output_dir, f"{season}_profiles.png")
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        plot_files[season] = plot_file
        print(f"[post.py] Created profile plot for {season} at {plot_file}")
        
    # Create a combined plot comparing seasons
    if len(seasons_to_plot) > 1:
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle("Comparison of Seasonal Resource Profiles", fontsize=16)
        
        # Dictionary to store data for each season
        season_data = {}
        
        # Collect data for all seasons
        for season in seasons_to_plot:
            if season not in integrated_network.season_networks:
                continue
                
            network = integrated_network.season_networks[season]
            wind_total = np.zeros(network.T)
            solar_total = np.zeros(network.T)
            thermal_cap = 0
            total_load = np.zeros(network.T)
            
            # Collect generator data
            for gen_id in network.generators.index:
                gen_type = network.generators.at[gen_id, 'type']
                gen_cap = network.generators.at[gen_id, 'p_nom']
                
                if gen_type == 'wind' and gen_id in network.generators_t['p_max_pu']:
                    profile = network.generators_t['p_max_pu'][gen_id].values
                    wind_total += profile
                elif gen_type == 'solar' and gen_id in network.generators_t['p_max_pu']:
                    profile = network.generators_t['p_max_pu'][gen_id].values
                    solar_total += profile
                elif gen_type == 'thermal':
                    thermal_cap += gen_cap
            
            # Collect load data
            for bus in network.buses.index:
                bus_load = np.zeros(network.T)
                for load_id in network.loads.index:
                    if network.loads.at[load_id, 'bus'] == bus:
                        if load_id in network.loads_t['p']:
                            bus_load += network.loads_t['p'][load_id].values
                        else:
                            bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
                total_load += bus_load
            
            season_data[season] = {
                'wind': wind_total,
                'solar': solar_total,
                'thermal': thermal_cap,
                'load': total_load,
                'hours': range(1, network.T + 1)
            }
        
        # Plot wind profiles for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[0].plot(season_data[season]['hours'], season_data[season]['wind'], 
                           label=season.capitalize(), linewidth=2)
        axs[0].set_title('Wind Availability Profiles')
        axs[0].set_ylabel('Available Capacity (MW)')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot solar profiles for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[1].plot(season_data[season]['hours'], season_data[season]['solar'], 
                           label=season.capitalize(), linewidth=2)
        axs[1].set_title('Solar Availability Profiles')
        axs[1].set_ylabel('Available Capacity (MW)')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot thermal capacity for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[2].axhline(y=season_data[season]['thermal'], 
                              label=f"{season.capitalize()} ({season_data[season]['thermal']} MW)", 
                              linewidth=2)
        axs[2].set_title('Thermal Capacity')
        axs[2].set_ylabel('Capacity (MW)')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # Plot load profiles for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[3].plot(season_data[season]['hours'], season_data[season]['load'], 
                           label=season.capitalize(), linewidth=2)
        axs[3].set_title('Total System Load Profiles')
        axs[3].set_xlabel('Hour')
        axs[3].set_ylabel('Load (MW)')
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
        
        # Set x-axis limits for all subplots
        for ax in axs:
            if len(season_data) > 0:
                first_season = next(iter(season_data))
                ax.set_xlim(1, len(season_data[first_season]['hours']))
        
        # Finalize and save the combined plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        combined_plot_file = os.path.join(output_dir, "seasonal_profiles_comparison.png")
        plt.savefig(combined_plot_file, dpi=300)
        plt.close(fig)
        
        plot_files['comparison'] = combined_plot_file
        print(f"[post.py] Created combined seasonal profile comparison at {combined_plot_file}")
    
    return plot_files