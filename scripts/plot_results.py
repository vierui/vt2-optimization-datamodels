#!/usr/bin/env python3
"""
Plot generator availability and load profiles for each season

This script creates visualizations of generator availability and load profiles
from the optimization results to provide insights into the model weeks.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Define season colors for consistent visualization
SEASON_COLORS = {
    'winter': 'blue',
    'summer': 'orange',
    'spri_autu': 'green'
}

SEASON_NAMES = {
    'winter': 'Winter',
    'summer': 'Summer',
    'spri_autu': 'Spring/Autumn'
}

def load_network_results(results_dir):
    """
    Load network models from pickle files for all seasons
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary with network models for each season
    """
    networks = {}
    
    for season in ['winter', 'summer', 'spri_autu']:
        network_file = os.path.join(results_dir, f"{season}_network.pkl")
        if os.path.exists(network_file):
            try:
                with open(network_file, 'rb') as f:
                    networks[season] = pickle.load(f)
                print(f"Loaded {season} network model")
            except Exception as e:
                print(f"Error loading {season} network: {e}")
        else:
            print(f"Network file not found: {network_file}")
    
    return networks

def create_time_axis(hours, season):
    """
    Create a datetime axis for the season
    
    Args:
        hours: Number of hours in the season profile
        season: Season name
        
    Returns:
        Array of datetime objects
    """
    # Create start dates based on the season
    start_dates = {
        'winter': datetime(2023, 1, 2),  # First Monday of January
        'summer': datetime(2023, 7, 31), # Last Monday of July
        'spri_autu': datetime(2023, 10, 16) # Middle Monday of October
    }
    
    start_date = start_dates.get(season, datetime(2023, 1, 1))
    return [start_date + timedelta(hours=h) for h in range(hours)]

def plot_generator_availability(networks, output_dir):
    """
    Plot generator availability (p_max_pu) for each season
    
    Args:
        networks: Dictionary with network models for each season
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each season
    for season, network in networks.items():
        if not hasattr(network, 'gen_p_max_pu'):
            print(f"No generator availability data for {season}")
            continue
            
        plt.figure(figsize=(12, 8))
        
        for gen_id, profile in network.gen_p_max_pu.items():
            # Get generator type if available
            gen_type = "Unknown"
            if 'type' in network.generators.columns and gen_id in network.generators.index:
                gen_type = network.generators.at[gen_id, 'type']
            
            # Create time axis
            time_axis = create_time_axis(len(profile), season)
            
            # Plot availability profile
            plt.plot(time_axis, profile, label=f"Gen {gen_id} ({gen_type})")
        
        # Format the plot
        plt.title(f"Generator Availability - {SEASON_NAMES[season]}")
        plt.xlabel("Time")
        plt.ylabel("Availability (p.u.)")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis to show days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{season}_generator_availability.png"))
        plt.close()
        
        print(f"Created generator availability plot for {season}")
    
    # Plot all seasons on one graph for comparison
    plt.figure(figsize=(12, 8))
    
    for season, network in networks.items():
        if not hasattr(network, 'gen_p_max_pu'):
            continue
            
        # Calculate total available capacity at each hour
        total_capacity = np.zeros(network.T)
        
        for gen_id, profile in network.gen_p_max_pu.items():
            # Convert profile to numpy array if it's not already
            profile_array = np.array(profile, dtype=float)
            
            # Get generator capacity
            capacity = network.generators.at[gen_id, 'capacity_mw']
            
            # Add to total available capacity
            total_capacity += profile_array * capacity
        
        # Create time axis normalized to day of week for comparison
        hours_per_day = 24
        days = network.T // hours_per_day
        day_hours = list(range(days * hours_per_day))
        
        # Plot total available capacity
        plt.plot(
            day_hours, 
            total_capacity[:len(day_hours)], 
            label=f"{SEASON_NAMES[season]}", 
            color=SEASON_COLORS[season]
        )
    
    # Format the plot
    plt.title("Total Available Generation Capacity - All Seasons")
    plt.xlabel("Hour")
    plt.ylabel("Available Capacity (MW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add day markers
    for day in range(1, days+1):
        plt.axvline(x=day*hours_per_day, color='gray', linestyle='--', alpha=0.5)
        plt.text(day*hours_per_day, plt.ylim()[1]*0.95, f"Day {day}", 
                ha='center', va='top', backgroundcolor='white', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_seasons_generator_availability.png"))
    plt.close()
    
    print("Created all-seasons generator availability comparison plot")

def plot_load_profiles(networks, output_dir):
    """
    Plot load profiles for each season
    
    Args:
        networks: Dictionary with network models for each season
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each season
    for season, network in networks.items():
        if not hasattr(network, 'loads_t') or network.loads_t.empty:
            print(f"No load profile data for {season}")
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Create time axis
        time_axis = create_time_axis(len(network.loads_t), season)
        
        # Plot each load profile
        for load_id in network.loads_t.columns:
            # Get load data
            load_profile = network.loads_t[load_id].values
            
            # Get load information if available
            load_name = f"Load {load_id}"
            if 'name' in network.loads.columns and load_id in network.loads.index:
                load_name = network.loads.at[load_id, 'name']
            
            # Plot load profile
            plt.plot(time_axis, load_profile, label=load_name)
        
        # Format the plot
        plt.title(f"Load Profiles - {SEASON_NAMES[season]}")
        plt.xlabel("Time")
        plt.ylabel("Load (MW)")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis to show days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{season}_load_profiles.png"))
        plt.close()
        
        print(f"Created load profiles plot for {season}")
    
    # Plot all seasons on one graph for comparison
    plt.figure(figsize=(12, 8))
    
    for season, network in networks.items():
        if not hasattr(network, 'loads_t') or network.loads_t.empty:
            continue
            
        # Calculate total load at each hour
        total_load = network.loads_t.sum(axis=1).values
        
        # Create time axis normalized to day of week for comparison
        hours_per_day = 24
        days = len(total_load) // hours_per_day
        day_hours = list(range(days * hours_per_day))
        
        # Plot total load
        plt.plot(
            day_hours, 
            total_load[:len(day_hours)], 
            label=f"{SEASON_NAMES[season]}", 
            color=SEASON_COLORS[season]
        )
    
    # Format the plot
    plt.title("Total System Load - All Seasons")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add day markers
    for day in range(1, days+1):
        plt.axvline(x=day*hours_per_day, color='gray', linestyle='--', alpha=0.5)
        plt.text(day*hours_per_day, plt.ylim()[1]*0.95, f"Day {day}", 
                ha='center', va='top', backgroundcolor='white', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_seasons_load_profiles.png"))
    plt.close()
    
    print("Created all-seasons load profiles comparison plot")

def plot_generation_vs_load(networks, output_dir):
    """
    Plot generation vs load for each season
    
    Args:
        networks: Dictionary with network models for each season
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for season, network in networks.items():
        if not hasattr(network, 'loads_t') or network.loads_t.empty:
            print(f"No load data for {season}")
            continue
            
        if not hasattr(network, 'gen_p_max_pu'):
            print(f"No generator data for {season}")
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Create time axis
        time_axis = create_time_axis(len(network.loads_t), season)
        
        # Calculate total load
        total_load = network.loads_t.sum(axis=1).values
        
        # Calculate available generation
        total_available_gen = np.zeros(network.T)
        
        for gen_id, profile in network.gen_p_max_pu.items():
            # Convert profile to numpy array if it's not already
            profile_array = np.array(profile, dtype=float)
            
            # Get generator capacity
            capacity = network.generators.at[gen_id, 'capacity_mw']
            
            # Add to total available capacity
            total_available_gen += profile_array * capacity
        
        # Calculate actual generation (dispatch)
        if hasattr(network, 'generators_t') and 'p' in network.generators_t:
            total_dispatch = network.generators_t['p'].sum(axis=1).values
            plt.plot(time_axis, total_dispatch, label="Actual Dispatch", color='red', linewidth=2)
        
        # Plot load and available generation
        plt.plot(time_axis, total_load, label="Total Load", color='blue')
        plt.plot(time_axis, total_available_gen, label="Available Generation", color='green', linestyle='--')
        
        # Format the plot
        plt.title(f"Generation vs Load - {SEASON_NAMES[season]}")
        plt.xlabel("Time")
        plt.ylabel("Power (MW)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis to show days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{season}_generation_vs_load.png"))
        plt.close()
        
        print(f"Created generation vs load plot for {season}")

def plot_dispatch_by_generator_type(networks, output_dir):
    """
    Plot dispatch by generator type for each season
    
    Args:
        networks: Dictionary with network models for each season
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for season, network in networks.items():
        if not hasattr(network, 'generators_t') or 'p' not in network.generators_t:
            print(f"No generator dispatch data for {season}")
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Create time axis
        time_axis = create_time_axis(len(network.generators_t['p']), season)
        
        # Group generators by type if type information is available
        if 'type' in network.generators.columns:
            # Get unique generator types
            gen_types = network.generators['type'].unique()
            
            # Calculate dispatch by type
            dispatch_by_type = {}
            for gen_type in gen_types:
                dispatch_by_type[gen_type] = np.zeros(network.T)
                
                # Get generators of this type
                gens_of_type = network.generators[network.generators['type'] == gen_type].index
                
                # Sum dispatch for all generators of this type
                for gen_id in gens_of_type:
                    if gen_id in network.generators_t['p'].columns:
                        dispatch_by_type[gen_type] += network.generators_t['p'][gen_id].values
            
            # Plot dispatch by type
            for gen_type, dispatch in dispatch_by_type.items():
                plt.plot(time_axis, dispatch, label=f"{gen_type}")
        else:
            # If no type information, plot by individual generator
            for gen_id in network.generators_t['p'].columns:
                plt.plot(time_axis, network.generators_t['p'][gen_id].values, label=f"Gen {gen_id}")
        
        # Format the plot
        plt.title(f"Generator Dispatch - {SEASON_NAMES[season]}")
        plt.xlabel("Time")
        plt.ylabel("Dispatch (MW)")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis to show days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{season}_dispatch_by_type.png"))
        plt.close()
        
        print(f"Created dispatch by generator type plot for {season}")

def plot_installation_decisions(networks, output_dir):
    """
    Plot the installation decisions for generators and storage units
    
    Args:
        networks: Dictionary with network models for each season
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each season
    for season, network in networks.items():
        if not hasattr(network, 'generators_installed'):
            print(f"No installation decisions for {season}")
            continue
            
        # Create plot with two subplots: one for generators, one for costs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot generator installation decisions
        gen_names = []
        gen_installed = []
        gen_colors = []
        
        for gen_id in network.generators.index:
            installed = network.generators_installed.get(gen_id, 0) > 0.5
            gen_type = network.generators.at[gen_id, 'type'] if 'type' in network.generators.columns else 'unknown'
            
            # Get generator name
            gen_name = f"Gen {gen_id}"
            if 'name' in network.generators.columns:
                gen_name = network.generators.at[gen_id, 'name']
            
            gen_names.append(gen_name)
            gen_installed.append(1 if installed else 0)
            
            # Set color based on generator type
            if gen_type == 'wind':
                gen_colors.append('blue')
            elif gen_type == 'solar':
                gen_colors.append('orange')
            else:  # thermal
                gen_colors.append('red')
        
        # Add storage units if present
        storage_names = []
        storage_installed = []
        
        if hasattr(network, 'storage_installed'):
            for storage_id in network.storage_units.index:
                installed = network.storage_installed.get(storage_id, 0) > 0.5
                
                # Get storage name
                storage_name = f"Storage {storage_id}"
                if 'name' in network.storage_units.columns:
                    storage_name = network.storage_units.at[storage_id, 'name']
                
                storage_names.append(storage_name)
                storage_installed.append(1 if installed else 0)
        
        # Combine generators and storage for plotting
        all_names = gen_names + storage_names
        all_installed = gen_installed + storage_installed
        all_colors = gen_colors + ['green'] * len(storage_names)
        
        # Plot installation decisions as horizontal bar chart
        y_pos = range(len(all_names))
        ax1.barh(y_pos, all_installed, color=all_colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(all_names)
        ax1.set_xlabel('Installed (1) or Not Installed (0)')
        ax1.set_title(f'Installation Decisions - {SEASON_NAMES[season]}')
        
        # Plot cost breakdown
        # Operational costs
        operational_costs = {}
        for gen_id in network.generators.index:
            gen_name = f"Gen {gen_id}"
            if 'name' in network.generators.columns:
                gen_name = network.generators.at[gen_id, 'name']
            
            if gen_id in network.generators_t['p'].columns:
                gen_sum = network.generators_t['p'][gen_id].sum()
                gen_cost = network.generators.loc[gen_id, 'cost_mwh']
                operational_costs[gen_name] = gen_cost * gen_sum
        
        # CAPEX costs
        capex_costs = {}
        for gen_id in network.generators.index:
            if network.generators_installed.get(gen_id, 0) > 0.5:
                gen_name = f"Gen {gen_id}"
                if 'name' in network.generators.columns:
                    gen_name = network.generators.at[gen_id, 'name']
                
                capacity = network.generators.loc[gen_id, 'capacity_mw']
                capex_per_mw = network.generators.loc[gen_id, 'capex_per_mw']
                lifetime = network.generators.loc[gen_id, 'lifetime_years']
                capex_costs[gen_name] = (capex_per_mw * capacity) / lifetime
        
        for storage_id in network.storage_units.index:
            if hasattr(network, 'storage_installed') and network.storage_installed.get(storage_id, 0) > 0.5:
                storage_name = f"Storage {storage_id}"
                if 'name' in network.storage_units.columns:
                    storage_name = network.storage_units.at[storage_id, 'name']
                
                capacity = network.storage_units.loc[storage_id, 'p_mw']
                capex_per_mw = network.storage_units.loc[storage_id, 'capex_per_mw']
                lifetime = network.storage_units.loc[storage_id, 'lifetime_years']
                capex_costs[storage_name] = (capex_per_mw * capacity) / lifetime
        
        # Create bar chart for costs
        cost_names = list(operational_costs.keys()) + list(capex_costs.keys())
        cost_values = list(operational_costs.values()) + list(capex_costs.values())
        cost_types = ['OPEX'] * len(operational_costs) + ['CAPEX'] * len(capex_costs)
        cost_colors = ['blue'] * len(operational_costs) + ['red'] * len(capex_costs)
        
        # Plot cost breakdown
        y_pos = range(len(cost_names))
        ax2.barh(y_pos, cost_values, color=cost_colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(cost_names)
        ax2.set_xlabel('Cost (EUR)')
        ax2.set_title(f'Cost Breakdown - {SEASON_NAMES[season]}')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='OPEX'),
            Patch(facecolor='red', label='CAPEX')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{season}_installation_decisions.png"))
        plt.close()
        
        print(f"Created installation decisions plot for {season}")

def main():
    """
    Main function to create plots from optimization results
    """
    # Get directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results/annual")
    plots_dir = os.path.join(project_root, "results/plots")
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    print(f"Saving plots to: {plots_dir}")
    
    # Load network results
    networks = load_network_results(results_dir)
    if not networks:
        print("No network results found")
        return
    
    # Create plots
    plot_generator_availability(networks, plots_dir)
    plot_load_profiles(networks, plots_dir)
    plot_generation_vs_load(networks, plots_dir)
    plot_dispatch_by_generator_type(networks, plots_dir)
    plot_installation_decisions(networks, plots_dir)
    
    print(f"All plots have been saved to: {plots_dir}")

if __name__ == "__main__":
    main() 