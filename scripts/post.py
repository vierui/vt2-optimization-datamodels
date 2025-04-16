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

    # Extract installation variables from result_vars
    # The variables are stored as strings like "gen_installed_1001_1"
    gen_installed = {}
    storage_installed = {}
    
    # Parse the variables based on their keys
    for key, value in var_dict.items():
        if key.startswith('gen_installed_'):
            # Parse the generator and year from the key
            parts = key.split('_')
            if len(parts) >= 3:
                g = parts[2]  # Generator ID
                y = int(parts[3])  # Year
                gen_installed[(g, y)] = value
        elif key.startswith('storage_installed_'):
            # Parse the storage and year from the key
            parts = key.split('_')
            if len(parts) >= 3:
                s = parts[2]  # Storage ID
                y = int(parts[3])  # Year
                storage_installed[(s, y)] = value

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
        # Initialize total_load array
        total_load = np.zeros(network.T)
        
        # Collect total load for each hour
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


def plot_annual_load_growth(integrated_network, output_dir="results"):
    """
    Create and save a plot showing how the total system load grows over the planning horizon
    based on the load growth factors defined in analysis.json.
    
    Args:
        integrated_network: IntegratedNetwork object with season networks and load growth factors
        output_dir: Directory to save output plot
    
    Returns:
        Path to the generated plot file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if load growth factors are available
    if not hasattr(integrated_network, 'load_growth') or not integrated_network.load_growth:
        print("[post.py] No load growth factors found on integrated_network")
        return None
    
    # Get years and load growth factors
    years = integrated_network.years
    load_growth_factors = integrated_network.load_growth
    
    # Choose a representative season to extract the base load
    # We'll use the first season in the list for simplicity
    if not integrated_network.seasons or not integrated_network.season_networks:
        print("[post.py] No seasons found in integrated_network")
        return None
    
    representative_season = integrated_network.seasons[0]
    network = integrated_network.season_networks[representative_season]
    
    # Calculate base load (sum of load across all buses for the representative period)
    base_load_by_bus = {}
    total_base_load = 0
    
    for bus in network.buses.index:
        # Get the load at this bus for each hour
        bus_load = np.zeros(network.T)
        for load_id in network.loads.index:
            if network.loads.at[load_id, 'bus'] == bus:
                if 'p' in network.loads_t and load_id in network.loads_t['p']:
                    bus_load += network.loads_t['p'][load_id].values[:network.T]
                else:
                    bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
        
        # Store the bus load for future reference
        base_load_by_bus[bus] = bus_load
        
        # Add to total base load
        total_base_load += np.sum(bus_load)
    
    # Calculate total load for each year by applying growth factors
    total_load_by_year = {}
    avg_load_by_year = {}
    
    for y in years:
        growth_factor = load_growth_factors.get(y, 1.0)
        total_load_by_year[y] = total_base_load * growth_factor
        avg_load_by_year[y] = total_load_by_year[y] / network.T  # Average MW over the period
    
    # Create a plot of annual load growth
    plt.figure(figsize=(10, 6))
    
    # Plot total load (MWh over representative period)
    plt.subplot(2, 1, 1)
    plt.plot(years, [total_load_by_year[y] for y in years], 'o-', linewidth=2, color='blue')
    plt.title('Annual Load Growth - Total Energy')
    plt.xlabel('Year')
    plt.ylabel('Total Load (MWh over representative period)')
    plt.grid(True, alpha=0.3)
    
    # Plot average load (MW)
    plt.subplot(2, 1, 2)
    plt.plot(years, [avg_load_by_year[y] for y in years], 'o-', linewidth=2, color='red')
    plt.title('Annual Load Growth - Average Power')
    plt.xlabel('Year')
    plt.ylabel('Average Load (MW)')
    plt.grid(True, alpha=0.3)
    
    # Add the growth factor as text annotations for a few selected years
    # (first, middle, and last year)
    selected_years = [years[0], years[len(years)//2], years[-1]]
    for y in selected_years:
        growth_factor = load_growth_factors.get(y, 1.0)
        plt.annotate(f'{growth_factor:.2f}x',
                     xy=(y, avg_load_by_year[y]),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center')
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "annual_load_growth.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    print(f"[post.py] Annual load growth plot saved to {plot_file}")
    
    # Also print the data as a table for reference
    print("\nAnnual Load Growth Summary:")
    print("Year | Growth Factor | Total Load (MWh) | Average Load (MW)")
    print("-" * 60)
    for y in years:
        growth_factor = load_growth_factors.get(y, 1.0)
        print(f"{y:4d} | {growth_factor:13.4f} | {total_load_by_year[y]:15.2f} | {avg_load_by_year[y]:16.2f}")
    
    return plot_file


def plot_generation_mix(integrated_network, output_dir="results"):
    """
    Create and save plots showing how the load is met by different generator types
    (thermal, wind, solar) for winter and summer seasons in years 1 and 10.
    
    The plot shows stacked area charts representing the generation mix hour by hour.
    
    Args:
        integrated_network: IntegratedNetwork object with season networks and optimization results
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with plot filenames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Check if we have optimization results
    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No optimization results found")
        return plot_files
        
    result = integrated_network.integrated_results
    if 'variables' not in result or not result['variables']:
        print("[post.py] Error: Optimization results not found or empty.")
        return plot_files
    
    # Get years - we want to compare first and last year
    if not integrated_network.years or len(integrated_network.years) < 2:
        print("[post.py] Error: Need at least 2 years in the planning horizon.")
        return plot_files
        
    first_year = integrated_network.years[0]
    last_year = integrated_network.years[-1]
    
    # Seasons to plot
    seasons_to_plot = ['winter', 'summer']
    
    # Get load growth factors
    load_growth_factors = getattr(integrated_network, 'load_growth', {})
    growth_factor_first_year = load_growth_factors.get(first_year, 1.0)
    growth_factor_last_year = load_growth_factors.get(last_year, 1.0)
    
    # For each season
    for season in seasons_to_plot:
        if season not in integrated_network.season_networks:
            print(f"[post.py] Season {season} not found in network, skipping")
            continue
            
        network = integrated_network.season_networks[season]
        hours = range(1, network.T + 1)
        
        # Collect generator types and IDs
        thermal_gens = []
        wind_gens = []
        solar_gens = []
        
        for gen_id in network.generators.index:
            gen_type = network.generators.at[gen_id, 'type']
            if gen_type == 'thermal':
                thermal_gens.append(gen_id)
            elif gen_type == 'wind':
                wind_gens.append(gen_id)
            elif gen_type == 'solar':
                solar_gens.append(gen_id)
        
        # Create arrays to store dispatch by generator type for each year
        thermal_dispatch_y1 = np.zeros(network.T)
        wind_dispatch_y1 = np.zeros(network.T)
        solar_dispatch_y1 = np.zeros(network.T)
        
        thermal_dispatch_y10 = np.zeros(network.T)
        wind_dispatch_y10 = np.zeros(network.T)
        solar_dispatch_y10 = np.zeros(network.T)
        
        # Calculate total load for both years
        total_load_y1 = np.zeros(network.T)
        total_load_y10 = np.zeros(network.T)
        
        # Get the load at each bus for each hour
        for bus in network.buses.index:
            bus_load = np.zeros(network.T)
            for load_id in network.loads.index:
                if network.loads.at[load_id, 'bus'] == bus:
                    if 'p' in network.loads_t and load_id in network.loads_t['p']:
                        bus_load += network.loads_t['p'][load_id].values[:network.T]
                    else:
                        bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
            
            # Apply load growth factors
            total_load_y1 += bus_load * growth_factor_first_year
            total_load_y10 += bus_load * growth_factor_last_year
        
        # Sum up generator dispatch by type for each hour
        for hour in range(1, network.T + 1):
            # Year 1
            for gen_id in thermal_gens:
                dispatch_key = f"p_gen_{season}_{gen_id}_{first_year}_{hour}"
                dispatch = result['variables'].get(dispatch_key, 0)
                thermal_dispatch_y1[hour-1] += dispatch
                
            for gen_id in wind_gens:
                dispatch_key = f"p_gen_{season}_{gen_id}_{first_year}_{hour}"
                dispatch = result['variables'].get(dispatch_key, 0)
                wind_dispatch_y1[hour-1] += dispatch
                
            for gen_id in solar_gens:
                dispatch_key = f"p_gen_{season}_{gen_id}_{first_year}_{hour}"
                dispatch = result['variables'].get(dispatch_key, 0)
                solar_dispatch_y1[hour-1] += dispatch
                
            # Year 10
            for gen_id in thermal_gens:
                dispatch_key = f"p_gen_{season}_{gen_id}_{last_year}_{hour}"
                dispatch = result['variables'].get(dispatch_key, 0)
                thermal_dispatch_y10[hour-1] += dispatch
                
            for gen_id in wind_gens:
                dispatch_key = f"p_gen_{season}_{gen_id}_{last_year}_{hour}"
                dispatch = result['variables'].get(dispatch_key, 0)
                wind_dispatch_y10[hour-1] += dispatch
                
            for gen_id in solar_gens:
                dispatch_key = f"p_gen_{season}_{gen_id}_{last_year}_{hour}"
                dispatch = result['variables'].get(dispatch_key, 0)
                solar_dispatch_y10[hour-1] += dispatch
        
        # Get actual storage charge and discharge from optimization results
        storage_units = list(network.storage_units.index)
        storage_charge_y1 = np.zeros(network.T)
        storage_discharge_y1 = np.zeros(network.T)
        storage_charge_y10 = np.zeros(network.T)
        storage_discharge_y10 = np.zeros(network.T)
        
        for hour in range(1, network.T + 1):
            # Year 1
            for stor_id in storage_units:
                charge_key = f"p_charge_{season}_{stor_id}_{first_year}_{hour}"
                discharge_key = f"p_discharge_{season}_{stor_id}_{first_year}_{hour}"
                
                charge = result['variables'].get(charge_key, 0)
                discharge = result['variables'].get(discharge_key, 0)
                
                storage_charge_y1[hour-1] -= charge  # Negative because charging reduces net generation
                storage_discharge_y1[hour-1] += discharge
            
            # Year 10
            for stor_id in storage_units:
                charge_key = f"p_charge_{season}_{stor_id}_{last_year}_{hour}"
                discharge_key = f"p_discharge_{season}_{stor_id}_{last_year}_{hour}"
                
                charge = result['variables'].get(charge_key, 0)
                discharge = result['variables'].get(discharge_key, 0)
                
                storage_charge_y10[hour-1] -= charge  # Negative because charging reduces net generation
                storage_discharge_y10[hour-1] += discharge
        
        # VERIFICATION: Check if load is always met using the actual optimizer variables
        print(f"\nLOAD BALANCE VERIFICATION FOR {season.upper()} SEASON:")
        
        # Year 1 verification
        total_gen_y1 = thermal_dispatch_y1 + wind_dispatch_y1 + solar_dispatch_y1
        total_with_storage_y1 = total_gen_y1 + storage_discharge_y1 + storage_charge_y1  # Charge is already negative
        
        # Calculate the maximum absolute error
        max_error_y1 = np.max(np.abs(total_with_storage_y1 - total_load_y1))
        avg_error_y1 = np.mean(np.abs(total_with_storage_y1 - total_load_y1))
        
        print(f"  Year {first_year} - Maximum load balance error: {max_error_y1:.6f} MW")
        print(f"  Year {first_year} - Average load balance error: {avg_error_y1:.6f} MW")
        
        if max_error_y1 > 0.01:  # If error is more than 0.01 MW
            print("  CAUTION: Load might not be fully met in some hours (Year 1)")
            # Print the hours with largest errors
            worst_hours_y1 = np.argsort(np.abs(total_with_storage_y1 - total_load_y1))[-5:]
            print("  Hours with largest imbalance (Year 1):")
            for h in worst_hours_y1:
                print(f"    Hour {h+1}: Load={total_load_y1[h]:.2f} MW, Generation+Storage={total_with_storage_y1[h]:.2f} MW, Diff={total_with_storage_y1[h]-total_load_y1[h]:.2f} MW")
        else:
            print(f"  Year {first_year} - Load is met for all hours within tolerance")
            
        # Year 10 verification
        total_gen_y10 = thermal_dispatch_y10 + wind_dispatch_y10 + solar_dispatch_y10
        total_with_storage_y10 = total_gen_y10 + storage_discharge_y10 + storage_charge_y10  # Charge is already negative
        
        # Calculate the maximum absolute error
        max_error_y10 = np.max(np.abs(total_with_storage_y10 - total_load_y10))
        avg_error_y10 = np.mean(np.abs(total_with_storage_y10 - total_load_y10))
        
        print(f"  Year {last_year} - Maximum load balance error: {max_error_y10:.6f} MW")
        print(f"  Year {last_year} - Average load balance error: {avg_error_y10:.6f} MW")
        
        if max_error_y10 > 0.01:  # If error is more than 0.01 MW
            print("  CAUTION: Load might not be fully met in some hours (Year 10)")
            # Print the hours with largest errors
            worst_hours_y10 = np.argsort(np.abs(total_with_storage_y10 - total_load_y10))[-5:]
            print("  Hours with largest imbalance (Year 10):")
            for h in worst_hours_y10:
                print(f"    Hour {h+1}: Load={total_load_y10[h]:.2f} MW, Generation+Storage={total_with_storage_y10[h]:.2f} MW, Diff={total_with_storage_y10[h]-total_load_y10[h]:.2f} MW")
        else:
            print(f"  Year {last_year} - Load is met for all hours within tolerance")
        
        # Create stacked area chart for Year 1
        plt.figure(figsize=(14, 8))
        
        # Plot load as a reference area (filled to make it clear the load must be met)
        plt.fill_between(hours, 0, total_load_y1, color='lightgray', alpha=0.3, label='Required Load')
        
        # Plot the generation stack
        plt.stackplot(hours, 
                     thermal_dispatch_y1, 
                     wind_dispatch_y1, 
                     solar_dispatch_y1,
                     storage_discharge_y1,
                     labels=['Thermal', 'Wind', 'Solar', 'Storage (Discharge)'],
                     colors=['brown', 'blue', 'gold', 'green'],
                     alpha=0.7)
        
        # Plot load as a solid line for comparison
        plt.plot(hours, total_load_y1, 'r-', linewidth=2.5, label='Load')
        
        # Plot storage charging as negative values (storage_charge_y1 is already negative)
        plt.fill_between(hours, 0, storage_charge_y1, color='purple', alpha=0.3, label='Storage (Charge)')
        
        # Plot the sum of generation (should exactly match load)
        plt.plot(hours, total_with_storage_y1, 'k--', linewidth=1.5, label='Net Generation')
        
        plt.title(f'Generation Mix for {season.capitalize()} Season - Year {first_year}', fontsize=16)
        plt.xlabel('Hour', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        # Add a text box with verification results
        plt.text(0.02, 0.02, f"Maximum balance error: {max_error_y1:.6f} MW\nAverage balance error: {avg_error_y1:.6f} MW", 
                transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the plot
        y1_plot_file = os.path.join(output_dir, f"{season}_mix_year{first_year}.png")
        plt.savefig(y1_plot_file, dpi=300)
        plt.close()
        
        plot_files[f"{season}_year{first_year}"] = y1_plot_file
        print(f"[post.py] Created generation mix plot for {season} year {first_year} at {y1_plot_file}")
        
        # Create stacked area chart for Year 10
        plt.figure(figsize=(14, 8))
        
        # Plot load as a reference area (filled to make it clear the load must be met)
        plt.fill_between(hours, 0, total_load_y10, color='lightgray', alpha=0.3, label='Required Load')
        
        # Plot the generation stack
        plt.stackplot(hours, 
                     thermal_dispatch_y10, 
                     wind_dispatch_y10, 
                     solar_dispatch_y10,
                     storage_discharge_y10,
                     labels=['Thermal', 'Wind', 'Solar', 'Storage (Discharge)'],
                     colors=['brown', 'blue', 'gold', 'green'],
                     alpha=0.7)
        
        # Plot load as a solid line for comparison
        plt.plot(hours, total_load_y10, 'r-', linewidth=2.5, label='Load')
        
        # Plot storage charging as negative values (storage_charge_y10 is already negative)
        plt.fill_between(hours, 0, storage_charge_y10, color='purple', alpha=0.3, label='Storage (Charge)')
        
        # Plot the sum of generation (should exactly match load)
        plt.plot(hours, total_with_storage_y10, 'k--', linewidth=1.5, label='Net Generation')
        
        plt.title(f'Generation Mix for {season.capitalize()} Season - Year {last_year}', fontsize=16)
        plt.xlabel('Hour', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        # Add a text box with verification results
        plt.text(0.02, 0.02, f"Maximum balance error: {max_error_y10:.6f} MW\nAverage balance error: {avg_error_y10:.6f} MW", 
                transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the plot
        y10_plot_file = os.path.join(output_dir, f"{season}_mix_year{last_year}.png")
        plt.savefig(y10_plot_file, dpi=300)
        plt.close()
        
        plot_files[f"{season}_year{last_year}"] = y10_plot_file
        print(f"[post.py] Created generation mix plot for {season} year {last_year} at {y10_plot_file}")
    
    return plot_files

# Comment out the generate_bus_asset_report function as it's no longer needed
'''
def generate_bus_asset_report(integrated_network, output_dir="results"):
    """
    Generate a detailed report showing the first and last 5 hours of each season,
    listing all assets (generators, storage, loads) at each bus with their actual values
    ***for the first year of the planning horizon***.
    
    Args:
        integrated_network: IntegratedNetwork object with season networks and optimization results
        output_dir: Directory to save the report
    
    Returns:
        Path to the generated report file
    """
    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No optimization results found")
        return None
        
    result = integrated_network.integrated_results
    if 'variables' not in result or not result['variables']:
        print("[post.py] Error: Optimization results not found or empty.")
        return None
        
    # Get the first year of the planning horizon
    if not integrated_network.years:
        print("[post.py] Error: No years defined in integrated_network.")
        return None
    first_year = integrated_network.years[0]
    
    # Get load growth factor for the first year
    load_growth_factors = getattr(integrated_network, 'load_growth', {})
    growth_factor_first_year = load_growth_factors.get(first_year, 1.0) # Default to 1.0 if not found

    # Check if variables exist
    report_file = os.path.join(output_dir, "bus_asset_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"BUS ASSET REPORT - FIRST AND LAST 5 HOURS PER SEASON (YEAR {first_year})\n") # Indicate year
        f.write("=" * 80 + "\n\n")
        
        # For each season
        for season in integrated_network.seasons:
            f.write(f"\nSEASON: {season.upper()}\n")
            f.write("-" * 40 + "\n")
            
            network = integrated_network.season_networks[season]
            T = network.T
            
            # Limit hours shown to avoid excessive output
            hours_to_show = list(range(1, 6)) + list(range(T-4, T+1)) # First 5 and last 5 hours
            
            for hour in hours_to_show:
                f.write(f"\nHour {hour}:\n")
                
                # For each bus
                for bus_id in network.buses.index:
                    f.write(f"\n  Bus {bus_id}:\n")
                    
                    hourly_total_gen = 0
                    hourly_total_stor_dispatch = 0
                    hourly_total_load = 0

                    # List generators at this bus and sum dispatch
                    bus_gens = network.generators[network.generators['bus'] == bus_id]
                    if not bus_gens.empty:
                        f.write("    Generators:\n")
                        for gen_id in bus_gens.index:
                            gen_type = network.generators.at[gen_id, 'type']
                            gen_cap = network.generators.at[gen_id, 'p_nom']
                            
                            # Get the actual dispatch value from optimization results for the first year
                            dispatch_key = f"p_gen_{season}_{gen_id}_{first_year}_{hour}"
                            dispatch = result['variables'].get(dispatch_key, 0) # Default to 0 if key not found
                            hourly_total_gen += dispatch
                            
                            # For wind/solar, also show time-varying availability
                            # For thermal, show nominal capacity as availability
                            availability = ""
                            if gen_type in ['wind', 'solar'] and 'p_max_pu' in network.generators_t and gen_id in network.generators_t['p_max_pu']:
                                # Ensure hour is within bounds (adjust if needed, assuming hour is 1-based index)
                                if 0 <= hour - 1 < len(network.generators_t['p_max_pu'][gen_id].values):
                                     avail = network.generators_t['p_max_pu'][gen_id].values[hour-1] * gen_cap
                                     availability = f" (Available: {avail:.2f} MW)"
                                else:
                                     availability = " (Availability data missing for hour)"
                            elif gen_type == 'thermal': # <<< Add thermal availability
                                availability = f" (Available: {gen_cap:.2f} MW)" # Use nominal capacity
                            
                            f.write(f"      {gen_id} ({gen_type}): {dispatch:.2f} MW{availability}\n") # Ensure MW is present
                    
                    # List storage at this bus and sum dispatch
                    bus_storage = network.storage_units[network.storage_units['bus'] == bus_id]
                    if not bus_storage.empty:
                        f.write("    Storage:\n")
                        for stor_id in bus_storage.index:
                            # Get storage dispatch and state of charge for the first year
                            charge_key = f"p_charge_{season}_{stor_id}_{first_year}_{hour}"
                            discharge_key = f"p_discharge_{season}_{stor_id}_{first_year}_{hour}"
                            soc_key = f"soc_{season}_{stor_id}_{first_year}_{hour}"
                            
                            charge = result['variables'].get(charge_key, 0) # Default to 0 if key not found
                            discharge = result['variables'].get(discharge_key, 0) # Default to 0 if key not found
                            soc = result['variables'].get(soc_key, 0) # Default to 0 if key not found
                            
                            # Net dispatch = discharge - charge
                            net_dispatch = discharge - charge
                            hourly_total_stor_dispatch += net_dispatch
                            
                            f.write(f"      {stor_id}: Dispatch={net_dispatch:.2f} MW, SOC={soc:.2f} MWh\n")
                    
                    # List loads at this bus and sum load
                    bus_loads = network.loads[network.loads['bus'] == bus_id]
                    if not bus_loads.empty:
                        f.write("    Loads:\n")
                        for load_id in bus_loads.index:
                            # Get the actual load value from input data
                            load_value = 0
                            # Ensure hour is within bounds (adjust if needed, assuming hour is 1-based index)
                            if 'p' in network.loads_t and load_id in network.loads_t['p']:
                                 if 0 <= hour - 1 < len(network.loads_t['p'][load_id].values):
                                     load_value = network.loads_t['p'][load_id].values[hour-1]
                                 else:
                                     f.write(f"      {load_id}: Error - Load data missing for hour {hour}\n")
                                     continue # Skip this load if data missing for the hour
                            else:
                                 load_value = network.loads.at[load_id, 'p_mw'] # Use static value if no time series
                            
                            # Apply load growth factor for the first year
                            scaled_load_value = load_value * growth_factor_first_year
                            hourly_total_load += scaled_load_value
                            
                            f.write(f"      {load_id}: {scaled_load_value:.2f} MW\n")
                        f.write(f"    Total Load: {hourly_total_load:.2f} MW\n")
                    else:
                         f.write("    Loads: None\n")
                         f.write(f"    Total Load: {hourly_total_load:.2f} MW\n")
                    
                    # Calculate and write hourly net injection
                    hourly_net_injection = hourly_total_gen + hourly_total_stor_dispatch - hourly_total_load
                    f.write(f"    Net Injection: {hourly_net_injection:.2f} MW\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    print(f"[post.py] Bus asset report saved to {report_file}")
    return report_file
'''

def plot_implementation_timeline(integrated_network, output_dir="results"):
    """
    Create a visual timeline showing when each asset (generator, storage) is 
    installed, replaced, and decommissioned over the planning horizon.
    
    Args:
        integrated_network: IntegratedNetwork object with implementation plan
        output_dir: Directory to save the output plot
    
    Returns:
        Path to the generated timeline plot
    """
    # Get the implementation plan
    plan = generate_implementation_plan(integrated_network, output_dir)
    if not plan or 'generators' not in plan or 'storage' not in plan:
        print("[post.py] Implementation plan not available or incomplete")
        return None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get years from the planning horizon
    years = integrated_network.years if hasattr(integrated_network, 'years') and integrated_network.years else plan.get('years', [])
    if not years:
        print("[post.py] No planning years found")
        return None
    
    # Select a representative season to get generator/storage metadata
    if not integrated_network.seasons or not integrated_network.season_networks:
        print("[post.py] No seasons found in integrated_network")
        return None
    
    representative_season = integrated_network.seasons[0]
    network = integrated_network.season_networks[representative_season]
    
    # Collect asset data
    assets = []
    
    # Process generators
    for gen_id, gen_data in plan['generators'].items():
        if not gen_data.get('years_installed'):
            continue
            
        # Get generator metadata
        gen_type = "unknown"
        gen_name = f"Generator {gen_id}"
        if gen_id in network.generators.index:
            gen_type = network.generators.at[gen_id, 'type']
            gen_name = network.generators.at[gen_id, 'name'] if 'name' in network.generators.columns else f"Generator {gen_id}"
        
        # Add generator to assets list
        assets.append({
            'id': gen_id,
            'name': gen_name,
            'type': gen_type,
            'category': 'generator',
            'years_installed': gen_data.get('years_installed', []),
            'lifetime': network.generators.at[gen_id, 'lifetime_years'] if gen_id in network.generators.index and 'lifetime_years' in network.generators.columns else 5
        })
    
    # Process storage
    for stor_id, stor_data in plan['storage'].items():
        if not stor_data.get('years_installed'):
            continue
            
        # Get storage metadata
        stor_name = f"Storage {stor_id}"
        if stor_id in network.storage_units.index:
            stor_name = network.storage_units.at[stor_id, 'name'] if 'name' in network.storage_units.columns else f"Storage {stor_id}"
        
        # Add storage to assets list
        assets.append({
            'id': stor_id,
            'name': stor_name,
            'type': 'storage',
            'category': 'storage',
            'years_installed': stor_data.get('years_installed', []),
            'lifetime': network.storage_units.at[stor_id, 'lifetime_years'] if stor_id in network.storage_units.index and 'lifetime_years' in network.storage_units.columns else 5
        })
    
    # Sort assets by category (generators first, then storage) and type
    assets.sort(key=lambda x: (0 if x['category'] == 'generator' else 1, x['type'], x['id']))
    
    # Calculate asset active periods based on installation year and lifetime
    for asset in assets:
        asset['active_periods'] = []
        asset['replacements'] = []
        asset['decommissions'] = []
        
        for install_year in asset['years_installed']:
            # The asset is active from installation year until its lifetime expires
            end_year = min(install_year + asset['lifetime'], years[-1] + 1)
            asset['active_periods'].append((install_year, end_year))
            
            # If the end year is within the planning horizon, it's decommissioned
            if end_year <= years[-1]:
                asset['decommissions'].append(end_year)
        
        # Identify replacements (an installation that happens after a decommission)
        for i in range(1, len(asset['years_installed'])):
            if asset['years_installed'][i] > asset['years_installed'][0]:
                asset['replacements'].append(asset['years_installed'][i])
    
    # Now create the plot
    # Determine how many assets we have to size the plot
    num_assets = len(assets)
    
    # Create a larger figure for better readability
    plt.figure(figsize=(16, max(8, 0.4 * num_assets)))
    
    # Set up the plot
    ax = plt.gca()
    
    # Define colors for different asset types
    colors = {
        'thermal': '#a1e8af',  # Light green
        'wind': '#a1e8af',     # Light green
        'solar': '#a1e8af',    # Light green
        'storage': '#a1d6e8',  # Light blue
        'unknown': '#cccccc'   # Gray
    }
    
    # Plot each asset
    y_positions = list(range(num_assets, 0, -1))
    asset_labels = []
    
    for i, asset in enumerate(assets):
        y_pos = y_positions[i]
        asset_labels.append(f"{asset['category'].title()[0]}: {asset['name']} ({asset['type']})")
        
        # Determine color based on category or type
        if asset['category'] == 'generator':
            color = colors.get(asset['type'], colors['unknown'])
        else:
            color = colors.get('storage', colors['unknown'])
        
        # Plot active periods as horizontal bars
        for start_year, end_year in asset['active_periods']:
            plt.axhspan(y_pos - 0.4, y_pos + 0.4, start_year, end_year, alpha=0.6, color=color)
        
        # Plot installation points
        for year in asset['years_installed']:
            plt.plot(year, y_pos, 'g^', markersize=10, markeredgecolor='black')
        
        # Plot replacement points (if any)
        for year in asset['replacements']:
            plt.plot(year, y_pos, 'yo', markersize=8, markeredgecolor='black')
        
        # Plot decommission points (if any)
        for year in asset['decommissions']:
            plt.plot(year, y_pos, 'rx', markersize=10, markeredgecolor='black', mew=2)
    
    # Set up the axis and labels
    plt.yticks(y_positions, asset_labels)
    plt.xlabel('Year')
    plt.ylabel('Assets')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to show all years in the planning horizon
    plt.xlim(years[0] - 0.5, years[-1] + 0.5)
    plt.xticks(years)
    
    # Fix y-axis to prevent auto-scaling
    plt.ylim(0.5, num_assets + 0.5)
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['thermal'], alpha=0.6, label='Generator Active'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['storage'], alpha=0.6, label='Storage Active'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, markeredgecolor='black', label='New Installation'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, markeredgecolor='black', label='Replacement'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=10, markeredgecolor='red', label='Decommission')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    
    plt.title('Implementation Plan - Asset Timeline', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    timeline_file = os.path.join(output_dir, "asset_timeline.png")
    plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[post.py] Asset timeline plot saved to {timeline_file}")
    return timeline_file