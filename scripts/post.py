#!/usr/bin/env python3
"""
Post-processing module for analyzing and visualizing optimization results
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta

def plot_generation_mix(network, output_dir="results", day=10, save_fig=True, show_fig=True):
    """
    Plot the generation mix showing contribution of different generator types
    
    Args:
        network: Network object with optimization results
        output_dir: Directory to save plots
        day: Day of the year (to label x-axis with proper dates)
        save_fig: Whether to save the figure to a file
        show_fig: Whether to display the figure
        
    Returns:
        Figure object
    """
    if not hasattr(network, 'generators_t') or 'p' not in network.generators_t:
        raise ValueError("No generation results available. Run dcopf() first.")
    
    # Create output directory if it doesn't exist
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get generation results
    gen_results = network.generators_t['p']
    
    # Group generators by type
    gen_types = {}
    for gen_id, gen_data in network.generators.iterrows():
        gen_type = gen_data.get('type', 'thermal')
        if gen_type not in gen_types:
            gen_types[gen_type] = []
        gen_types[gen_type].append(gen_id)
    
    # Sum generation by type
    gen_by_type = pd.DataFrame(index=gen_results.index)
    for gen_type, gen_ids in gen_types.items():
        gen_by_type[gen_type] = gen_results[gen_ids].sum(axis=1)
    
    # Get total load
    load_results = pd.DataFrame(index=gen_results.index)
    load_results['total'] = 0
    for load_id in network.loads.index:
        if load_id in network.loads_t.columns:
            load_results['total'] += network.loads_t[load_id]
    
    # Get storage charging/discharging
    if hasattr(network, 'storage_units_t') and 'p_charge' in network.storage_units_t:
        storage_charge = network.storage_units_t['p_charge'].sum(axis=1)
        storage_discharge = network.storage_units_t['p_discharge'].sum(axis=1)
        
        # Add storage to generation mix
        gen_by_type['storage'] = storage_discharge
        
        # Add storage charging to the load
        load_results['storage_charging'] = storage_charge
    
    # Create x-axis with proper timestamps
    start_date = datetime(2023, 1, 1) + timedelta(days=day-1)
    timestamps = [start_date + timedelta(hours=h) for h in range(len(gen_results))]
    
    # Create stacked area plot for generation mix
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot generation by type (stacked area)
    bottom = np.zeros(len(gen_by_type))
    
    # Define colors for different generator types
    colors = {
        'thermal': '#E57373',   # Red
        'wind': '#64B5F6',      # Blue
        'solar': '#FFF176',     # Yellow
        'storage': '#81C784'    # Green
    }
    
    for gen_type in gen_by_type.columns:
        color = colors.get(gen_type, '#9E9E9E')  # Default to gray if type not in colors dict
        ax.fill_between(
            timestamps, 
            bottom, 
            bottom + gen_by_type[gen_type].values,
            label=f"{gen_type.capitalize()}",
            color=color,
            alpha=0.7
        )
        bottom += gen_by_type[gen_type].values
    
    # Plot load as a black line
    ax.plot(timestamps, load_results['total'], 'k--', linewidth=2, label='Load')
    
    # Format x-axis with hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Set nice-looking ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    # Set labels and title
    ax.set_xlabel(f'Hour of Day ({start_date.strftime("%Y-%m-%d")})')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Generation Mix by Generator Type')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to only use integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        plot_path = os.path.join(output_dir, f'generation_mix_day_{day}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Generation mix plot saved to: {plot_path}")
    
    # Show figure if requested
    if show_fig:
        plt.show()
    
    return fig

def plot_storage_operation(network, output_dir="results", day=10, save_fig=True, show_fig=True):
    """
    Plot storage charging, discharging, and state of charge
    
    Args:
        network: Network object with optimization results
        output_dir: Directory to save plots
        day: Day of the year (to label x-axis with proper dates)
        save_fig: Whether to save the figure to a file
        show_fig: Whether to display the figure
        
    Returns:
        Figure object
    """
    if not hasattr(network, 'storage_units_t') or 'state_of_charge' not in network.storage_units_t:
        raise ValueError("No storage results available. Run dcopf() first.")
    
    # Create output directory if it doesn't exist
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get storage results
    soc = network.storage_units_t['state_of_charge']
    charge = network.storage_units_t['p_charge']
    discharge = network.storage_units_t['p_discharge']
    
    # Sum over all storage units
    total_soc = soc.sum(axis=1)
    total_charge = charge.sum(axis=1)
    total_discharge = discharge.sum(axis=1)
    
    # Create x-axis with proper timestamps
    start_date = datetime(2023, 1, 1) + timedelta(days=day-1)
    timestamps = [start_date + timedelta(hours=h) for h in range(len(total_soc))]
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot charging and discharging on the left y-axis
    ax1.fill_between(timestamps, 0, total_discharge, label='Discharging', color='#FF7043', alpha=0.7)
    ax1.fill_between(timestamps, 0, -total_charge, label='Charging', color='#42A5F5', alpha=0.7)
    
    # Plot state of charge on the right y-axis
    ax2.plot(timestamps, total_soc, 'k-', label='State of Charge', linewidth=2)
    
    # Format x-axis with hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    # Set labels and title
    ax1.set_xlabel(f'Hour of Day ({start_date.strftime("%Y-%m-%d")})')
    ax1.set_ylabel('Power (MW)')
    ax2.set_ylabel('Energy (MWh)')
    ax1.set_title('Storage Operation')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        plot_path = os.path.join(output_dir, f'storage_operation_day_{day}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Storage operation plot saved to: {plot_path}")
    
    # Show figure if requested
    if show_fig:
        plt.show()
    
    return fig

def plot_load_profile(network, output_dir="results", day=10, save_fig=True, show_fig=True):
    """
    Plot load profile for each bus
    
    Args:
        network: Network object with optimization results
        output_dir: Directory to save plots
        day: Day of the year (to label x-axis with proper dates)
        save_fig: Whether to save the figure to a file
        show_fig: Whether to display the figure
        
    Returns:
        Figure object
    """
    if network.loads_t.empty:
        raise ValueError("No load data available.")
    
    # Create output directory if it doesn't exist
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get load data
    load_data = network.loads_t
    
    # Group loads by bus
    bus_loads = {}
    for load_id, load_data_row in network.loads.iterrows():
        bus_id = load_data_row['bus_id']
        if bus_id not in bus_loads:
            bus_loads[bus_id] = []
        bus_loads[bus_id].append(load_id)
    
    # Sum load by bus
    load_by_bus = pd.DataFrame(index=load_data.index)
    for bus_id, load_ids in bus_loads.items():
        # Filter for load IDs that are actually in the load_data DataFrame
        valid_load_ids = [lid for lid in load_ids if lid in load_data.columns]
        
        if valid_load_ids:
            bus_name = network.buses.loc[bus_id, 'name']
            load_by_bus[f"Bus {bus_id} ({bus_name})"] = load_data[valid_load_ids].sum(axis=1)
    
    # Check if we have any load data to plot
    if load_by_bus.empty:
        raise ValueError("No valid load data to plot.")
    
    # Create x-axis with proper timestamps
    start_date = datetime(2023, 1, 1) + timedelta(days=day-1)
    timestamps = [start_date + timedelta(hours=h) for h in range(len(load_data))]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot load for each bus
    for bus in load_by_bus.columns:
        ax.plot(timestamps, load_by_bus[bus], 'o-', linewidth=2, label=bus)
    
    # Format x-axis with hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    # Set labels and title
    ax.set_xlabel(f'Hour of Day ({start_date.strftime("%Y-%m-%d")})')
    ax.set_ylabel('Load (MW)')
    ax.set_title('Load Profile by Bus')
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        plot_path = os.path.join(output_dir, f'load_profile_day_{day}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Load profile plot saved to: {plot_path}")
    
    # Show figure if requested
    if show_fig:
        plt.show()
    
    return fig

def create_results_summary(network, output_dir="results", day=10):
    """
    Create and save a textual summary of optimization results
    
    Args:
        network: Network object with optimization results
        output_dir: Directory to save summary
        day: Day of the year (for filename)
        
    Returns:
        Path to the summary file
    """
    if not hasattr(network, 'objective_value'):
        raise ValueError("No optimization results available. Run dcopf() first.")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a summary string
    summary = ["# Optimization Results Summary\n"]
    
    # Add date information
    start_date = datetime(2023, 1, 1) + timedelta(days=day-1)
    summary.append(f"**Date:** {start_date.strftime('%Y-%m-%d')}\n")
    
    # Add objective value
    summary.append(f"**Total Cost:** {network.objective_value:.2f} EUR\n")
    
    # Add generation statistics
    if hasattr(network, 'generators_t') and 'p' in network.generators_t:
        gen_results = network.generators_t['p']
        
        summary.append("\n## Generation Statistics\n")
        summary.append("| Generator | Type | Capacity (MW) | Mean Output (MW) | Max Output (MW) | Total Energy (MWh) |\n")
        summary.append("|-----------|------|--------------|-----------------|----------------|------------------|\n")
        
        for gen_id, gen_data in network.generators.iterrows():
            gen_type = gen_data.get('type', 'thermal')
            capacity = gen_data['capacity_mw']
            mean_output = gen_results[gen_id].mean()
            max_output = gen_results[gen_id].max()
            total_energy = gen_results[gen_id].sum()
            
            summary.append(f"| {gen_data['name']} | {gen_type} | {capacity:.1f} | {mean_output:.1f} | {max_output:.1f} | {total_energy:.1f} |\n")
    
    # Add storage statistics
    if hasattr(network, 'storage_units_t') and 'state_of_charge' in network.storage_units_t:
        soc = network.storage_units_t['state_of_charge']
        charge = network.storage_units_t['p_charge']
        discharge = network.storage_units_t['p_discharge']
        
        summary.append("\n## Storage Statistics\n")
        summary.append("| Storage Unit | Capacity (MWh) | Mean SoC (MWh) | Min SoC (MWh) | Max SoC (MWh) | Total Charging (MWh) | Total Discharging (MWh) |\n")
        summary.append("|-------------|---------------|---------------|--------------|--------------|---------------------|----------------------|\n")
        
        for storage_id, storage_data in network.storage_units.iterrows():
            capacity = storage_data['energy_mwh']
            mean_soc = soc[storage_id].mean()
            min_soc = soc[storage_id].min()
            max_soc = soc[storage_id].max()
            total_charge = charge[storage_id].sum()
            total_discharge = discharge[storage_id].sum()
            
            summary.append(f"| {storage_data['name']} | {capacity:.1f} | {mean_soc:.1f} | {min_soc:.1f} | {max_soc:.1f} | {total_charge:.1f} | {total_discharge:.1f} |\n")
    
    # Add line flow statistics
    if hasattr(network, 'lines_t') and 'p' in network.lines_t:
        line_results = network.lines_t['p']
        
        summary.append("\n## Line Flow Statistics\n")
        summary.append("| Line | From Bus | To Bus | Capacity (MW) | Mean Flow (MW) | Max Flow (MW) | Min Flow (MW) |\n")
        summary.append("|------|----------|--------|--------------|---------------|--------------|-------------|\n")
        
        for line_id, line_data in network.lines.iterrows():
            from_bus = line_data['bus_from']
            to_bus = line_data['bus_to']
            capacity = line_data['capacity_mw']
            mean_flow = line_results[line_id].mean()
            max_flow = line_results[line_id].max()
            min_flow = line_results[line_id].min()
            
            summary.append(f"| {line_data['name']} | {from_bus} | {to_bus} | {capacity:.1f} | {mean_flow:.1f} | {max_flow:.1f} | {min_flow:.1f} |\n")
    
    # Write summary to file
    summary_path = os.path.join(output_dir, f'results_summary_day_{day}.md')
    with open(summary_path, 'w') as f:
        f.write(''.join(summary))
    
    print(f"Results summary saved to: {summary_path}")
    
    return summary_path

def analyze_and_plot_results(network, output_dir="results", day=10, show_plots=False):
    """
    Analyze optimization results and create plots and summaries
    
    Args:
        network: Network object with optimization results
        output_dir: Directory to save results
        day: Day of the year
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with paths to output files
    """
    output_files = {}
    
    # Create generation mix plot
    try:
        plot_generation_mix(network, output_dir, day, True, show_plots)
        output_files['generation_mix'] = os.path.join(output_dir, f'generation_mix_day_{day}.png')
    except Exception as e:
        print(f"Error creating generation mix plot: {e}")
    
    # Create storage operation plot
    try:
        plot_storage_operation(network, output_dir, day, True, show_plots)
        output_files['storage_operation'] = os.path.join(output_dir, f'storage_operation_day_{day}.png')
    except Exception as e:
        print(f"Error creating storage operation plot: {e}")
    
    # Create load profile plot
    try:
        plot_load_profile(network, output_dir, day, True, show_plots)
        output_files['load_profile'] = os.path.join(output_dir, f'load_profile_day_{day}.png')
    except Exception as e:
        print(f"Error creating load profile plot: {e}")
    
    # Create results summary
    try:
        summary_path = create_results_summary(network, output_dir, day)
        output_files['summary'] = summary_path
    except Exception as e:
        print(f"Error creating results summary: {e}")
    
    return output_files

if __name__ == "__main__":
    import argparse
    from network import Network
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze and visualize optimization results")
    parser.add_argument("--use-day-profiles", action="store_true", help="Use time-dependent profiles")
    parser.add_argument("--day", type=int, default=10, help="Day of the year to use (1-365)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--show-plots", action="store_true", help="Display plots")
    args = parser.parse_args()
    
    # Run optimization on example network with specified day profiles
    from main import build_example_network_from_csv, build_example_network_programmatically
    
    try:
        net = build_example_network_from_csv(use_day_profiles=args.use_day_profiles, day=args.day)
        data_source = "CSV files"
    except Exception as e:
        print(f"Could not load from CSV: {e}")
        print("Building network programmatically instead...")
        net = build_example_network_programmatically(use_day_profiles=args.use_day_profiles, day=args.day)
        data_source = "programmatic definition"
    
    print(f"Network structure created from {data_source}")
    
    # Run optimization
    print("Solving DC optimal power flow problem...")
    success = net.dcopf()
    
    if success:
        # Analyze and plot results
        print("Analyzing and plotting results...")
        output_files = analyze_and_plot_results(net, args.output_dir, args.day, args.show_plots)
        
        print(f"\nResults saved to: {args.output_dir}")
        for file_type, file_path in output_files.items():
            print(f"- {file_type}: {os.path.basename(file_path)}")
    else:
        print("Optimization failed!") 