#!/usr/bin/env python3

"""
planning.py

Test script for the DCOPF investment planning model which creates an
installation plan across the planning horizon.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Add the scripts directory to the Python path
sys.path.insert(0, script_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(project_root, 'results', 'dcopf_planning')
os.makedirs(results_dir, exist_ok=True)

# Import our investment DCOPF planning implementation
from scripts.optimization import investment_dcopf_planning
from scripts.investment import create_test_system

def plot_installation_timeline(planning_results):
    """
    Plot the installation timeline across the planning horizon.
    
    Args:
        planning_results: Results from investment_dcopf_planning
    """
    installation_timeline = planning_results['installation_timeline']
    start_year = planning_results['start_year']
    planning_horizon = planning_results['planning_horizon']
    asset_lifetimes = planning_results['asset_lifetimes']
    
    # Create a colormap for different installation actions
    action_colors = {
        'Install': 'green',
        'Reinstall': 'blue',
        'Retire': 'red'
    }
    
    plt.figure(figsize=(14, 8))
    
    # Plot the installation timeline
    y_positions = {}
    for i, asset_id in enumerate(sorted(installation_timeline.keys())):
        y_pos = i + 1
        y_positions[asset_id] = y_pos
        
        # Plot asset lifetime spans
        for year, action in installation_timeline[asset_id].items():
            if action == 'Install' or action == 'Reinstall':
                # Calculate end of life
                lifetime = asset_lifetimes[asset_id]
                end_year = min(year + lifetime, start_year + planning_horizon)
                
                # Draw a bar from installation to end of life
                plt.barh(y_pos, end_year - year, left=year, height=0.5, 
                         color='lightgray', alpha=0.7)
        
        # Plot installation/reinstall/retire events
        for year, action in installation_timeline[asset_id].items():
            plt.scatter(year, y_pos, color=action_colors[action], s=100, 
                       marker='o' if action != 'Retire' else 'x', 
                       label=f"{action}" if f"{action}" not in plt.gca().get_legend_handles_labels()[1] else "")
            
            plt.text(year, y_pos + 0.2, action, ha='center', va='bottom', fontsize=9)
    
    # Set plot properties
    plt.yticks(list(y_positions.values()), [f"Asset {asset_id}" for asset_id in y_positions.keys()])
    plt.xlabel('Year')
    plt.ylabel('Asset')
    plt.title('Installation Timeline Across Planning Horizon')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits and ticks
    plt.xlim(start_year - 0.5, start_year + planning_horizon + 0.5)
    plt.xticks(range(start_year, start_year + planning_horizon + 1))
    
    # Add legend for actions
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                         label=action) for action, color in action_colors.items()]
    plt.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(results_dir, 'installation_timeline.png')
    plt.savefig(output_path)
    print(f"Installation timeline plot saved to: {output_path}")
    plt.close()

def plot_active_assets_by_year(planning_results):
    """
    Plot the active assets by year.
    
    Args:
        planning_results: Results from investment_dcopf_planning
    """
    active_assets = planning_results['active_assets_by_year']
    start_year = planning_results['start_year']
    planning_horizon = planning_results['planning_horizon']
    
    # Create a dataframe for easier plotting
    years = []
    asset_ids = []
    statuses = []
    
    for year in range(start_year, start_year + planning_horizon):
        assets_this_year = active_assets.get(year, [])
        for asset_id in planning_results['investment_required']:
            years.append(year)
            asset_ids.append(asset_id)
            statuses.append(1 if asset_id in assets_this_year else 0)
    
    df = pd.DataFrame({
        'Year': years,
        'Asset': asset_ids,
        'Active': statuses
    })
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Asset', columns='Year', values='Active')
    
    plt.figure(figsize=(14, 8))
    
    # Plot heatmap
    plt.imshow(pivot_df.values, cmap='YlGn', aspect='auto', interpolation='none')
    
    # Set axis labels and ticks
    plt.xlabel('Year')
    plt.ylabel('Asset')
    plt.yticks(range(len(pivot_df.index)), [f"Asset {asset_id}" for asset_id in pivot_df.index])
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
    
    # Add color bar
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_ticklabels(['Inactive', 'Active'])
    
    plt.title('Active Assets by Year')
    plt.grid(False)
    
    # Add text annotations to the heatmap
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            if pivot_df.iloc[i, j] == 1:
                plt.text(j, i, "Active", ha="center", va="center", color="darkgreen")
            else:
                plt.text(j, i, "Inactive", ha="center", va="center", color="darkred")
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(results_dir, 'active_assets_heatmap.png')
    plt.savefig(output_path)
    print(f"Active assets heatmap saved to: {output_path}")
    plt.close()

def run_test_planning():
    """Run the test for the DCOPF investment planning model."""
    print("Creating test system...")
    
    # Create 24 hours of data for a typical day
    start_time = datetime(2023, 1, 1)
    time_periods = [start_time + timedelta(hours=h) for h in range(24)]
    
    gen_time_series, branch, bus, demand_time_series = create_test_system(
        time_periods=time_periods
    )
    
    # Define asset lifetimes and capex from the generator data
    asset_lifetimes = {}
    asset_capex = {}
    
    for asset_id in gen_time_series['id'].unique():
        asset_row = gen_time_series[gen_time_series['id'] == asset_id].iloc[0]
        asset_lifetimes[asset_id] = asset_row['lifetime']
        asset_capex[asset_id] = asset_row['capex']
    
    # We'll use a longer planning horizon to observe reinstallations
    planning_horizon = 20  # 20 years planning horizon
    
    print(f"\n===== Running DCOPF Investment Planning Model ({planning_horizon} Years) =====")
    print("Note: Nuclear generator at bus 1 is mandatory, all other assets are optional")
    
    planning_results = investment_dcopf_planning(
        gen_time_series=gen_time_series,
        branch=branch,
        bus=bus,
        demand_time_series=demand_time_series,
        planning_horizon=planning_horizon,
        asset_lifetimes=asset_lifetimes,
        asset_capex=asset_capex,
        start_year=2023,
        delta_t=1  # 1 hour time steps
    )
    
    if planning_results is None:
        print("Investment planning optimization failed.")
        return
    
    # Plot the installation timeline
    plot_installation_timeline(planning_results)
    
    # Plot active assets by year
    plot_active_assets_by_year(planning_results)
    
    print(f"\nResults saved to: {results_dir}")
    
    return planning_results

if __name__ == "__main__":
    run_test_planning() 