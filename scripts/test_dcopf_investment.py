#!/usr/bin/env python3

"""
test_dcopf_investment.py

Test script for the integrated DCOPF investment model.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Add the scripts directory to the Python path
sys.path.insert(0, script_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(project_root, 'results', 'dcopf_investment')
os.makedirs(results_dir, exist_ok=True)

# Import our investment DCOPF implementation
from scripts.optimization import investment_dcopf

def create_test_system(time_periods=None):
    """
    Create a simple power system with mandatory nuclear and optional green assets.
    
    Args:
        time_periods: Optional list of time periods (default: 24 hours)
        
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    # Create time periods if not provided
    if time_periods is None:
        # Create 24 hours of data
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_periods = [start_time + timedelta(hours=h) for h in range(24)]
    
    # Create buses
    bus_data = [
        {'bus_i': 1, 'type': 3, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},  # Slack bus
        {'bus_i': 2, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
        {'bus_i': 3, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
        {'bus_i': 4, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
        {'bus_i': 5, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
    ]
    
    # Create branches with line limits
    branch_data = [
        {'fbus': 1, 'tbus': 2, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 100.0, 'sus': 10.0},
        {'fbus': 1, 'tbus': 3, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 100.0, 'sus': 10.0},
        {'fbus': 2, 'tbus': 4, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 100.0, 'sus': 10.0},
        {'fbus': 3, 'tbus': 4, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 100.0, 'sus': 10.0},
        {'fbus': 3, 'tbus': 5, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 100.0, 'sus': 10.0},
        {'fbus': 4, 'tbus': 5, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 100.0, 'sus': 10.0},
    ]
    
    # Create generators
    gen_data = []
    
    # 1. Mandatory nuclear generator at bus 1 (always active with non-zero cost)
    for t in time_periods:
        gen_data.append({
            'id': 1, 'time': t, 'bus': 1, 'pmin': 40, 'pmax': 120, 'gencost': 10,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 9,  # Nuclear: 9 years
            'capex': 0,  # $0 as it's mandatory/already installed
            'investment_required': 0  # 0 means already installed (mandatory)
        })
    
    # 2. Optional green generator at bus 1 (can be invested in)
    for t in time_periods:
        gen_data.append({
            'id': 2, 'time': t, 'bus': 1, 'pmin': 0, 'pmax': 120, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 8,  # 8 years
            'capex': 1500000,  # $1.5M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 3. Optional green generator at bus 2
    for t in time_periods:
        gen_data.append({
            'id': 3, 'time': t, 'bus': 2, 'pmin': 0, 'pmax': 80, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 7,  # 7 years
            'capex': 1200000,  # $1.2M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 4. Optional green generator at bus 3
    for t in time_periods:
        gen_data.append({
            'id': 4, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': 50, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 6,  # 6 years
            'capex': 900000,  # $0.9M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 5. Optional renewable generator at bus 3
    for t in time_periods:
        gen_data.append({
            'id': 5, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': 60, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 5,  # 5 years
            'capex': 800000,  # $0.8M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 6. Storage unit at bus 3
    for t in time_periods:
        gen_data.append({
            'id': 6, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': 40, 'gencost': 0,
            'emax': 160,  # 160 MWh energy capacity
            'einitial': 40,  # Start 25% charged
            'eta': 0.9,  # 90% round-trip efficiency
            'lifetime': 4,  # 4 years
            'capex': 350000,  # $0.35M/MW (based on power capacity)
            'investment_required': 1  # 1 means investment required
        })
    
    # 7. Storage unit at bus 4
    for t in time_periods:
        gen_data.append({
            'id': 7, 'time': t, 'bus': 4, 'pmin': 0, 'pmax': 50, 'gencost': 0,
            'emax': 200,  # 200 MWh energy capacity
            'einitial': 50,  # Start 25% charged
            'eta': 0.9,  # 90% round-trip efficiency
            'lifetime': 3,  # 3 years
            'capex': 300000,  # $0.3M/MW (based on power capacity)
            'investment_required': 1  # 1 means investment required
        })
    
    # Create time-varying demand
    demand_data = []
    
    # Base demand at each bus
    base_demand_bus4 = 50  # MW
    base_demand_bus5 = 100  # MW
    
    # Create daily pattern with morning and evening peaks
    for t in time_periods:
        hour = t.hour
        
        # Morning peak 7-9 AM, evening peak 6-8 PM
        if 7 <= hour < 10:
            factor = 1.5  # Morning peak
        elif 18 <= hour < 21:
            factor = 1.8  # Evening peak
        elif 0 <= hour < 5:
            factor = 0.7  # Night valley
        else:
            factor = 1.0  # Base load
        
        # Add demand at bus 4
        demand_data.append({
            'time': t,
            'bus': 4,
            'pd': base_demand_bus4 * factor
        })
        
        # Add demand at bus 5
        demand_data.append({
            'time': t,
            'bus': 5,
            'pd': base_demand_bus5 * factor
        })
    
    # Convert to DataFrames
    gen_time_series = pd.DataFrame(gen_data)
    branch = pd.DataFrame(branch_data)
    bus = pd.DataFrame(bus_data)
    demand_time_series = pd.DataFrame(demand_data)
    
    return gen_time_series, branch, bus, demand_time_series

def plot_investment_decisions(investment_results):
    """Plot the investment decisions."""
    decisions = investment_results['investment_decisions']
    
    # Create directory for plots if it doesn't exist
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    # Create bar chart
    assets = list(decisions.keys())
    values = list(decisions.values())
    
    # Color mapping: 1=green (selected), 0=red (not selected)
    colors = ['green' if v == 1 else 'red' for v in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(assets, values, color=colors)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 'Selected' if height == 1 else 'Not Selected',
                 ha='center', va='bottom')
    
    plt.ylim(0, 1.5)  # Set y-axis limit
    plt.xticks(assets, [f"Asset {asset}" for asset in assets])
    plt.ylabel('Decision (1=Selected, 0=Not Selected)')
    plt.title('Investment Decisions by Asset')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'plots', 'dcopf_investment_decisions.png'))
    plt.close()

def plot_generation(investment_results):
    """Plot the generation by asset and time."""
    gen_df = investment_results['generation']
    
    # Create a directory for plots if it doesn't exist
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    # Pivot the data for plotting
    pivot_df = gen_df.pivot(index='time', columns='id', values='gen')
    
    # Identify storage and non-storage units
    storage_ids = []
    non_storage_ids = []
    
    for asset_id in pivot_df.columns:
        # Check if the asset has both positive and negative values (storage)
        if (pivot_df[asset_id] > 0).any() and (pivot_df[asset_id] < 0).any():
            storage_ids.append(asset_id)
        else:
            non_storage_ids.append(asset_id)
    
    # 1. Plot generation (non-storage assets and discharging storage)
    plt.figure(figsize=(12, 6))
    
    # Create copy for plotting generation only
    gen_only_df = pivot_df.copy()
    
    # For storage assets, replace negative values (charging) with zeros
    for asset_id in storage_ids:
        if asset_id in gen_only_df.columns:
            gen_only_df[asset_id] = gen_only_df[asset_id].clip(lower=0)
    
    # Plot stacked area chart for generation
    gen_only_df.plot(kind='area', stacked=True, ax=plt.gca())
    
    plt.xlabel('Time')
    plt.ylabel('Generation (MW)')
    plt.title('Generation and Discharge by Asset and Time')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Asset ID')
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'plots', 'dcopf_generation.png'))
    plt.close()
    
    # 2. Plot consumption (charging storage only)
    if storage_ids:
        plt.figure(figsize=(12, 6))
        
        # Create dataframe for plotting storage charging only
        storage_df = pd.DataFrame(index=pivot_df.index)
        
        # For storage assets, replace positive values (discharging) with zeros and take absolute value of charging
        for asset_id in storage_ids:
            if asset_id in pivot_df.columns:
                storage_df[f'Storage {asset_id}'] = -pivot_df[asset_id].clip(upper=0)
        
        # Plot stacked area chart for storage charging
        if not storage_df.empty:
            storage_df.plot(kind='area', stacked=True, ax=plt.gca())
            
            plt.xlabel('Time')
            plt.ylabel('Charging (MW)')
            plt.title('Storage Charging by Asset and Time')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Storage ID')
            
            # Save the plot
            plt.savefig(os.path.join(results_dir, 'plots', 'dcopf_storage_charging.png'))
            plt.close()
    
    # 3. Plot storage state of charge if available
    if 'storage_soc' in investment_results:
        soc_df = investment_results['storage_soc']
        
        if not soc_df.empty:
            plt.figure(figsize=(12, 6))
            
            # Pivot the data for plotting
            soc_pivot = soc_df.pivot(index='time', columns='id', values='soc')
            soc_pivot.plot(ax=plt.gca())
            
            plt.xlabel('Time')
            plt.ylabel('State of Charge (MWh)')
            plt.title('Storage State of Charge by Time')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Storage ID')
            
            # Save the plot
            plt.savefig(os.path.join(results_dir, 'plots', 'dcopf_storage_soc.png'))
            plt.close()

def run_test():
    """Run the test for the investment DCOPF model."""
    print("Creating test system...")
    
    # Create 24 hours of data
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
    
    print("\n===== Running Integrated DCOPF Investment Model =====")
    print("Note: Nuclear generator at bus 1 is mandatory, all other assets are optional")
    
    investment_results = investment_dcopf(
        gen_time_series=gen_time_series,
        branch=branch,
        bus=bus,
        demand_time_series=demand_time_series,
        planning_horizon=10,  # 10 years planning horizon
        asset_lifetimes=asset_lifetimes,
        asset_capex=asset_capex,
        delta_t=1  # 1 hour time steps
    )
    
    if investment_results is None:
        print("Investment optimization failed.")
        return
    
    # Print investment decisions
    print("\n===== Investment Decisions =====")
    for asset_id, decision in investment_results['investment_decisions'].items():
        decision_text = "SELECTED" if decision == 1 else "NOT selected"
        print(f"Asset {asset_id}: {decision_text}")
    
    # Print cost summary
    print("\n===== Cost Summary =====")
    print(f"Total objective cost: ${investment_results['cost']:,.2f}")
    print(f"Investment cost: ${investment_results['investment_cost']:,.2f}")
    print(f"Operational cost: ${investment_results['operational_cost']:,.2f}")
    
    # Plot results
    plot_investment_decisions(investment_results)
    plot_generation(investment_results)
    
    print(f"\nResults saved to: {results_dir}")
    
    return investment_results

if __name__ == "__main__":
    run_test() 