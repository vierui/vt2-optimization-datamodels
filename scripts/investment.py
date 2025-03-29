#!/usr/bin/env python3

"""
investment.py

Test script for the integrated DCOPF investment model.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

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

def create_test_system(time_periods=None, data_mapping=None):
    """
    Create a simple power system with mandatory nuclear and optional renewable assets,
    using real load, solar and wind data from CSV files.
    
    Args:
        time_periods: Optional list of time periods (default: 24 hours)
        data_mapping: Dictionary mapping timestamps to load, solar, and wind data
        
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    # Verify that time_periods and data_mapping exist
    if not time_periods:
        raise ValueError("No time periods provided. This function requires real data timestamps.")
    
    if not data_mapping:
        raise ValueError("No data mapping provided. Please provide the real data mapping.")
    
    # Check if all time periods have entries in the mapping
    missing_timestamps = [t for t in time_periods if t not in data_mapping]
    if missing_timestamps:
        raise ValueError(f"Missing data for {len(missing_timestamps)} timestamps. First missing: {missing_timestamps[0]}")
    
    print(f"Using real data for {len(time_periods)} time periods")
    
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
        {'fbus': 1, 'tbus': 2, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 110.0, 'sus': 10.0},
        {'fbus': 1, 'tbus': 3, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 110.0, 'sus': 10.0},
        {'fbus': 2, 'tbus': 4, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 110.0, 'sus': 10.0},
        {'fbus': 3, 'tbus': 4, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 110.0, 'sus': 10.0},
        {'fbus': 3, 'tbus': 5, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 110.0, 'sus': 10.0},
        {'fbus': 4, 'tbus': 5, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': 110.0, 'sus': 10.0},
    ]
    
    # Create generators
    gen_data = []
    
    # Get raw values directly from data mapping
    load_values = [data_mapping[t]['load'] for t in time_periods]
    solar_values = [data_mapping[t]['solar'] for t in time_periods]
    wind_values = [data_mapping[t]['wind'] for t in time_periods]
    
    max_load = max(load_values)
    max_solar = max(solar_values)
    max_wind = max(wind_values)
    
    # Identify the season to help with debugging
    seasons = [data_mapping[t]['season'] for t in time_periods]
    unique_seasons = set(seasons)
    print(f"Data includes {len(unique_seasons)} seasons: {', '.join(unique_seasons)}")
    
    print(f"Maximum load value: {max_load:.2f}")
    print(f"Maximum solar value: {max_solar:.2f}")
    print(f"Maximum wind value: {max_wind:.2f}")
    
    # 1. Mandatory nuclear generator at bus 1 (always active with non-zero cost)
    for t in time_periods:
        gen_data.append({
            'id': 1, 'time': t, 'bus': 1, 'pmin': 40, 'pmax': 100, 'gencost': 10,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 9,  # Nuclear: 9 years
            'capex': 0,  # already installed
            'investment_required': 0  # 0 means already installed (mandatory)
        })
    
    # 2. Optional solar generator at bus 1 
    for t in time_periods:
        # Direct use of solar value without seasonal normalization
        # Maximum installed capacity is 90 MW
        pmax_solar = 90 * data_mapping[t]['solar']
        
        gen_data.append({
            'id': 2, 'time': t, 'bus': 1, 'pmin': 0, 'pmax': pmax_solar, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 8,  # 8 years
            'capex': 1500000,  # $1.5M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 3. Optional wind generator at bus 2
    for t in time_periods:
        # Direct use of wind value without seasonal normalization
        # Apply 10% lower factor for this generator
        pmax_wind_3 = 80 * data_mapping[t]['wind'] * 0.9
        
        gen_data.append({
            'id': 3, 'time': t, 'bus': 2, 'pmin': 0, 'pmax': pmax_wind_3, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 7,  # 7 years
            'capex': 1200000,  # $1.2M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 4. Optional wind generator at bus 3 (different wind pattern)
    for t in time_periods:
        # Direct use of wind value without seasonal normalization
        # Apply 10% higher factor for this generator
        pmax_wind_4 = 50 * data_mapping[t]['wind'] * 1.1
        
        gen_data.append({
            'id': 4, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': pmax_wind_4, 'gencost': 0,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 6,  # 6 years
            'capex': 900000,  # $0.9M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 5. Optional solar generator at bus 3 (different solar pattern)
    for t in time_periods:
        # Direct use of solar value without seasonal normalization
        # Apply 5% lower factor for this generator
        pmax_solar_5 = 60 * data_mapping[t]['solar'] * 0.95
        
        gen_data.append({
            'id': 5, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': pmax_solar_5, 'gencost': 0,
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
            'id': 7, 'time': t, 'bus': 4, 'pmin': 0, 'pmax': 60, 'gencost': 0,
            'emax': 200,  # 200 MWh energy capacity
            'einitial': 50,  # Start 25% charged
            'eta': 0.9,  # 90% round-trip efficiency
            'lifetime': 3,  # 3 years
            'capex': 300000,  # $0.3M/MW (based on power capacity)
            'investment_required': 1  # 1 means investment required
        })
    
    # Create time-varying demand
    demand_data = []
    
    # Create power demand at buses 4 & 5
    # NOTE: 'b' x 'd' where 'b' is base demand and 'd' is the normalized demand data
    base_demand_bus4 = 4.0  # MW - Reduced from 50 MW to 4.0 MW
    base_demand_bus5 = 2.0  # MW - Reduced from 3 MW to 2.0 MW
    # Load(b,t) = base(b) * relative_load(t)
    for t in time_periods:
        demand_data.append({
            'time': t,
            'bus': 4,
            'pd': base_demand_bus4 * data_mapping[t]['load']
        })
        demand_data.append({
            'time': t,
            'bus': 5,
            'pd': base_demand_bus5 * data_mapping[t]['load']
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
    plt.title('Investment Decisions')
    plt.ylabel('Decision (1=Selected, 0=Not Selected)')
    plt.xlabel('Asset ID')
    
    plt.xticks(assets)
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'plots', 'investment_decisions.png'))
    plt.close()

def run_test():
    """Run the test for the investment DCOPF model."""
    print("Creating test system...")
    
    # Create 24 hours of data and synthetic data mapping
    start_time = datetime(2023, 1, 1)
    time_periods = [start_time + timedelta(hours=h) for h in range(24)]
    
    # Create synthetic data mapping
    data_mapping = {}
    for t in time_periods:
        hour = t.hour
        # Simple diurnal patterns
        solar_value = 0.0
        if 6 <= hour <= 18:  # Daylight hours
            solar_value = max(0, 0.8 * math.sin(math.pi * (hour - 6) / 12))
        
        # Wind tends to be stronger in evening/night
        wind_value = 0.4 + 0.2 * math.sin(math.pi * hour / 12)
        
        # Load peaks in morning and evening
        load_value = 0.6 + 0.2 * math.sin(math.pi * hour / 12) + 0.2 * math.sin(math.pi * hour / 6)
        
        data_mapping[t] = {
            'solar': solar_value,
            'wind': wind_value,
            'load': load_value,
            'season': 'winter'  # Just use winter for this test
        }
    
    gen_time_series, branch, bus, demand_time_series = create_test_system(
        time_periods=time_periods,
        data_mapping=data_mapping
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

if __name__ == "__main__":
    run_test()

'''
# This is example code showing how to use the Network class
# Commented out to prevent execution

# Create a Network instance
net = Network(name="MyPowerSystem")

# Define snapshots and weightings
snapshots = winter_hours + summer_hours + spring_autumn_hours
net.set_snapshots(snapshots)

weights = {}
for h in winter_hours:
    weights[h] = 13  # Winter represents 13 weeks
for h in summer_hours:
    weights[h] = 13  # Summer represents 13 weeks
for h in spring_autumn_hours:
    weights[h] = 26  # Spring/autumn represents 26 weeks
    
net.set_snapshot_weightings(weights)

# For operations only:
results = net.solve_dc()

# For investment decisions:
results = net.solve_dc(investment=True)

# For multi-period planning:
results = net.solve_dc(investment=True, multi_period=True)

# Print summary
net.summary()

# Access specific results
investment_decisions = net.results['investment_decisions']
generation = net.results['generation']
prices = net.results['marginal_prices']

# Create visualizations
plot_investment_decisions(net, results_dir)
plot_generation(net, results_dir)
''' 