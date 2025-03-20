#!/usr/bin/env python3

"""
test_investment.py

Test script for the power system investment model using a lifetime-based approach.
This script:
1. Creates a simple power system model for investment planning
2. Runs the investment model with different asset lifetimes
3. Visualizes the investment decisions and operational results
4. Analyzes the costs and benefits of different investment strategies
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from numpy import ones

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Add the scripts directory to the Python path
sys.path.insert(0, script_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(project_root, 'results', 'investment')
os.makedirs(results_dir, exist_ok=True)

# Import the CPLEX-based optimizer implementation
from scripts.optimizer import run_investment_model

# Define a simple function to create a power system for testing
def create_simple_power_system_direct(time_periods=None, line_limits=None):
    """
    Create a simple power system model for testing.
    
    Args:
        time_periods: List of time periods
        line_limits: Optional dictionary to override default line limits
        
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    if time_periods is None:
        time_periods = [datetime.now()]
    
    # Create buses
    bus_data = [
        {'bus_i': 1, 'type': 3, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},  # Slack bus
        {'bus_i': 2, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
        {'bus_i': 3, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
        {'bus_i': 4, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
        {'bus_i': 5, 'type': 1, 'pd': 0.0, 'qd': 0.0, 'gs': 0.0, 'bs': 0.0, 'vm': 1.0, 'va': 0.0},
    ]
    
    # Create branches
    if line_limits is None:
        line_limits = {
            (1, 2): 100.0,
            (1, 3): 100.0,
            (2, 4): 100.0,
            (3, 4): 100.0,
            (3, 5): 100.0,
            (4, 5): 100.0
        }
    
    branch_data = [
        {'fbus': 1, 'tbus': 2, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': line_limits.get((1, 2), 100.0), 'sus': 10.0},
        {'fbus': 1, 'tbus': 3, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': line_limits.get((1, 3), 100.0), 'sus': 10.0},
        {'fbus': 2, 'tbus': 4, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': line_limits.get((2, 4), 100.0), 'sus': 10.0},
        {'fbus': 3, 'tbus': 4, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': line_limits.get((3, 4), 100.0), 'sus': 10.0},
        {'fbus': 3, 'tbus': 5, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': line_limits.get((3, 5), 100.0), 'sus': 10.0},
        {'fbus': 4, 'tbus': 5, 'r': 0.01, 'x': 0.1, 'b': 0.0, 'ratea': line_limits.get((4, 5), 100.0), 'sus': 10.0},
    ]
    
    # Create generators
    gen_data = []
    for t in time_periods:
        # Baseload generator at bus 1
        gen_data.append({
            'id': 1, 'time': t, 'bus': 1, 'pmin': 20, 'pmax': 100, 'gencost': 20,
            'emax': 0, 'einitial': 0, 'eta': 0
        })
        
        # Mid-merit generator at bus 2
        gen_data.append({
            'id': 2, 'time': t, 'bus': 2, 'pmin': 10, 'pmax': 80, 'gencost': 40,
            'emax': 0, 'einitial': 0, 'eta': 0
        })
        
        # Peaker generator at bus 3
        gen_data.append({
            'id': 3, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': 50, 'gencost': 80,
            'emax': 0, 'einitial': 0, 'eta': 0
        })
    
    # Create demand
    demand_data = []
    for t in time_periods:
        # Demand at bus 4
        demand_data.append({
            'time': t, 'bus': 4, 'pd': 50
        })
        
        # Demand at bus 5
        demand_data.append({
            'time': t, 'bus': 5, 'pd': 100
        })
    
    # Convert to DataFrames
    gen_time_series = pd.DataFrame(gen_data)
    branch = pd.DataFrame(branch_data)
    bus = pd.DataFrame(bus_data)
    demand_time_series = pd.DataFrame(demand_data)
    
    return gen_time_series, branch, bus, demand_time_series

def create_investment_test_system(planning_horizon=10, time_periods=None, line_limits=None):
    """
    Extended version of the simple power system for investment testing.
    
    Args:
        planning_horizon: Planning horizon in years
        time_periods: Optional list of time periods to use (default: 24 hours)
        line_limits: Optional dictionary to override default line limits
        
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    # Create time periods if not provided
    if time_periods is None:
        # Create 24 hours of data
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_periods = [start_time + timedelta(hours=h) for h in range(24)]
    
    # Get the base system
    gen_time_series, branch, bus, demand_time_series = create_simple_power_system_direct(
        time_periods=time_periods, 
        line_limits=line_limits
    )
    
    # Add investment-related fields to generators
    gen_investment = []
    
    for _, row in gen_time_series.iterrows():
        gen_id = row['id']
        gen_bus = row['bus']
        gen_cost = row['gencost']
        time = row['time']
        
        # Add lifetime and capex based on generator type
        if gen_id == 1:  # Baseload
            lifetime = 20  # 20 years
            capex = 1500000  # $1.5M/MW
        elif gen_id == 2:  # Mid-merit
            lifetime = 15  # 15 years
            capex = 1000000  # $1M/MW
        elif gen_id == 3:  # Peaker
            lifetime = 10  # 10 years
            capex = 500000  # $0.5M/MW
        else:
            lifetime = 20  # Default
            capex = 1000000  # Default
        
        gen_investment.append({
            'id': gen_id,
            'time': time,
            'bus': gen_bus,
            'pmin': row['pmin'],
            'pmax': row['pmax'],
            'gencost': gen_cost,
            'emax': row['emax'],
            'einitial': row['einitial'],
            'eta': row['eta'],
            'lifetime': lifetime,
            'capex': capex
        })
    
    # Create a new generator for investment (renewable)
    for t in time_periods:
        gen_investment.append({
            'id': 4,  # New renewable generator
            'time': t,
            'bus': 5,  # At bus 5 where there's demand
            'pmin': 0,
            'pmax': 80,  # 80 MW capacity
            'gencost': 5,  # Very low marginal cost
            'emax': 0,  # Not storage
            'einitial': 0,
            'eta': 0,
            'lifetime': 7,  # 7 years lifetime
            'capex': 800000  # $0.8M/MW
        })
    
    # Create a new storage unit for investment
    for t in time_periods:
        gen_investment.append({
            'id': 5,  # Storage unit
            'time': t,
            'bus': 4,  # At bus 4 where there's demand
            'pmin': 0,
            'pmax': 50,  # 50 MW power capacity
            'gencost': 0,  # No fuel cost
            'emax': 200,  # 200 MWh energy capacity
            'einitial': 50,  # Start 25% charged
            'eta': 0.9,  # 90% round-trip efficiency
            'lifetime': 5,  # 5 years lifetime
            'capex': 400000  # $0.4M/MWh
        })
    
    # Create time-varying demand
    demand_pattern = []
    
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
        demand_pattern.append({
            'time': t,
            'bus': 4,
            'pd': base_demand_bus4 * factor
        })
        
        # Add demand at bus 5
        demand_pattern.append({
            'time': t,
            'bus': 5,
            'pd': base_demand_bus5 * factor
        })
    
    gen_time_series_inv = pd.DataFrame(gen_investment)
    demand_time_series_inv = pd.DataFrame(demand_pattern)
    
    return gen_time_series_inv, branch, bus, demand_time_series_inv

def create_investment_test_system_from_csv(planning_horizon=10, time_periods=None):
    """
    Create a power system for investment testing using grid configuration from CSV files.
    
    Includes:
    - A mandatory generator at bus 1 (always active)
    - Optional generators at bus 1, bus 2, and bus 3
    - Loads at bus 4 and bus 5
    - Storage units at bus 3 and bus 4
    
    Args:
        planning_horizon: Planning horizon in years
        time_periods: Optional list of time periods to use (default: 24 hours)
        
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    # Create time periods if not provided
    if time_periods is None:
        # Create 24 hours of data
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_periods = [start_time + timedelta(hours=h) for h in range(24)]
    
    # Load grid configuration from CSV files
    bus_csv_path = os.path.join(project_root, 'data', 'working', 'bus.csv')
    branch_csv_path = os.path.join(project_root, 'data', 'working', 'branch.csv')
    
    # Read bus and branch data
    bus = pd.read_csv(bus_csv_path)
    branch = pd.read_csv(branch_csv_path)
    
    # Create generators
    gen_data = []
    
    # 1. Mandatory generator at bus 1 (always active)
    for t in time_periods:
        gen_data.append({
            'id': 1, 'time': t, 'bus': 1, 'pmin': 20, 'pmax': 100, 'gencost': 20,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 20,  # 20 years
            'capex': 1500000,  # $1.5M/MW
            'investment_required': 0  # 0 means already installed (mandatory)
        })
    
    # 2. Optional generator at bus 1 (can be invested in)
    for t in time_periods:
        gen_data.append({
            'id': 2, 'time': t, 'bus': 1, 'pmin': 0, 'pmax': 120, 'gencost': 30,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 15,  # 15 years
            'capex': 1200000,  # $1.2M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 3. Optional generator at bus 2 (mid-merit)
    for t in time_periods:
        gen_data.append({
            'id': 3, 'time': t, 'bus': 2, 'pmin': 0, 'pmax': 80, 'gencost': 40,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 15,  # 15 years
            'capex': 1000000,  # $1M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 4. Optional generator at bus 3 (peaker)
    for t in time_periods:
        gen_data.append({
            'id': 4, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': 50, 'gencost': 70,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 10,  # 10 years
            'capex': 600000,  # $0.6M/MW
            'investment_required': 1  # 1 means investment required
        })
    
    # 5. Optional renewable generator at bus 3
    for t in time_periods:
        gen_data.append({
            'id': 5, 'time': t, 'bus': 3, 'pmin': 0, 'pmax': 60, 'gencost': 5,
            'emax': 0, 'einitial': 0, 'eta': 0,
            'lifetime': 7,  # 7 years
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
            'lifetime': 5,  # 5 years
            'capex': 400000,  # $0.4M/MWh
            'investment_required': 1  # 1 means investment required
        })
    
    # 7. Storage unit at bus 4
    for t in time_periods:
        gen_data.append({
            'id': 7, 'time': t, 'bus': 4, 'pmin': 0, 'pmax': 50, 'gencost': 0,
            'emax': 200,  # 200 MWh energy capacity
            'einitial': 50,  # Start 25% charged
            'eta': 0.9,  # 90% round-trip efficiency
            'lifetime': 5,  # 5 years
            'capex': 400000,  # $0.4M/MWh
            'investment_required': 1  # 1 means investment required
        })
    
    # Create time-varying demand
    demand_data = []
    
    # Base demand at each bus (from bus.csv)
    base_demand_bus4 = float(bus[bus['bus_i'] == 4]['Pd'].values[0])  # From CSV
    base_demand_bus5 = float(bus[bus['bus_i'] == 5]['Pd'].values[0])  # From CSV
    
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
    demand_time_series = pd.DataFrame(demand_data)
    
    # Return the power system data
    return gen_time_series, branch, bus, demand_time_series

def run_test_model_from_csv():
    """
    Run the investment model on the test system with CSV data.
    """
    print("Creating investment test system from CSV files...")
    
    # Create 24 hours of data for 4 typical days
    time_periods = []
    for season in range(4):  # Spring, Summer, Fall, Winter
        start_time = datetime(2023, 1 + season*3, 1)  # Start of each season
        time_periods.extend([start_time + timedelta(hours=h) for h in range(24)])
    
    gen_time_series, branch, bus, demand_time_series = create_investment_test_system_from_csv(
        planning_horizon=10,
        time_periods=time_periods
    )
    
    # Define asset lifetimes
    asset_lifetimes = {
        1: 20,  # Mandatory generator at bus 1: 20 years
        2: 15,  # Optional generator at bus 1: 15 years
        3: 15,  # Mid-merit at bus 2: 15 years
        4: 10,  # Peaker at bus 3: 10 years
        5: 7,   # Renewable at bus 3: 7 years
        6: 5,   # Storage at bus 3: 5 years
        7: 5    # Storage at bus 4: 5 years
    }
    
    # Define asset capex
    asset_capex = {
        1: 1500000,  # Mandatory generator: $1.5M/MW
        2: 1200000,  # Optional generator at bus 1: $1.2M/MW
        3: 1000000,  # Mid-merit: $1M/MW
        4: 600000,   # Peaker: $0.6M/MW
        5: 800000,   # Renewable: $0.8M/MW
        6: 400000,   # Storage at bus 3: $0.4M/MWh
        7: 400000    # Storage at bus 4: $0.4M/MWh
    }
    
    print("\n===== Running Investment Model =====")
    investment_results = run_investment_model(
        gen_time_series, branch, bus, demand_time_series,
        planning_horizon=10,
        start_year=2023,
        asset_lifetimes=asset_lifetimes,
        asset_capex=asset_capex,
        operational_periods_per_year=4,  # 4 typical seasons
        hours_per_period=24  # 24 hours per season
    )
    
    if investment_results is None:
        print("Investment optimization failed.")
        return
    
    # Print investment decisions
    print("\n===== Investment Decisions =====")
    inv_df = investment_results['investment']
    print(inv_df[inv_df['decision'] == 1].sort_values(['asset_id', 'lifetime_period']))
    
    # Print cost summary
    print("\n===== Cost Summary =====")
    for cost_type, cost_value in investment_results['cost_summary'].items():
        print(f"{cost_type}: ${cost_value:,.2f}")
    
    # Calculate cost details by asset
    print("\n===== Cost Details by Asset =====")
    asset_costs = {}
    for (asset_id, period_idx), cost in investment_results['investment_costs'].items():
        if asset_id not in asset_costs:
            asset_costs[asset_id] = 0
        asset_costs[asset_id] += cost
    
    for asset_id, cost in asset_costs.items():
        print(f"Asset {asset_id}: ${cost:,.2f}")
    
    # Calculate generation by asset (sum across all lifetime periods)
    print("\n===== Generation by Asset =====")
    total_gen_by_asset = {}
    
    for period_key, gen_df in investment_results['generation_by_period'].items():
        asset_id, period_idx = period_key
        for _, row in gen_df.iterrows():
            gen_id = row['id']
            if gen_id not in total_gen_by_asset:
                total_gen_by_asset[gen_id] = 0
            total_gen_by_asset[gen_id] += row['gen']
    
    for gen_id, total_gen in total_gen_by_asset.items():
        print(f"Generator {gen_id}: {total_gen:.2f} MWh")
    
    # Plot investment decisions
    plot_investment_decisions(investment_results)
    
    # Plot generation by lifetime period
    plot_generation_by_period(investment_results)
    
    return investment_results

def plot_investment_decisions(investment_results):
    """
    Plot the investment decisions by asset and year.
    """
    inv_df = investment_results['investment']
    inv_df_pos = inv_df[inv_df['decision'] == 1].copy()
    
    if inv_df_pos.empty:
        print("No positive investment decisions to plot.")
        return
    
    # Create a colormap for assets
    asset_colors = {
        1: 'navy',     # Mandatory generator at bus 1
        2: 'forestgreen',  # Optional generator at bus 1
        3: 'darkgreen',  # Mid-merit at bus 2
        4: 'firebrick',  # Peaker at bus 3
        5: 'gold',     # Renewable at bus 3
        6: 'purple',   # Storage at bus 3
        7: 'orchid'    # Storage at bus 4
    }
    
    # Create asset labels
    asset_labels = {
        1: 'Mandatory Gen (Bus 1)',
        2: 'Optional Gen (Bus 1)',
        3: 'Mid-merit (Bus 2)',
        4: 'Peaker (Bus 3)',
        5: 'Renewable (Bus 3)',
        6: 'Storage (Bus 3)',
        7: 'Storage (Bus 4)'
    }
    
    plt.figure(figsize=(12, 6))
    
    # For each asset, create a horizontal line showing when it's active
    for _, row in inv_df_pos.iterrows():
        asset_id = row['asset_id']
        start_year = row['start_year']
        end_year = row['end_year']
        
        plt.plot([start_year, end_year], [asset_id, asset_id], linewidth=10, 
                 color=asset_colors.get(asset_id, 'gray'), 
                 label=asset_labels.get(asset_id, f'Asset {asset_id}'))
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.yticks(list(asset_labels.keys()), list(asset_labels.values()))
    plt.xlabel('Year')
    plt.ylabel('Asset')
    plt.title('Investment Decisions by Asset and Year')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits
    min_year = inv_df_pos['start_year'].min()
    max_year = inv_df_pos['end_year'].max()
    plt.xlim(min_year - 0.5, max_year + 0.5)
    
    plt.tight_layout()
    
    # Save the plot and print confirmation message
    output_path = os.path.join(results_dir, 'investment_decisions.png')
    plt.savefig(output_path)
    print(f"Investment decisions plot saved to: {output_path}")
    plt.close()

def plot_generation_by_period(results):
    """Plot the generation for each period and asset."""
    # Create asset labels
    asset_labels = {
        1: 'Baseload',
        2: 'Mid-merit',
        3: 'Peaker',
        4: 'Renewable',
        5: 'Storage'
    }
    
    # Extract generation data from the results
    # Check if we have generation_by_period data
    if 'generation_by_period' not in results or not results['generation_by_period']:
        print("No generation data to plot.")
        return
    
    # Combine generation data from all periods into one DataFrame
    all_gen_data = []
    for period_key, period_df in results['generation_by_period'].items():
        all_gen_data.append(period_df)
    
    if not all_gen_data:
        print("No generation data to plot.")
        return
        
    period_gen = pd.concat(all_gen_data, ignore_index=True)
    
    if isinstance(period_gen, pd.DataFrame):
        # Handle the case where period_gen is already a DataFrame
        pass
    else:
        period_gen = pd.DataFrame(period_gen)
        period_gen['time'] = pd.to_datetime(period_gen['time'])
    
    # Group by time and generator
    pivoted = period_gen.pivot_table(index='time', columns='id', values='gen', aggfunc='sum')
    
    plt.figure(figsize=(12, 6))
    
    # Separate positive and negative values for plotting
    pos_data = pivoted.copy()
    neg_data = pivoted.copy()
    
    # Set negative values to 0 in pos_data
    pos_data[pos_data < 0] = 0
    
    # Set positive values to 0 in neg_data
    neg_data[neg_data > 0] = 0
    
    # Plot positive values as stacked area
    ax = plt.gca()
    if not pos_data.empty and (pos_data > 0).any().any():
        pos_data.plot.area(ax=ax, stacked=True, alpha=0.7)
    
    # Plot negative values as stacked area, but on the negative side
    if not neg_data.empty and (neg_data < 0).any().any():
        neg_data.plot.area(ax=ax, stacked=True, alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Generation/Consumption (MW)')
    plt.title('Hourly Generation and Storage Operation')
    
    # Use asset labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        asset_id = int(label)
        if asset_id in asset_labels:
            new_labels.append(asset_labels[asset_id])
        else:
            new_labels.append(f'Generator {label}')
    
    if handles and new_labels:
        plt.legend(handles, new_labels)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'generation_by_period.png'))
    print("Generation plot saved to:", os.path.join(results_dir, 'generation_by_period.png'))
    plt.close()

if __name__ == "__main__":
    # Run the modified investment model using CSV data
    investment_results = run_test_model_from_csv() 