#!/usr/bin/env python3

"""
test_investment.py

Test script for the multi-stage investment model using a chunk-based approach.
This script:
1. Creates a simple power system model for investment planning
2. Runs the multi-stage investment model with different asset lifetimes
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

# Import the modules
from scripts.dcopf_investment import dcopf_investment
from scripts.test_dcopf_mip import create_simple_power_system_direct

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

def run_investment_model():
    """
    Run the multi-stage investment model on the test system.
    """
    print("Creating investment test system...")
    
    # Create 24 hours of data for 4 typical days
    time_periods = []
    for season in range(4):  # Spring, Summer, Fall, Winter
        start_time = datetime(2023, 1 + season*3, 1)  # Start of each season
        time_periods.extend([start_time + timedelta(hours=h) for h in range(24)])
    
    gen_time_series, branch, bus, demand_time_series = create_investment_test_system(
        planning_horizon=10,
        time_periods=time_periods
    )
    
    # Define asset lifetimes
    asset_lifetimes = {
        1: 20,  # Baseload: 20 years
        2: 15,  # Mid-merit: 15 years
        3: 10,  # Peaker: 10 years
        4: 7,   # Renewable: 7 years
        5: 5    # Storage: 5 years
    }
    
    # Define asset capex
    asset_capex = {
        1: 1500000,  # Baseload: $1.5M/MW
        2: 1000000,  # Mid-merit: $1M/MW
        3: 500000,   # Peaker: $0.5M/MW
        4: 800000,   # Renewable: $0.8M/MW
        5: 400000    # Storage: $0.4M/MWh
    }
    
    print("\n===== Running Multi-Stage Investment Model =====")
    investment_results = dcopf_investment(
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
    print(inv_df[inv_df['decision'] == 1].sort_values(['asset_id', 'chunk_idx']))
    
    # Print cost summary
    print("\n===== Cost Summary =====")
    for cost_type, cost_value in investment_results['cost_summary'].items():
        print(f"{cost_type}: ${cost_value:,.2f}")
    
    # Calculate cost details by asset
    print("\n===== Cost Details by Asset =====")
    asset_costs = {}
    for (asset_id, chunk_idx), cost in investment_results['investment_costs'].items():
        if asset_id not in asset_costs:
            asset_costs[asset_id] = 0
        asset_costs[asset_id] += cost
    
    for asset_id, cost in asset_costs.items():
        print(f"Asset {asset_id}: ${cost:,.2f}")
    
    # Calculate generation by asset (sum across all chunks)
    print("\n===== Generation by Asset =====")
    total_gen_by_asset = {}
    
    for chunk_key, gen_df in investment_results['generation_by_chunk'].items():
        asset_id, chunk_idx = chunk_key
        for _, row in gen_df.iterrows():
            gen_id = row['id']
            if gen_id not in total_gen_by_asset:
                total_gen_by_asset[gen_id] = 0
            total_gen_by_asset[gen_id] += row['gen']
    
    for gen_id, total_gen in total_gen_by_asset.items():
        print(f"Generator {gen_id}: {total_gen:.2f} MWh")
    
    # Plot investment decisions
    plot_investment_decisions(investment_results)
    
    # Plot generation by chunk
    plot_generation_by_chunk(investment_results)
    
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
        1: 'navy',     # Baseload
        2: 'forestgreen',  # Mid-merit
        3: 'firebrick',    # Peaker
        4: 'gold',      # Renewable
        5: 'purple'    # Storage
    }
    
    # Create asset labels
    asset_labels = {
        1: 'Baseload',
        2: 'Mid-merit',
        3: 'Peaker',
        4: 'Renewable',
        5: 'Storage'
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
    plt.savefig(os.path.join(results_dir, 'investment_decisions.png'))
    plt.close()

def plot_generation_by_chunk(investment_results):
    """
    Plot the generation by chunk and asset.
    """
    # Get a sample chunk to plot
    sample_key = list(investment_results['generation_by_chunk'].keys())[0]
    sample_gen = investment_results['generation_by_chunk'][sample_key]
    
    if sample_gen.empty:
        print("No generation data to plot.")
        return
    
    # Asset labels
    asset_labels = {
        1: 'Baseload',
        2: 'Mid-merit',
        3: 'Peaker',
        4: 'Renewable',
        5: 'Storage'
    }
    
    # Create hourly generation plots for a specific asset and chunk
    asset_id, chunk_idx = sample_key
    chunk_gen = sample_gen.copy()
    
    # Convert times to datetime if they're not already
    if not isinstance(chunk_gen['time'].iloc[0], datetime):
        chunk_gen['time'] = pd.to_datetime(chunk_gen['time'])
    
    # Group by time and generator
    pivoted = chunk_gen.pivot_table(index='time', columns='id', values='gen', aggfunc='sum')
    
    plt.figure(figsize=(12, 6))
    
    # Create a stacked area plot
    pivoted.plot.area(ax=plt.gca(), stacked=True, alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Generation (MW)')
    plt.title(f'Hourly Generation by Generator Type for Asset {asset_id}, Chunk {chunk_idx}')
    
    # Use asset labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [asset_labels.get(int(label), f'Generator {label}') for label in labels]
    plt.legend(handles, new_labels)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'generation_asset{asset_id}_chunk{chunk_idx}.png'))
    plt.close()

if __name__ == "__main__":
    # Run the investment model
    investment_results = run_investment_model() 