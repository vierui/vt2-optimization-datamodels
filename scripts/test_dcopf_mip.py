#!/usr/bin/env python3

"""
test_dcopf_mip.py

Test script for the Simplified Mixed Integer Programming DC Optimal Power Flow (DCOPF-MIP) implementation.
This script:
1. Creates a simple power system model directly
2. Runs the simplified DCOPF-MIP solver with binary commitment variables
3. Compares the results with the standard DCOPF solver
4. Displays the unit commitment decisions and startup/shutdown costs
5. Simulates normal, congested, and highly congested scenarios and compares the results

The simplified MIP implementation only includes binary commitment variables and 
startup/shutdown variables, without linking constraints or temporal operational constraints.
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
results_dir = os.path.join(project_root, 'results', 'test-mip')
os.makedirs(results_dir, exist_ok=True)

# Import the modules
from scripts.dcopf import dcopf
from scripts.dcopf_mip import dcopf_mip

def create_simple_power_system_direct(time_periods=None, line_limits=None):
    """
    Simple 5-bus power system model created directly from the repository.
    
    Args:
        time_periods: Optional list of time periods to use (default: current time)
        line_limits: Optional dictionary to override default line limits {(from_bus, to_bus): limit}
    
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    # Create time periods if not provided
    if time_periods is None:
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_periods = [current_time]
    
    # Read bus data directly from the repository
    bus_file = os.path.join(project_root, 'bus.csv')
    if not os.path.exists(bus_file):
        # Try alternative location
        bus_file = os.path.join(project_root, 'data', 'working', 'bus.csv')
    
    if not os.path.exists(bus_file):
        raise FileNotFoundError(f"Could not find bus.csv in {project_root} or {os.path.join(project_root, 'data', 'working')}")
    
    bus = pd.read_csv(bus_file)
    
    # Ensure numeric types for critical columns
    bus['bus_i'] = bus['bus_i'].astype(int)
    bus['type'] = bus['type'].astype(int)
    bus['Pd'] = bus['Pd'].astype(float)
    bus['Qd'] = bus['Qd'].astype(float)
    
    # Read branch data directly from the repository
    branch_file = os.path.join(project_root, 'branch.csv')
    if not os.path.exists(branch_file):
        # Try alternative location
        branch_file = os.path.join(project_root, 'data', 'working', 'branch.csv')
    
    if not os.path.exists(branch_file):
        raise FileNotFoundError(f"Could not find branch.csv in {project_root} or {os.path.join(project_root, 'data', 'working')}")
    
    branch = pd.read_csv(branch_file)
    
    # Ensure numeric types for critical columns
    branch['fbus'] = branch['fbus'].astype(int)
    branch['tbus'] = branch['tbus'].astype(int)
    branch['r'] = branch['r'].astype(float)
    branch['x'] = branch['x'].astype(float)
    branch['b'] = branch['b'].astype(float)
    branch['ratea'] = branch['ratea'].astype(float)
    branch['status'] = branch['status'].astype(int)
    
    # Override line limits if provided
    if line_limits:
        for (from_bus, to_bus), limit in line_limits.items():
            mask = (branch['fbus'] == from_bus) & (branch['tbus'] == to_bus)
            if mask.any():
                branch.loc[mask, 'ratea'] = limit
    
    # Calculate susceptance (1/x) for DC power flow
    branch['sus'] = 1.0 / branch['x']
    
    # Create generators
    # For each time period, create generator data
    gen_data = []
    
    # Generator at bus 1 (reference bus) - cheap baseload
    for t in time_periods:
        gen_data.append({
            'id': 1,
            'time': t,
            'bus': 1,
            'pmin': 0,
            'pmax': 200,
            'gencost': 10,  # $10/MWh - cheap baseload
            'emax': 0,      # Not storage
            'einitial': 0,
            'eta': 0
        })
    
    # Generator at bus 2 - medium cost
    for t in time_periods:
        gen_data.append({
            'id': 2,
            'time': t,
            'bus': 2,
            'pmin': 0,
            'pmax': 100,
            'gencost': 30,  # $30/MWh - medium cost
            'emax': 0,      # Not storage
            'einitial': 0,
            'eta': 0
        })
    
    # Generator at bus 3 - expensive peaker
    for t in time_periods:
        gen_data.append({
            'id': 3,
            'time': t,
            'bus': 3,
            'pmin': 0,
            'pmax': 100,
            'gencost': 50,  # $50/MWh - expensive peaker
            'emax': 0,      # Not storage
            'einitial': 0,
            'eta': 0
        })
    
    gen_time_series = pd.DataFrame(gen_data)
    
    # Create demand
    demand_data = []
    
    # Demand at bus 4
    for t in time_periods:
        demand_data.append({
            'time': t,
            'bus': 4,
            'pd': 50  # 50 MW demand
        })
    
    # Demand at bus 5
    for t in time_periods:
        demand_data.append({
            'time': t,
            'bus': 5,
            'pd': 100  # 100 MW demand
        })
    
    demand_time_series = pd.DataFrame(demand_data)
    
    return gen_time_series, branch, bus, demand_time_series

def run_and_compare():
    """
    Run both DCOPF and DCOPF-MIP on the same data and compare results.
    """
    print("Creating simple power system model...")
    
    try:
        gen_time_series, branch, bus, demand_time_series = create_simple_power_system_direct()
        
        # Run standard DCOPF
        print("\n===== Running standard DCOPF (LP) =====")
        lp_results = dcopf(gen_time_series, branch, bus, demand_time_series)
        
        if lp_results is None:
            print("DCOPF optimization failed.")
            return
        
        # Run DCOPF-MIP with default parameters
        print("\n===== Running simplified DCOPF-MIP with binary commitment variables =====")
        mip_results = dcopf_mip(gen_time_series, branch, bus, demand_time_series)
        
        if mip_results is None:
            print("DCOPF-MIP optimization failed.")
            return
        
        # Compare objective values
        print("\n===== Cost Comparison =====")
        print(f"LP Total Cost: ${lp_results['cost']:.2f}")
        print(f"MIP Total Cost: ${mip_results['cost']:.2f}")
        print(f"Cost Difference: ${mip_results['cost'] - lp_results['cost']:.2f}")
        
        # Compare generation dispatch
        print("\n===== Generation Comparison =====")
        lp_gen = lp_results['generation']
        mip_gen = mip_results['generation']
        
        # Merge the two dataframes for comparison
        gen_compare = pd.merge(
            lp_gen[['time', 'id', 'gen']].rename(columns={'gen': 'lp_gen'}),
            mip_gen[['time', 'id', 'gen']].rename(columns={'gen': 'mip_gen'}),
            on=['time', 'id']
        )
        gen_compare['diff'] = gen_compare['mip_gen'] - gen_compare['lp_gen']
        
        print(gen_compare)
        
        # Show commitment decisions from MIP
        print("\n===== Commitment Decisions =====")
        print(mip_results['commitment'])
        
        # Show startup/shutdown decisions from MIP
        print("\n===== Startup/Shutdown Decisions =====")
        print(mip_results['startup_shutdown'])
        
        # Compare LMPs
        print("\n===== LMP Comparison =====")
        lp_lmp = lp_results.get('marginal_prices')
        mip_lmp = mip_results.get('marginal_prices')
        
        # Check if both LP and MIP have marginal prices before comparing
        if lp_lmp is not None and not lp_lmp.empty and mip_lmp is not None and not mip_lmp.empty:
            try:
                lmp_compare = pd.merge(
                    lp_lmp[['time', 'bus', 'price']].rename(columns={'price': 'lp_price'}),
                    mip_lmp[['time', 'bus', 'price']].rename(columns={'price': 'mip_price'}),
                    on=['time', 'bus']
                )
                lmp_compare['diff'] = lmp_compare['mip_price'] - lmp_compare['lp_price']
                print(lmp_compare)
            except Exception as e:
                print(f"Error comparing LMPs: {e}")
        else:
            print("Marginal prices not available for one or both models.")
            if lp_lmp is not None and not lp_lmp.empty:
                print("LP LMPs:")
                print(lp_lmp)
            if mip_lmp is not None and not mip_lmp.empty:
                print("MIP LMPs:")
                print(mip_lmp)
            else:
                print("MIP LMPs are not available because mixed-integer programs don't provide dual values directly.")
        
        # Save the results to a CSV file
        results_file = os.path.join(results_dir, 'lp_vs_mip_comparison.csv')
        gen_compare.to_csv(results_file, index=False)
        print(f"\nComparison results saved to {results_file}")
        
        return lp_results, mip_results
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def run_with_varying_demand():
    """
    Run DCOPF-MIP with varying demand to demonstrate startup/shutdown decisions.
    """
    print("Creating simple power system model with multiple time periods...")
    
    try:
        # Create multiple time periods
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_periods = [current_time + timedelta(hours=i) for i in range(10)]
        
        # Create power system model with the time periods
        gen_time_series, branch, bus, _ = create_simple_power_system_direct(time_periods)
        
        # Define startup and shutdown costs
        startup_costs = {1: 500, 2: 200, 3: 100}  # More expensive for baseload units
        shutdown_costs = {1: 300, 2: 100, 3: 50}  # More expensive for baseload units
        
        # Create time-varying demand
        demand_data = []
        
        # Demand at bus 4 - different for each hour
        for i, t in enumerate(time_periods):
            # Create a demand profile that varies over time
            # First low, then high, then low again
            if i < 3:
                load = 50
            elif i < 7:
                load = 150 + (i-3) * 20  # Increasing load
            else:
                load = 130 - (i-7) * 20  # Decreasing load
                
            demand_data.append({
                'time': t,
                'bus': 4,
                'pd': load
            })
        
        # Demand at bus 5 - constant
        for t in time_periods:
            demand_data.append({
                'time': t,
                'bus': 5,
                'pd': 50  # Lower constant demand
            })
        
        demand_time_series = pd.DataFrame(demand_data)
        
        # Run LP DCOPF for comparison
        print("\n===== Running standard DCOPF (LP) with varying demand =====")
        lp_results = dcopf(gen_time_series, branch, bus, demand_time_series)
        
        if lp_results is None:
            print("LP DCOPF optimization failed.")
            return None
        
        # Run DCOPF-MIP with varying demand
        print("\n===== Running DCOPF-MIP with varying demand =====")
        mip_results = dcopf_mip(
            gen_time_series, 
            branch, 
            bus, 
            demand_time_series,
            startup_costs=startup_costs,
            shutdown_costs=shutdown_costs
        )
        
        if mip_results is None:
            print("DCOPF-MIP optimization failed.")
            return None
        
        # Print commitment results
        print("\n===== Commitment Schedule =====")
        commitment_df = mip_results['commitment']
        # Pivot to show generators as columns and time as rows
        pivot_commitment = commitment_df.pivot(index='time', columns='id', values='commitment')
        print(pivot_commitment)
        
        # Print startup/shutdown results
        print("\n===== Startup/Shutdown Schedule =====")
        startup_shutdown_df = mip_results['startup_shutdown']
        # Create separate dataframes for startup and shutdown
        startup_df = startup_shutdown_df[['time', 'id', 'startup']].pivot(index='time', columns='id', values='startup')
        shutdown_df = startup_shutdown_df[['time', 'id', 'shutdown']].pivot(index='time', columns='id', values='shutdown')
        
        print("Startups:")
        print(startup_df)
        print("\nShutdowns:")
        print(shutdown_df)
        
        # Print generation
        print("\n===== Generation Schedule =====")
        generation_df = mip_results['generation']
        pivot_gen = generation_df.pivot(index='time', columns='id', values='gen')
        print(pivot_gen)
        
        # Print demand
        print("\n===== Demand Schedule =====")
        demand_pivot = demand_time_series.pivot(index='time', columns='bus', values='pd')
        print(demand_pivot)
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Generation and commitment
        plt.subplot(2, 1, 1)
        pivot_gen.plot(kind='bar', stacked=True, ax=plt.gca())
        
        # Add commitment markers
        markers = ['o', 's', '^']
        for i, gen_id in enumerate(pivot_commitment.columns):
            # Only show markers where commitment = 1
            for t_idx, t in enumerate(pivot_commitment.index):
                if pivot_commitment.loc[t, gen_id] == 1:
                    plt.plot(t_idx, pivot_gen.iloc[t_idx].sum() + 10, markers[i % len(markers)], 
                            markersize=8, label=f'Gen {gen_id} Committed' if t_idx == 0 else "")
        
        plt.title('Generation Schedule and Commitment')
        plt.ylabel('Generation (MW)')
        plt.xlabel('Time Period')
        plt.legend()
        
        # Plot 2: Demand
        plt.subplot(2, 1, 2)
        total_demand = demand_pivot.sum(axis=1)
        total_demand.plot(marker='o', ax=plt.gca())
        
        plt.title('Total System Demand')
        plt.ylabel('Demand (MW)')
        plt.xlabel('Time Period')
        plt.grid(True)
        
        plt.tight_layout()
        # Save the figure to the results directory
        plt.savefig(os.path.join(results_dir, 'dcopf_mip_results.png'))
        plt.close()
        
        # Save results to CSV
        commitment_df.to_csv(os.path.join(results_dir, 'commitment.csv'), index=False)
        startup_shutdown_df.to_csv(os.path.join(results_dir, 'startup_shutdown.csv'), index=False)
        generation_df.to_csv(os.path.join(results_dir, 'generation.csv'), index=False)
        
        # Create additional plots
        plot_generation_cost_comparison(lp_results, mip_results, gen_time_series)
        plot_marginal_price_comparison(lp_results, mip_results)
        
        return mip_results, lp_results
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def plot_generation_cost_comparison(lp_results, mip_results, gen_time_series):
    """
    Plot the generation cost per solver per hour.
    
    Args:
        lp_results: Results from LP solver
        mip_results: Results from MIP solver
        gen_time_series: Generator data including costs
    """
    print("\n===== Creating Generation Cost Comparison Plot =====")
    
    # Extract generation data
    lp_gen = lp_results['generation']
    mip_gen = mip_results['generation']
    
    # Create cost dataframes
    lp_costs = []
    mip_costs = []
    
    # Group by time
    for time, group in lp_gen.groupby('time'):
        # Calculate total cost for this time period
        total_cost = 0
        for _, row in group.iterrows():
            # Get generator cost
            gen_cost = gen_time_series[
                (gen_time_series['id'] == row['id']) & 
                (gen_time_series['time'] == time)
            ]['gencost'].iloc[0]
            
            # Add to total
            total_cost += row['gen'] * gen_cost
        
        lp_costs.append({
            'time': time,
            'cost': total_cost
        })
    
    # Group by time for MIP
    for time, group in mip_gen.groupby('time'):
        # Calculate total cost for this time period
        total_cost = 0
        for _, row in group.iterrows():
            # Get generator cost
            gen_cost = gen_time_series[
                (gen_time_series['id'] == row['id']) & 
                (gen_time_series['time'] == time)
            ]['gencost'].iloc[0]
            
            # Add to total
            total_cost += row['gen'] * gen_cost
        
        mip_costs.append({
            'time': time,
            'cost': total_cost
        })
    
    # Also add the startup/shutdown costs to MIP
    if 'startup_shutdown' in mip_results:
        startup_shutdown_df = mip_results['startup_shutdown']
        startup_costs_added = False
        
        for time, group in startup_shutdown_df.groupby('time'):
            # Find the matching time in mip_costs
            for i, cost_entry in enumerate(mip_costs):
                if cost_entry['time'] == time:
                    # Get the startup and shutdown costs for this time period
                    for _, row in group.iterrows():
                        if row['startup'] == 1:
                            gen_id = row['id']
                            # Find the generator cost - assuming startup is 10% of gencost if not otherwise specified
                            gen_cost = gen_time_series[
                                (gen_time_series['id'] == gen_id) & 
                                (gen_time_series['time'] == time)
                            ]['gencost'].iloc[0]
                            startup_cost = 0.1 * gen_cost
                            mip_costs[i]['cost'] += startup_cost
                            startup_costs_added = True
                            
                        if row['shutdown'] == 1:
                            gen_id = row['id']
                            # Find the generator cost - assuming shutdown is 5% of gencost if not otherwise specified
                            gen_cost = gen_time_series[
                                (gen_time_series['id'] == gen_id) & 
                                (gen_time_series['time'] == time)
                            ]['gencost'].iloc[0]
                            shutdown_cost = 0.05 * gen_cost
                            mip_costs[i]['cost'] += shutdown_cost
                            startup_costs_added = True
        
        if startup_costs_added:
            print("Added startup/shutdown costs to MIP total cost calculation")
    
    # Convert to dataframes
    lp_costs_df = pd.DataFrame(lp_costs)
    mip_costs_df = pd.DataFrame(mip_costs)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot both cost curves
    plt.plot(lp_costs_df['time'], lp_costs_df['cost'], 'o-', label='LP Solver')
    plt.plot(mip_costs_df['time'], mip_costs_df['cost'], 's-', label='MIP Solver')
    
    # Format the x-axis to show hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.title('Generation Cost Comparison Between LP and MIP Solvers')
    plt.xlabel('Hour')
    plt.ylabel('Cost ($)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(results_dir, 'generation_cost_comparison.png'))
    plt.close()
    
    # Save data to CSV
    comparison_df = pd.merge(
        lp_costs_df.rename(columns={'cost': 'lp_cost'}),
        mip_costs_df.rename(columns={'cost': 'mip_cost'}),
        on='time'
    )
    comparison_df['cost_diff'] = comparison_df['mip_cost'] - comparison_df['lp_cost']
    comparison_df.to_csv(os.path.join(results_dir, 'generation_cost_comparison.csv'), index=False)
    
    print(f"Generation cost comparison saved to {os.path.join(results_dir, 'generation_cost_comparison.png')}")
    print(f"Generation cost data saved to {os.path.join(results_dir, 'generation_cost_comparison.csv')}")

def plot_marginal_price_comparison(lp_results, mip_results):
    """
    Plot the marginal prices per solver per bus per hour.
    
    Args:
        lp_results: Results from LP solver
        mip_results: Results from MIP solver
    """
    print("\n===== Creating Marginal Price Comparison Plot =====")
    
    # Extract marginal prices
    lp_lmp = lp_results.get('marginal_prices')
    mip_lmp = mip_results.get('marginal_prices')
    
    # Check if marginal prices are available
    if lp_lmp is None or lp_lmp.empty:
        print("LP marginal prices not available. Skipping marginal price comparison.")
        return
    
    # Create plot - we'll focus on LP prices since MIP duals aren't typically available
    plt.figure(figsize=(15, 8))
    
    # Get unique buses and times
    buses = sorted(lp_lmp['bus'].unique())
    times = sorted(lp_lmp['time'].unique())
    
    # Plot LP prices for each bus over time
    for bus in buses:
        bus_prices = lp_lmp[lp_lmp['bus'] == bus]
        plt.plot(bus_prices['time'], bus_prices['price'], 'o-', label=f'Bus {bus}')
    
    # Format the x-axis to show hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.title('Marginal Prices (LMP) per Bus over Time')
    plt.xlabel('Hour')
    plt.ylabel('Marginal Price ($/MWh)')
    plt.grid(True)
    plt.legend(title='Bus')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(results_dir, 'marginal_price_comparison.png'))
    plt.close()
    
    # Create a heatmap view for better visualization with multiple buses and times
    if len(times) > 1:
        # Pivot the data to create a heatmap (times as rows, buses as columns)
        lp_heatmap_data = lp_lmp.pivot(index='time', columns='bus', values='price')
        
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(lp_heatmap_data.columns, range(len(lp_heatmap_data)), lp_heatmap_data.values, 
                      cmap='viridis', shading='auto')
        
        plt.colorbar(label='Marginal Price ($/MWh)')
        plt.title('Marginal Prices (LMP) Heatmap')
        plt.xlabel('Bus')
        plt.ylabel('Time Period')
        
        # Use actual time labels on the y-axis
        time_labels = [t.strftime('%H:%M') for t in lp_heatmap_data.index]
        plt.yticks(range(len(time_labels)), time_labels)
        
        plt.tight_layout()
        # Save the heatmap
        plt.savefig(os.path.join(results_dir, 'marginal_price_heatmap.png'))
        plt.close()
        
        print(f"Marginal price heatmap saved to {os.path.join(results_dir, 'marginal_price_heatmap.png')}")
    
    # Save data to CSV
    lp_lmp.to_csv(os.path.join(results_dir, 'marginal_prices.csv'), index=False)
    
    print(f"Marginal price comparison saved to {os.path.join(results_dir, 'marginal_price_comparison.png')}")
    print(f"Marginal price data saved to {os.path.join(results_dir, 'marginal_prices.csv')}")

def run_congestion_analysis():
    """
    Run DCOPF and DCOPF-MIP solvers with different congestion scenarios:
    1. Normal (no congestion)
    2. Congested (reduced limit on one line)
    3. Highly congested (reduced limits on multiple lines but still feasible)
    
    Compare the results across scenarios.
    """
    print("\n===== Congestion Analysis =====")
    print("Comparing normal, congested, and highly congested scenarios")
    
    try:
        # Create multiple time periods
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_periods = [current_time + timedelta(hours=i) for i in range(10)]
        
        # Define startup and shutdown costs
        startup_costs = {1: 500, 2: 200, 3: 100}  # More expensive for baseload units
        shutdown_costs = {1: 300, 2: 100, 3: 50}  # More expensive for baseload units
        
        # 1. Normal scenario (no congestion)
        print("\n--- Normal Scenario (No Congestion) ---")
        gen_time_series, branch, bus, _ = create_simple_power_system_direct(time_periods)
        
        # Create time-varying demand
        demand_data = []
        
        # Demand at bus 4 - different for each hour
        for i, t in enumerate(time_periods):
            # Create a demand profile that varies over time
            # First low, then high, then low again
            if i < 3:
                load = 50
            elif i < 7:
                load = 150 + (i-3) * 20  # Increasing load
            else:
                load = 130 - (i-7) * 20  # Decreasing load
                
            demand_data.append({
                'time': t,
                'bus': 4,
                'pd': load
            })
        
        # Demand at bus 5 - constant
        for t in time_periods:
            demand_data.append({
                'time': t,
                'bus': 5,
                'pd': 50  # Lower constant demand
            })
        
        demand_time_series = pd.DataFrame(demand_data)
        
        # Run LP DCOPF for normal scenario
        print("Running standard DCOPF (LP) for normal scenario...")
        lp_normal_results = dcopf(gen_time_series, branch, bus, demand_time_series)
        
        if lp_normal_results is None:
            print("LP DCOPF optimization failed for normal scenario.")
            lp_normal_results = None
        
        # Run DCOPF-MIP for normal scenario
        print("Running DCOPF-MIP for normal scenario...")
        mip_normal_results = dcopf_mip(
            gen_time_series, 
            branch, 
            bus, 
            demand_time_series,
            startup_costs=startup_costs,
            shutdown_costs=shutdown_costs
        )
        
        if mip_normal_results is None:
            print("DCOPF-MIP optimization failed for normal scenario.")
            mip_normal_results = None
        
        # 2. Congested scenario (reduce limit on line 1->2)
        print("\n--- Congested Scenario ---")
        gen_time_series_congested, branch_congested, bus_congested, _ = create_simple_power_system_direct(
            time_periods, 
            line_limits={(1, 2): 50}  # Reduce the limit on line 1->2 to 50 MW
        )
        
        # Run LP DCOPF for congested scenario
        print("Running standard DCOPF (LP) for congested scenario...")
        lp_congested_results = dcopf(gen_time_series_congested, branch_congested, bus_congested, demand_time_series)
        
        if lp_congested_results is None:
            print("LP DCOPF optimization failed for congested scenario.")
            lp_congested_results = None
        
        # Run DCOPF-MIP for congested scenario
        print("Running DCOPF-MIP for congested scenario...")
        mip_congested_results = dcopf_mip(
            gen_time_series_congested, 
            branch_congested, 
            bus_congested, 
            demand_time_series,
            startup_costs=startup_costs,
            shutdown_costs=shutdown_costs
        )
        
        if mip_congested_results is None:
            print("DCOPF-MIP optimization failed for congested scenario.")
            mip_congested_results = None
        
        # 3. Highly congested scenario (reduce limits on multiple lines but keep problem feasible)
        print("\n--- Highly Congested Scenario ---")
        gen_time_series_highly, branch_highly, bus_highly, _ = create_simple_power_system_direct(
            time_periods, 
            line_limits={(1, 2): 40, (2, 5): 50}  # Less restrictive limits to keep problem feasible
        )
        
        # Run LP DCOPF for highly congested scenario
        print("Running standard DCOPF (LP) for highly congested scenario...")
        lp_highly_results = dcopf(gen_time_series_highly, branch_highly, bus_highly, demand_time_series)
        
        if lp_highly_results is None:
            print("LP DCOPF optimization failed for highly congested scenario.")
            lp_highly_results = None
        else:
            print("LP optimization succeeded for highly congested scenario.")
        
        # Run DCOPF-MIP for highly congested scenario
        if lp_highly_results is not None:  # Only try MIP if LP succeeded
            print("Running DCOPF-MIP for highly congested scenario...")
            mip_highly_results = dcopf_mip(
                gen_time_series_highly, 
                branch_highly, 
                bus_highly, 
                demand_time_series,
                startup_costs=startup_costs,
                shutdown_costs=shutdown_costs
            )
            
            if mip_highly_results is None:
                print("DCOPF-MIP optimization failed for highly congested scenario.")
                mip_highly_results = None
            else:
                print("MIP optimization succeeded for highly congested scenario.")
        else:
            mip_highly_results = None
        
        # Group results for comparison (only including successful optimizations)
        lp_results = {"Normal": lp_normal_results}
        if lp_congested_results is not None:
            lp_results["Congested"] = lp_congested_results
        if lp_highly_results is not None:
            lp_results["Highly Congested"] = lp_highly_results
        
        mip_results = {"Normal": mip_normal_results}
        if mip_congested_results is not None:
            mip_results["Congested"] = mip_congested_results
        if mip_highly_results is not None:
            mip_results["Highly Congested"] = mip_highly_results
        
        # Compare congestion results if we have at least one scenario for each solver
        if lp_results and mip_results:
            compare_congestion_results(lp_results, mip_results)
            
            return lp_results, mip_results
        else:
            print("Not enough successful scenarios to compare. Skipping comparison.")
            return None, None
    
    except Exception as e:
        print(f"Error in congestion analysis: {e}")
        return None, None

def compare_congestion_results(lp_results, mip_results):
    """
    Compare results across different congestion scenarios for both LP and MIP solvers.
    
    Args:
        lp_results: Dictionary of LP results for each scenario {scenario_name: results}
        mip_results: Dictionary of MIP results for each scenario {scenario_name: results}
    """
    print("\n===== Comparing Congestion Scenarios =====")
    
    # Create results directory for congestion analysis
    congestion_dir = os.path.join(results_dir, 'congestion')
    os.makedirs(congestion_dir, exist_ok=True)
    
    # Filter out any None results
    lp_results = {k: v for k, v in lp_results.items() if v is not None}
    mip_results = {k: v for k, v in mip_results.items() if v is not None}
    
    if not lp_results or not mip_results:
        print("Not enough valid results to compare. Skipping comparison.")
        return
    
    # 1. Compare total costs
    print("\n--- Total Cost Comparison ---")
    cost_data = []
    
    for scenario, results in lp_results.items():
        cost_data.append({
            'Scenario': scenario,
            'Solver': 'LP',
            'Cost': results['cost']
        })
    
    for scenario, results in mip_results.items():
        cost_data.append({
            'Scenario': scenario,
            'Solver': 'MIP',
            'Cost': results['cost']
        })
    
    cost_df = pd.DataFrame(cost_data)
    print(cost_df)
    
    # Save cost comparison to CSV
    cost_file = os.path.join(congestion_dir, 'congestion_cost_comparison.csv')
    cost_df.to_csv(cost_file, index=False)
    print(f"Cost comparison saved to {cost_file}")
    
    # Plot cost comparison
    plt.figure(figsize=(10, 6))
    # Pivot the DataFrame for grouped bar chart
    cost_pivot = cost_df.pivot(index='Scenario', columns='Solver', values='Cost')
    cost_pivot.plot(kind='bar')
    plt.title('Total Cost Comparison Across Congestion Scenarios')
    plt.ylabel('Total Cost ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(congestion_dir, 'congestion_cost_comparison.png'))
    plt.close()
    
    # 2. Compare LMPs (Locational Marginal Prices)
    print("\n--- LMP Comparison ---")
    
    # Extract LMPs for LP scenarios
    lmp_data = []
    for scenario, results in lp_results.items():
        lmps = results.get('marginal_prices')
        if lmps is not None and not lmps.empty:
            for _, row in lmps.iterrows():
                lmp_data.append({
                    'Scenario': scenario,
                    'Solver': 'LP',
                    'Bus': int(row['bus']),
                    'Price': row['price']
                })
    
    lmp_df = pd.DataFrame(lmp_data)
    
    # Save LMP comparison to CSV
    if not lmp_df.empty:
        lmp_file = os.path.join(congestion_dir, 'congestion_lmp_comparison.csv')
        lmp_df.to_csv(lmp_file, index=False)
        print(f"LMP comparison saved to {lmp_file}")
        
        # Create grouped bar chart of LMPs by scenario and bus
        plt.figure(figsize=(12, 8))
        
        # Pivot the data to get scenario as columns, bus as index
        lmp_pivot = lmp_df.pivot_table(index='Bus', columns=['Scenario'], values='Price')
        
        # Plot as grouped bar chart
        lmp_pivot.plot(kind='bar')
        plt.title('Locational Marginal Prices Across Congestion Scenarios')
        plt.xlabel('Bus')
        plt.ylabel('Price ($/MWh)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Scenario')
        plt.tight_layout()
        plt.savefig(os.path.join(congestion_dir, 'congestion_lmp_comparison.png'))
        plt.close()
    else:
        print("No LMP data available for comparison.")
    
    # 3. Compare congested lines
    print("\n--- Congested Lines Comparison ---")
    congestion_data = []
    
    for scenario, results in lp_results.items():
        congestion_info = results.get('congestion')
        if congestion_info is not None and not congestion_info.empty:
            # Filter to only show congested lines
            congested_lines = congestion_info[congestion_info['is_congested']]
            for _, row in congested_lines.iterrows():
                congestion_data.append({
                    'Scenario': scenario,
                    'From Bus': int(row['from_bus']),
                    'To Bus': int(row['to_bus']),
                    'Upper Limit Price': row['upper_limit_price'],
                    'Lower Limit Price': row['lower_limit_price']
                })
    
    if congestion_data:
        congestion_df = pd.DataFrame(congestion_data)
        print(congestion_df)
        
        # Save congested lines to CSV
        congestion_file = os.path.join(congestion_dir, 'congested_lines.csv')
        congestion_df.to_csv(congestion_file, index=False)
        print(f"Congested lines information saved to {congestion_file}")
    else:
        print("No congested lines found in any scenario.")
    
    # 4. Compare generation dispatch
    print("\n--- Generation Dispatch Comparison ---")
    
    # Extract generation for each scenario
    for solver_name, results_dict in [('LP', lp_results), ('MIP', mip_results)]:
        if not results_dict:
            continue
            
        plt.figure(figsize=(15, 10))
        
        # Position for subplots
        num_scenarios = len(results_dict)
        num_rows = (num_scenarios + 1) // 2  # Calculate needed rows
        
        for i, (scenario, results) in enumerate(results_dict.items(), 1):
            gen = results.get('generation')
            if gen is not None and not gen.empty:
                if num_scenarios == 1:
                    ax = plt.subplot(1, 1, 1)
                else:
                    ax = plt.subplot(num_rows, 2, i)
                
                # Pivot to show time on x-axis, generators as different bars
                gen_pivot = gen.pivot(index='time', columns='id', values='gen')
                gen_pivot.plot(kind='bar', stacked=True, ax=ax)
                
                # Add title and labels
                ax.set_title(f'{scenario} Scenario - {solver_name} Solver')
                ax.set_xlabel('Time')
                ax.set_ylabel('Generation (MW)')
                ax.legend(title='Generator')
                
                # Format x-axis labels
                x_labels = [t.strftime('%H:%M') for t in gen_pivot.index]
                ax.set_xticklabels(x_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(congestion_dir, f'{solver_name.lower()}_generation_comparison.png'))
        plt.close()
        
        print(f"{solver_name} generation dispatch comparison plot saved.")
    
    print("\nAll congestion scenario comparisons complete and saved to", congestion_dir)

if __name__ == "__main__":
    print("Testing Simplified DCOPF-MIP with Binary Commitment Variables...")
    
    # Run and compare LP vs MIP for single time period
    lp_results, mip_results = run_and_compare()
    
    # Only continue if both solvers succeeded
    if lp_results is None or mip_results is None:
        print("One or both solvers failed for single time period. Skipping varying demand test.")
    else:
        # Run with varying demand to demonstrate startup/shutdown decisions
        mip_varying_results, lp_varying_results = run_with_varying_demand()
        
        if mip_varying_results is None:
            print("Varying demand test failed.")
    
    # Run congestion analysis to compare normal, congested, and highly congested scenarios
    print("\n===== Running Congestion Analysis =====")
    lp_congestion_results, mip_congestion_results = run_congestion_analysis()
    
    if lp_congestion_results is None or mip_congestion_results is None:
        print("Congestion analysis did not produce complete results. Some scenarios may have failed.")
    
    print("\nDCOPF-MIP testing complete! Results saved to", results_dir) 