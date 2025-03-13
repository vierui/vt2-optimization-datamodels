#!/usr/bin/env python3

"""
test_dcopf_mip.py

Test script for the Simplified Mixed Integer Programming DC Optimal Power Flow (DCOPF-MIP) implementation.
This script:
1. Creates a simple power system model directly
2. Runs the simplified DCOPF-MIP solver with binary commitment variables
3. Compares the results with the standard DCOPF solver
4. Displays the unit commitment decisions and startup/shutdown costs

The simplified MIP implementation only includes binary commitment variables and 
startup/shutdown variables, without linking constraints or temporal operational constraints.
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
results_dir = os.path.join(project_root, 'results', 'test-mip')
os.makedirs(results_dir, exist_ok=True)

# Import the modules
from scripts.dcopf import dcopf
from scripts.dcopf_mip import dcopf_mip

def create_simple_power_system_direct(time_periods=None):
    """
    Simple 5-bus power system model created directly from the repository.
    
    Args:
        time_periods: Optional list of time periods to use (default: current time)
    
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
        
        return mip_results
    
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Testing Simplified DCOPF-MIP with Binary Commitment Variables...")
    
    # Run and compare LP vs MIP for single time period
    lp_results, mip_results = run_and_compare()
    
    # Only continue if both solvers succeeded
    if lp_results is None or mip_results is None:
        print("One or both solvers failed. Terminating test.")
        sys.exit(1)
    
    # Run with varying demand to demonstrate startup/shutdown decisions
    mip_varying_demand_results = run_with_varying_demand()
    
    if mip_varying_demand_results is None:
        print("Varying demand test failed. Terminating test.")
        sys.exit(1)
    
    print("\nDCOPF-MIP testing complete! Results saved to", results_dir) 