#!/usr/bin/env python3

"""
marginal_price_analysis.py

A learning tool to understand marginal prices in power systems using the DCOPF implementation.
This script:
1. Creates a simple power system model with 5 buses (one empty)
2. Runs the DCOPF solver
3. Extracts and analyzes the marginal prices
4. Allows modification of line limits to observe congestion effects
5. Visualizes the results with simple plots
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scripts.dcopf import dcopf

def create_simple_power_system(line_limits=None):
    """
    Simple 5-bus power system model for learning about marginal prices.
    One bus will be empty (no generation or load) to observe price formation.
    
    Args:
        line_limits: Optional dictionary to override default line limits {(from_bus, to_bus): limit}
    
    Returns:
        Tuple of (gen_time_series, branch, bus, demand_time_series)
    """
    # Create time periods - just use a single time period for simplicity
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    time_periods = [current_time]
    
    # 1. Read bus data from CSV file
    # bus_i: bus number
    # type: 1=PQ bus (load), 2=PV bus (generator), 3=reference bus
    # Pd: real power demand (MW)
    # Qd: reactive power demand (MVAr) - not used in DC model
    # Gs, Bs: shunt conductance and susceptance - not used in DC model
    # area: area number
    # Vm: voltage magnitude - not used in DC model
    # Va: voltage angle - not used in DC model
    # baseKV: base voltage (kV) - not used in DC model
    # zone: loss zone - not used in DC model
    # Vmax, Vmin: max/min voltage magnitude - not used in DC model
    bus_file = os.path.join('data', 'working', 'bus.csv')
    bus = pd.read_csv(bus_file)
    
    # Ensure numeric types for critical columns
    bus['bus_i'] = bus['bus_i'].astype(int)
    bus['type'] = bus['type'].astype(int)
    bus['Pd'] = bus['Pd'].astype(float)
    bus['Qd'] = bus['Qd'].astype(float)
    
    # 2. Read branch data from CSV file
    # fbus, tbus: from/to bus number
    # r, x: resistance and reactance (p.u.)
    # b: total line charging susceptance (p.u.)
    # ratea: MVA rating A (long term rating)
    # rateb, ratec: MVA rating B, C (short term, emergency) - not used
    # ratio: transformer off nominal turns ratio - not used in DC model
    # angle: transformer phase shift angle (degrees) - not used in DC model
    # status: initial branch status, 1 = in-service, 0 = out-of-service
    # angmin, angmax: min/min angle difference - not used in DC model
    branch_file = os.path.join('data', 'working', 'branch.csv')
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
    
    # Override line limits if provided
    if line_limits:
        for (from_bus, to_bus), limit in line_limits.items():
            mask = (branch['fbus'] == from_bus) & (branch['tbus'] == to_bus)
            if mask.any():
                branch.loc[mask, 'ratea'] = limit
    
    # 3. Create generators
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
    
    # 4. Create demand
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

def run_dcopf_analysis(line_limits=None):
    """
    Run DCOPF analysis with the simple power system model.
    
    Args:
        line_limits: Optional dictionary to override default line limits {(from_bus, to_bus): limit}
    
    Returns:
        DCOPF results dictionary
    """
    # Create the power system model
    gen_time_series, branch, bus, demand_time_series = create_simple_power_system(line_limits)
    
    # Our new dcopf.py implementation will automatically use CPLEX if available,
    # otherwise it will fall back to PuLP with CBC solver
    results = dcopf(gen_time_series, branch, bus, demand_time_series)
    
    return results

def analyze_marginal_prices(results):
    """
    Analyze the marginal prices from the DCOPF results.
    
    Args:
        results: DCOPF results dictionary
    """
    if results is None:
        print("DCOPF optimization failed.")
        return
    
    # Extract marginal prices
    prices = results.get('marginal_prices')
    
    if prices is None or prices.empty:
        print("\n=== Marginal Prices ===")
        print("Error: No marginal prices found in the CPLEX solution.")
        return None
    
    # Print marginal prices for each bus
    print("\n=== Marginal Prices ($/MWh) ===")
    for _, row in prices.iterrows():
        print(f"Bus {int(row['bus'])}: ${row['price']:.2f}/MWh")
    
    # Check for congestion
    congestion = results['congestion']
    congested_lines = congestion[congestion['is_congested']]
    
    if len(congested_lines) > 0:
        print("\n=== Congested Lines ===")
        for _, row in congested_lines.iterrows():
            print(f"Line {int(row['from_bus'])} -> {int(row['to_bus'])}")
            print(f"  Upper limit shadow price: ${row['upper_limit_price']:.2f}/MW")
            print(f"  Lower limit shadow price: ${row['lower_limit_price']:.2f}/MW")
    else:
        print("\nNo congested lines.")
    
    # Print generation dispatch
    print("\n=== Generation Dispatch (MW) ===")
    generation = results['generation']
    for _, row in generation.iterrows():
        print(f"Generator {int(row['id'])} at Bus {int(row['node'])}: {row['gen']:.2f} MW")
    
    # Print flows
    print("\n=== Line Flows (MW) ===")
    flows = results['flows']
    for _, row in flows.iterrows():
        print(f"Line {int(row['from_bus'])} -> {int(row['to_bus'])}: {row['flow']:.2f} MW")
        
    # Calculate total cost
    total_cost = results['cost']
    print(f"\nTotal System Cost: ${total_cost:.2f}")
    
    return prices

def plot_marginal_prices(results):
    """
    Plot the marginal prices from the DCOPF results.
    
    Args:
        results: DCOPF results dictionary
    """
    if results is None:
        print("DCOPF optimization failed.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results/marginal_price_analysis', exist_ok=True)
    
    # Extract marginal prices
    prices = results.get('marginal_prices')
    if prices is None or prices.empty:
        print("Error: No marginal prices available to plot.")
        return None
    
    # Create a bar chart of marginal prices
    plt.figure(figsize=(10, 6))
    plt.bar(prices['bus'], prices['price'])
    plt.xlabel('Bus')
    plt.ylabel('Marginal Price ($/MWh)')
    plt.title('Marginal Prices at Each Bus')
    plt.grid(True, alpha=0.3)
    plt.xticks(prices['bus'])
    plt.tight_layout()
    
    # Save to results folder
    plot_path = 'results/marginal_price_analysis/marginal_prices.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Create a plot of generation dispatch
    generation = results['generation']
    gen_by_bus = generation.groupby('node')['gen'].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(gen_by_bus['node'], gen_by_bus['gen'])
    plt.xlabel('Bus')
    plt.ylabel('Generation (MW)')
    plt.title('Generation Dispatch by Bus')
    plt.grid(True, alpha=0.3)
    plt.xticks(gen_by_bus['node'])
    plt.tight_layout()
    
    # Save to results folder
    plot_path = 'results/marginal_price_analysis/generation_dispatch.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plots saved in results/marginal_price_analysis/")
    
    return prices

def compare_scenarios(scenarios):
    """
    Compare marginal prices across different scenarios.
    
    Args:
        scenarios: Dictionary of {scenario_name: results}
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/marginal_price_analysis', exist_ok=True)
    
    # 1. Compare marginal prices
    # Extract marginal prices for each scenario
    scenario_prices = {}
    for name, results in scenarios.items():
        if results is not None:
            prices = results.get('marginal_prices')
            if prices is None or prices.empty:
                # If we don't have prices in the results, analyze them
                prices = analyze_marginal_prices(results)
            
            if prices is not None and not prices.empty:
                scenario_prices[name] = prices
    
    if not scenario_prices:
        print("No valid scenarios to compare.")
        return
    
    # Create a grouped bar chart of marginal prices
    plt.figure(figsize=(12, 8))
    
    # Get all buses
    all_buses = set()
    for prices in scenario_prices.values():
        all_buses.update(prices['bus'])
    all_buses = sorted(all_buses)
    
    # Set width of bars
    bar_width = 0.8 / len(scenario_prices)
    
    # Set position of bars on x axis
    positions = {}
    for i, bus in enumerate(all_buses):
        positions[bus] = i
    
    # Plot bars for each scenario
    for i, (name, prices) in enumerate(scenario_prices.items()):
        # Create a dictionary of bus -> price for easy lookup
        bus_to_price = {row['bus']: row['price'] for _, row in prices.iterrows()}
        
        # Get prices for all buses (use 0 if bus not in this scenario)
        scenario_values = [bus_to_price.get(bus, 0) for bus in all_buses]
        
        # Calculate bar positions
        bar_positions = [pos + (i - len(scenario_prices)/2 + 0.5) * bar_width for pos in positions.values()]
        
        plt.bar(bar_positions, scenario_values, width=bar_width, label=name)
    
    plt.xlabel('Bus')
    plt.ylabel('Marginal Price ($/MWh)')
    plt.title('Comparison of Marginal Prices Across Scenarios')
    plt.xticks(list(positions.values()), all_buses)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to results folder
    plot_path = 'results/marginal_price_analysis/price_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    
    print("Marginal price comparison plot saved in results/marginal_price_analysis/")
    
    # 2. Compare line flows
    # Extract flows for each scenario
    scenario_flows = {}
    for name, results in scenarios.items():
        if results is not None:
            flows = results.get('flows')
            if flows is not None and not flows.empty:
                scenario_flows[name] = flows
    
    if not scenario_flows:
        print("No valid flow data to compare.")
        return
    
    # Create a grouped bar chart of line flows
    plt.figure(figsize=(14, 8))
    
    # Create line labels
    all_lines = set()
    for flows in scenario_flows.values():
        for _, row in flows.iterrows():
            line_label = f"{int(row['from_bus'])}->{int(row['to_bus'])}"
            all_lines.add(line_label)
    all_lines = sorted(all_lines)
    
    # Set width of bars
    bar_width = 0.8 / len(scenario_flows)
    
    # Set position of bars on x axis
    positions = {}
    for i, line in enumerate(all_lines):
        positions[line] = i
    
    # Plot bars for each scenario
    for i, (name, flows) in enumerate(scenario_flows.items()):
        # Create a dictionary of line -> flow for easy lookup
        line_to_flow = {}
        for _, row in flows.iterrows():
            line_label = f"{int(row['from_bus'])}->{int(row['to_bus'])}"
            line_to_flow[line_label] = row['flow']
        
        # Get flows for all lines (use 0 if line not in this scenario)
        scenario_values = [line_to_flow.get(line, 0) for line in all_lines]
        
        # Calculate bar positions
        bar_positions = [pos + (i - len(scenario_flows)/2 + 0.5) * bar_width for pos in positions.values()]
        
        plt.bar(bar_positions, scenario_values, width=bar_width, label=name)
    
    plt.xlabel('Line (From->To)')
    plt.ylabel('Flow (MW)')
    plt.title('Comparison of Line Flows Across Scenarios')
    plt.xticks(list(positions.values()), all_lines, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to results folder
    plot_path = 'results/marginal_price_analysis/flow_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    
    print("Line flow comparison plot saved in results/marginal_price_analysis/")

def main():
    """
    Main function to run the marginal price analysis.
    """
    print("=== Marginal Price Analysis in Power Systems ===")
    print("This script demonstrates how marginal prices are formed in power systems.")
    
    # Run base case (no congestion)
    print("\n--- Base Case (No Congestion) ---")
    base_results = run_dcopf_analysis()
    base_prices = analyze_marginal_prices(base_results)
    plot_marginal_prices(base_results)
    
    # Run congested case (reduce line limit to create congestion)
    print("\n--- Congested Case ---")
    # Reduce the limit on line 1->2 to create congestion
    congested_results = run_dcopf_analysis(line_limits={(1, 2): 50})
    congested_prices = analyze_marginal_prices(congested_results)
    plot_marginal_prices(congested_results)
    
    # Run highly congested case (reduce multiple line limits)
    print("\n--- Highly Congested Case ---")
    highly_congested_results = run_dcopf_analysis(line_limits={(1, 2): 30, (2, 5): 40})
    highly_congested_prices = analyze_marginal_prices(highly_congested_results)
    plot_marginal_prices(highly_congested_results)
    
    # Compare all scenarios
    scenarios = {
        "Base Case": base_results,
        "Congested Case": congested_results,
        "Highly Congested Case": highly_congested_results
    }
    compare_scenarios(scenarios)
    
    print("\n=== Analysis Complete ===")
    print("All plots have been saved to the results/marginal_price_analysis/ directory.")
    print("You can modify the line limits in the code to observe different congestion patterns.")
    print("The relationship between congestion and marginal prices is demonstrated in the results.")
    print("\nUsing CPLEX solver to get accurate marginal prices (dual variables) from the optimization.")

if __name__ == "__main__":
    main() 