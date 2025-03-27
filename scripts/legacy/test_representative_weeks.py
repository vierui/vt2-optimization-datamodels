#!/usr/bin/env python3

"""
test_representative_weeks.py

Test script for the representative weeks approach for power system optimization.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Import our modules
from scripts.representative_weeks import run_representative_weeks_dcopf, plot_representative_weeks_results
from scripts.investment import create_test_system

# Create results directory if it doesn't exist
results_dir = os.path.join(project_root, 'results', 'representative_weeks')
os.makedirs(results_dir, exist_ok=True)

def create_synthetic_year_data(year=2023):
    """
    Create synthetic data for a full year to test the representative weeks approach.
    
    Args:
        year: Year to generate data for
        
    Returns:
        Tuple of (gen_data, branch, bus, demand_data)
    """
    print("Creating synthetic data for testing representative weeks...")
    
    # Create a simple test system with representative weeks
    
    # Create base timestamps for each block
    winter_start = datetime(year, 1, 9)
    summer_start = datetime(year, 7, 31)
    autumn_start = datetime(year, 10, 23)
    
    # Create timestamps for each block (168 hours per block)
    winter_hours = [winter_start + timedelta(hours=h) for h in range(168)]
    summer_hours = [summer_start + timedelta(hours=h) for h in range(168)]
    autumn_hours = [autumn_start + timedelta(hours=h) for h in range(168)]
    
    # Create test system for winter
    print("Creating winter week...")
    winter_gen, winter_branch, winter_bus, winter_demand = create_test_system(time_periods=winter_hours)
    
    # Create test system for summer
    print("Creating summer week...")
    summer_gen, summer_branch, summer_bus, summer_demand = create_test_system(time_periods=summer_hours)
    
    # Create test system for spring/autumn
    print("Creating autumn week...")
    autumn_gen, autumn_branch, autumn_bus, autumn_demand = create_test_system(time_periods=autumn_hours)
    
    # Add season identifiers
    winter_gen['season'] = 'winter'
    summer_gen['season'] = 'summer'
    autumn_gen['season'] = 'spring_autumn'
    
    winter_demand['season'] = 'winter'
    summer_demand['season'] = 'summer'
    autumn_demand['season'] = 'spring_autumn'
    
    # Combine all data
    gen_data = pd.concat([winter_gen, summer_gen, autumn_gen], ignore_index=True)
    demand_data = pd.concat([winter_demand, summer_demand, autumn_demand], ignore_index=True)
    
    # Calculate the maximum possible demand across all seasons
    max_demand = demand_data.groupby('time')['pd'].sum().max()
    
    # Update generator capacities and investment parameters for all generators
    
    # Scale gen1's capacity by factor 100 to ensure it can always meet demand
    gen_data.loc[gen_data['id'] == 1, 'pmax'] = max_demand * 100
    print(f"Scaled gen1 capacity to {max_demand * 100:.2f} MW to easily cover peak demand of {max_demand:.2f} MW")
    
    # Apply smaller pmax factors to generators 2-5
    # Gen 2: 60-90% of max demand
    gen_data.loc[gen_data['id'] == 2, 'pmax'] = max_demand * 0.9
    # Gen 3: 50-80% of max demand
    gen_data.loc[gen_data['id'] == 3, 'pmax'] = max_demand * 0.8
    # Gen 4: 40-70% of max demand
    gen_data.loc[gen_data['id'] == 4, 'pmax'] = max_demand * 0.7
    # Gen 5: 30-60% of max demand
    gen_data.loc[gen_data['id'] == 5, 'pmax'] = max_demand * 0.6
    
    # Setup investment parameters for all generators
    # Gen 1: Already installed (no investment required)
    gen_data.loc[gen_data['id'] == 1, 'investment_required'] = 0
    gen_data.loc[gen_data['id'] == 1, 'capex'] = 0
    gen_data.loc[gen_data['id'] == 1, 'lifetime'] = 20
    
    # Gen 2-7: Different investment costs and lifetimes
    gen_data.loc[gen_data['id'] == 2, 'investment_required'] = 1
    gen_data.loc[gen_data['id'] == 2, 'capex'] = 800000  # $0.8M/MW
    gen_data.loc[gen_data['id'] == 2, 'lifetime'] = 15
    
    gen_data.loc[gen_data['id'] == 3, 'investment_required'] = 1
    gen_data.loc[gen_data['id'] == 3, 'capex'] = 700000  # $0.7M/MW
    gen_data.loc[gen_data['id'] == 3, 'lifetime'] = 12
    
    gen_data.loc[gen_data['id'] == 4, 'investment_required'] = 1
    gen_data.loc[gen_data['id'] == 4, 'capex'] = 600000  # $0.6M/MW
    gen_data.loc[gen_data['id'] == 4, 'lifetime'] = 10
    
    gen_data.loc[gen_data['id'] == 5, 'investment_required'] = 1
    gen_data.loc[gen_data['id'] == 5, 'capex'] = 500000  # $0.5M/MW
    gen_data.loc[gen_data['id'] == 5, 'lifetime'] = 8
    
    # Storage units (6 and 7) - update their parameters too
    gen_data.loc[gen_data['id'] == 6, 'investment_required'] = 1
    gen_data.loc[gen_data['id'] == 6, 'capex'] = 350000  # $0.35M/MW
    gen_data.loc[gen_data['id'] == 6, 'lifetime'] = 7
    
    gen_data.loc[gen_data['id'] == 7, 'investment_required'] = 1
    gen_data.loc[gen_data['id'] == 7, 'capex'] = 300000  # $0.3M/MW
    gen_data.loc[gen_data['id'] == 7, 'lifetime'] = 6
    
    print("Updated generator parameters for all units:")
    for gen_id in range(1, 8):
        sample_row = gen_data[gen_data['id'] == gen_id].iloc[0]
        investment_status = "Not Required" if sample_row['investment_required'] == 0 else "Required"
        print(f"Gen {gen_id}: pmax={sample_row['pmax']:.2f} MW, capex=${sample_row['capex']:,.0f}/MW, "
              f"lifetime={sample_row['lifetime']} years, Investment: {investment_status}")
    
    # Use winter branch and bus data for all seasons
    branch = winter_branch
    bus = winter_bus
    
    # Print summary
    print(f"Generated data summary:")
    print(f"- Generator data: {len(gen_data)} rows")
    print(f"- Demand data: {len(demand_data)} rows")
    print(f"- Branch data: {len(branch)} rows")
    print(f"- Bus data: {len(bus)} rows")
    
    return gen_data, branch, bus, demand_data

def generate_results_summary(results, results_dir, test_type):
    """
    Generate a summary of results and save it as a markdown file.
    
    Args:
        results: Dictionary with the results from run_representative_weeks_dcopf
        results_dir: Directory to save the summary
        test_type: String indicating the type of test ('investment' or 'operation')
    """
    print(f"Generating {test_type} results summary...")
    
    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a markdown file
    summary_file = os.path.join(results_dir, f"{test_type}_results_summary.md")
    
    with open(summary_file, 'w') as f:
        f.write(f"# Representative Weeks {test_type.capitalize()} Results Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic results info
        f.write("## System Information\n\n")
        f.write("This summary is generated from a representative weeks model with:\n")
        f.write("- Winter Week (Week 2): 13 weeks weighting\n")
        f.write("- Summer Week (Week 31): 13 weeks weighting\n")
        f.write("- Spring/Autumn Week (Week 43): 26 weeks weighting\n\n")
        
        # Add cost information
        if 'cost' in results:
            f.write("## Cost Summary\n\n")
            f.write(f"- Total Cost: ${results['cost']:,.2f}\n")
            
            if 'investment_cost' in results:
                f.write(f"- Investment Cost: ${results['investment_cost']:,.2f}\n")
                f.write(f"- Operational Cost: ${results.get('operational_cost', results['cost'] - results['investment_cost']):,.2f}\n")
        
        # Add generation summary
        if 'annual_generation' in results:
            f.write("\n## Annual Generation by Asset\n\n")
            f.write("| Asset ID | Generation (MWh) | % of Total |\n")
            f.write("|----------|-----------------|------------|\n")
            
            total_gen = sum([gen for gen in results['annual_generation'].values() if gen > 0])
            
            for asset_id, generation in results['annual_generation'].items():
                if generation > 0:  # Only show positive generation (not charging)
                    percentage = (generation / total_gen) * 100 if total_gen > 0 else 0
                    f.write(f"| {asset_id} | {generation:,.2f} | {percentage:.2f}% |\n")
                    
            # Add charging information separately if there's storage
            charging = {asset_id: -gen for asset_id, gen in results['annual_generation'].items() if gen < 0}
            if charging:
                f.write("\n### Storage Charging\n\n")
                f.write("| Storage ID | Charging (MWh) |\n")
                f.write("|------------|---------------|\n")
                for asset_id, charge in charging.items():
                    f.write(f"| {asset_id} | {charge:,.2f} |\n")
        
        # Add investment decisions if available
        if 'investment_decisions' in results:
            f.write("\n## Investment Decisions\n\n")
            f.write("| Asset ID | Type | Decision | Annualized Cost |\n")
            f.write("|----------|------|----------|----------------|\n")
            
            decisions = results['investment_decisions']
            
            # Get the annualized costs from the original input data
            asset_costs = {}
            if test_type == 'investment' and 'investment_candidates' in results:
                candidates = results.get('investment_candidates', {})
                for i, row in candidates.iterrows():
                    asset_costs[row['id']] = row.get('annualized_cost', 'N/A')
            
            for asset_id, decision in decisions.items():
                decision_text = "Selected" if decision == 1 else "Not Selected"
                asset_type = "Storage" if asset_id >= 6 else "Generator"  # Assume IDs 6+ are storage
                cost = asset_costs.get(asset_id, 'N/A')
                cost_text = f"${cost:,.2f}/yr" if isinstance(cost, (int, float)) else cost
                f.write(f"| {asset_id} | {asset_type} | {decision_text} | {cost_text} |\n")
        
        # Add multi-year implementation plan if available
        if 'implementation_plan' in results:
            f.write("\n## Multi-Year Implementation Plan\n\n")
            plan = results['implementation_plan']
            
            # Create a timeline table
            years = sorted(set().union(*[years.keys() for years in plan.values()]))
            
            f.write("| Asset ID | " + " | ".join([f"Year {year}" for year in years]) + " |\n")
            f.write("|----------|" + "|".join(["------" for _ in years]) + "|\n")
            
            for asset_id, timeline in plan.items():
                row = [f"{asset_id}"]
                for year in years:
                    action = timeline.get(year, "")
                    row.append(action)
                f.write("| " + " | ".join(row) + " |\n")
    
    print(f"Results summary saved to {summary_file}")
    return summary_file

def generate_implementation_plan(investment_decisions, asset_lifetimes, planning_horizon=25, start_year=2023):
    """
    Generate a multi-year implementation plan based on investment decisions and asset lifetimes.
    
    Args:
        investment_decisions: Dictionary of asset IDs and their investment decisions (1=selected, 0=not selected)
        asset_lifetimes: Dictionary of asset IDs and their lifetimes in years
        planning_horizon: Number of years to plan for
        start_year: Starting year
        
    Returns:
        Dictionary mapping asset IDs to dictionaries of {year: action}
    """
    plan = {}
    
    for asset_id, decision in investment_decisions.items():
        plan[asset_id] = {}
        
        if decision == 1:  # Selected for investment
            lifetime = asset_lifetimes.get(asset_id, 0)
            
            if lifetime <= 0:
                continue  # Skip if lifetime is invalid
                
            # Installation in the first year
            plan[asset_id][start_year] = "Install"
            
            # Add replacements based on lifetime
            year = start_year + lifetime
            while year <= start_year + planning_horizon:
                plan[asset_id][year] = "Replace"
                year += lifetime
                
            # Add decommissioning at the end of life
            for year in range(start_year, start_year + planning_horizon + 1):
                if year not in plan[asset_id] and year > start_year:
                    # Check if the previous year had the asset installed
                    prev_year_installed = any(
                        plan[asset_id].get(y) in ["Install", "Replace"] 
                        for y in range(start_year, year)
                    )
                    
                    # Find the most recent installation year
                    installation_years = [y for y, action in plan[asset_id].items() 
                                        if action in ["Install", "Replace"] and y < year]
                    
                    if prev_year_installed and installation_years:
                        most_recent = max(installation_years)
                        if year == most_recent + lifetime:
                            plan[asset_id][year] = "Decommission"
    
    return plan

def run_investment_test():
    """
    Run an investment test of the representative weeks approach.
    """
    print("\n===== Testing Representative Weeks for Investment Planning =====")
    
    # Create synthetic data
    gen_data, branch, bus, demand_data = create_synthetic_year_data()
    
    # Create investment candidates for all generators except gen1 (which is already installed)
    # This includes both conventional generators (2-5) and storage units (6-7)
    investment_candidates = pd.DataFrame({
        'id': [2, 3, 4, 5, 6, 7],  # All investment-required units
        'annualized_cost': [
            800000/15,  # Gen 2: $800k / 15 years
            700000/12,  # Gen 3: $700k / 12 years
            600000/10,  # Gen 4: $600k / 10 years
            500000/8,   # Gen 5: $500k / 8 years
            350000/7,   # Gen 6: $350k / 7 years
            300000/6    # Gen 7: $300k / 6 years
        ],
        'type': ['gen', 'gen', 'gen', 'gen', 'storage', 'storage'],  # Types of assets
        'binary': [True, True, True, True, True, True]  # Binary decision variables
    })
    
    print("Investment candidates:")
    for i, row in investment_candidates.iterrows():
        print(f"Asset {row['id']}: Type={row['type']}, Annualized Cost=${row['annualized_cost']:.2f}/yr")
    
    print("\nRunning investment planning using representative weeks...")
    # Run the representative weeks approach
    investment_results = run_representative_weeks_dcopf(
        gen_data, branch, bus, demand_data,
        investment_mode=True,  # Enable investment decisions
        investment_candidates=investment_candidates
    )
    
    # Add investment_candidates to the results for the summary
    investment_results['investment_candidates'] = investment_candidates
    
    # Create specific results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'results', 'representative_weeks', 'investment_test')
    
    # Generate multi-year implementation plan
    if 'investment_decisions' in investment_results:
        # Extract lifetimes from generator data
        asset_lifetimes = {}
        for i, row in investment_candidates.iterrows():
            asset_id = row['id']
            # Find the first instance of this asset in the gen_data
            asset_data = gen_data[gen_data['id'] == asset_id].iloc[0]
            asset_lifetimes[asset_id] = asset_data['lifetime']
        
        # Generate the implementation plan
        implementation_plan = generate_implementation_plan(
            investment_results['investment_decisions'],
            asset_lifetimes,
            planning_horizon=25,  # Look 25 years ahead
            start_year=2023
        )
        
        # Add to results
        investment_results['implementation_plan'] = implementation_plan
        
        # Print the implementation plan
        print("\nMulti-Year Implementation Plan:")
        for asset_id, timeline in implementation_plan.items():
            if timeline:  # Only show assets with actions
                print(f"Asset {asset_id}:")
                for year, action in sorted(timeline.items()):
                    print(f"  - Year {year}: {action}")
    
    # Plot and save the results
    plot_representative_weeks_results(investment_results, results_dir)
    
    # Generate and save the results summary
    generate_results_summary(investment_results, results_dir, 'investment')
    
    return investment_results

def run_operation_test():
    """
    Run an operational test of the representative weeks approach.
    """
    print("\n===== Testing Representative Weeks for Operations Only =====")
    
    # Create synthetic data
    gen_data, branch, bus, demand_data = create_synthetic_year_data()
    
    print("Running DCOPF using representative weeks...")
    # Run the representative weeks approach
    operation_results = run_representative_weeks_dcopf(
        gen_data, branch, bus, demand_data,
        investment_mode=False  # No investment decisions
    )
    
    # Create specific results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'results', 'representative_weeks', 'operation_test')
    
    # Plot and save the results
    plot_representative_weeks_results(operation_results, results_dir)
    
    # Generate and save the results summary
    generate_results_summary(operation_results, results_dir, 'operation')
    
    return operation_results

if __name__ == "__main__":
    print("Testing the representative weeks approach...")
    
    # Run the operation test (without investment)
    operation_results = run_operation_test()
    
    # Run the investment test
    investment_results = run_investment_test()
    
    print("\nTesting completed.") 