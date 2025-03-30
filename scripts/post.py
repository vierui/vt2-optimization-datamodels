#!/usr/bin/env python3
"""
Post-processing module for calculating annual costs from representative season results
"""
import os
import json
import pandas as pd
from datetime import datetime
from pre import SEASON_WEIGHTS

# Constants for annual calculation
SEASON_WEIGHTS = {
    'winter': 13,     # Winter represents 13 weeks
    'summer': 13,     # Summer represents 13 weeks
    'spri_autu': 26   # Spring/Autumn represents 26 weeks
}

def calculate_annual_cost(season_costs):
    """
    Calculate the annual cost based on the representative seasons
    
    Args:
        season_costs: Dictionary with costs for each season (winter, summer, spri_autu)
        
    Returns:
        Total annual cost
    """
    winter_cost = season_costs.get('winter', 0)
    summer_cost = season_costs.get('summer', 0)
    spri_autu_cost = season_costs.get('spri_autu', 0)
    
    annual_cost = (
        SEASON_WEIGHTS['winter'] * winter_cost +
        SEASON_WEIGHTS['summer'] * summer_cost +
        SEASON_WEIGHTS['spri_autu'] * spri_autu_cost
    )
    
    return annual_cost

def save_annual_cost_report(season_costs, output_file="annual_cost_report.json"):
    """
    Save the annual cost calculation to a JSON file
    
    Args:
        season_costs: Dictionary with costs for each season
        output_file: Path to the output file
        
    Returns:
        Path to the output file
    """
    annual_cost = calculate_annual_cost(season_costs)
    
    report = {
        'season_costs': season_costs,
        'annual_cost': annual_cost,
        'calculation': {
            'winter': {
                'cost': season_costs.get('winter', 0),
                'weeks': SEASON_WEIGHTS['winter'],
                'subtotal': SEASON_WEIGHTS['winter'] * season_costs.get('winter', 0)
            },
            'summer': {
                'cost': season_costs.get('summer', 0),
                'weeks': SEASON_WEIGHTS['summer'],
                'subtotal': SEASON_WEIGHTS['summer'] * season_costs.get('summer', 0)
            },
            'spri_autu': {
                'cost': season_costs.get('spri_autu', 0),
                'weeks': SEASON_WEIGHTS['spri_autu'],
                'subtotal': SEASON_WEIGHTS['spri_autu'] * season_costs.get('spri_autu', 0)
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Annual cost report saved to: {output_file}")
    return output_file

def load_costs_from_files(input_files):
    """
    Load costs from optimization result files
    
    Args:
        input_files: Dictionary mapping season names to file paths
        
    Returns:
        Dictionary with costs for each season
    """
    season_costs = {}
    
    for season, filepath in input_files.items():
        if not os.path.exists(filepath):
            print(f"Warning: File not found for {season} season: {filepath}")
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Try to find the cost in different possible locations in the file
                if 'objective_value' in data:
                    season_costs[season] = data['objective_value']
                elif 'total_cost' in data:
                    season_costs[season] = data['total_cost']
                elif 'cost' in data:
                    season_costs[season] = data['cost']
                elif 'value' in data:
                    season_costs[season] = data['value']
                else:
                    print(f"Warning: Could not find cost value in {filepath}")
        except json.JSONDecodeError:
            print(f"Error: File is not valid JSON: {filepath}")
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
    
    return season_costs

def analyze_annual_costs_from_files(winter_file, summer_file, spri_autu_file, output_dir="results"):
    """
    Calculate and report annual costs from individual season result files
    
    Args:
        winter_file: Path to winter season results file
        summer_file: Path to summer season results file
        spri_autu_file: Path to spring/autumn season results file
        output_dir: Directory to save the report
        
    Returns:
        Dictionary with annual cost calculation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load costs from files
    input_files = {
        'winter': winter_file,
        'summer': summer_file,
        'spri_autu': spri_autu_file
    }
    
    # Filter out None values
    input_files = {k: v for k, v in input_files.items() if v is not None}
    
    # Load costs
    season_costs = load_costs_from_files(input_files)
    
    # Calculate annual cost
    annual_cost = calculate_annual_cost(season_costs)
    
    # Print annual cost calculation
    print("\nAnnual cost calculation:")
    for season, cost in season_costs.items():
        weeks = SEASON_WEIGHTS.get(season, 0)
        print(f"{season.capitalize()} cost: {cost:.2f} × {weeks} weeks = {cost * weeks:.2f}")
    print(f"Total annual cost: {annual_cost:.2f}")
    
    # Save annual cost report
    report_path = os.path.join(output_dir, "annual_cost_report.json")
    save_annual_cost_report(season_costs, report_path)
    
    return {
        'season_costs': season_costs,
        'annual_cost': annual_cost,
        'report_path': report_path
    }

def save_detailed_cost_report(network, output_file):
    """
    Save a detailed cost report with OPEX and CAPEX breakdown
    
    Args:
        network: Network object with results
        output_file: Path to save the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if we're dealing with multi-year results
        multi_year = hasattr(network, 'years') and len(network.years) > 0
        
        if multi_year:
            # Handle multi-year planning results
            return save_multi_year_cost_report(network, output_file)
        
        # Single-year planning (original implementation)
        # Initialize cost components
        operational_costs = {}
        capex_costs = {}
        total_opex = 0
        total_capex = 0
        
        # Calculate operational costs for generators
        if hasattr(network, 'generators_t') and 'p' in network.generators_t:
            for gen_id in network.generators.index:
                gen_sum = network.generators_t['p'][gen_id].sum()
                gen_cost = network.generators.loc[gen_id, 'cost_mwh']
                opex = gen_cost * gen_sum
                total_opex += opex
                operational_costs[f'generator_{gen_id}'] = opex
        
        # Calculate CAPEX for installed generators
        if hasattr(network, 'generators_installed'):
            for gen_id in network.generators.index:
                if network.generators_installed.get(gen_id, 0) > 0.5:
                    capacity = network.generators.loc[gen_id, 'capacity_mw']
                    # Use get() with default values to handle missing columns
                    capex_per_mw = network.generators.loc[gen_id].get('capex_per_mw', 0)
                    lifetime = network.generators.loc[gen_id].get('lifetime_years', 25)
                    capex = (capex_per_mw * capacity) / lifetime
                    total_capex += capex
                    capex_costs[f'generator_{gen_id}'] = capex
        
        # Calculate CAPEX for installed storage
        if hasattr(network, 'storage_installed'):
            for storage_id in network.storage_units.index:
                if network.storage_installed.get(storage_id, 0) > 0.5:
                    capacity = network.storage_units.loc[storage_id, 'p_mw']
                    # Use get() with default values to handle missing columns
                    capex_per_mw = network.storage_units.loc[storage_id].get('capex_per_mw', 0)
                    lifetime = network.storage_units.loc[storage_id].get('lifetime_years', 15)
                    capex = (capex_per_mw * capacity) / lifetime
                    total_capex += capex
                    capex_costs[f'storage_{storage_id}'] = capex
        
        # Create the report
        report = {
            'total_cost': total_opex + total_capex,
            'operational_cost': total_opex,
            'capital_cost': total_capex,
            'operational_costs_breakdown': operational_costs,
            'capital_costs_breakdown': capex_costs
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Detailed cost report saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving detailed cost report: {e}")
        return False

def save_multi_year_cost_report(network, output_file):
    """
    Save a detailed cost report for multi-year planning with OPEX and CAPEX breakdown
    
    Args:
        network: Network object with multi-year results
        output_file: Path to save the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        years = network.years
        discount_rate = getattr(network, 'discount_rate', 0.05)  # Default 5% if not specified
        
        # Initialize the report structure
        report = {
            'planning_horizon': {
                'years': years,
                'discount_rate': discount_rate
            },
            'total_discounted_cost': getattr(network, 'total_discounted_cost', 0),
            'years_data': {}
        }
        
        # Process each year's data
        for year_idx, year in enumerate(years):
            discount_factor = 1 / ((1 + discount_rate) ** year_idx)
            
            # Initialize cost components for this year
            operational_costs = {}
            capex_costs = {}
            total_opex = 0
            total_capex = 0
            
            # Calculate operational costs for generators
            if hasattr(network, 'generators_t_by_year') and year in network.generators_t_by_year:
                for gen_id in network.generators.index:
                    gen_sum = network.generators_t_by_year[year]['p'][gen_id].sum()
                    gen_cost = network.generators.loc[gen_id, 'cost_mwh']
                    opex = gen_cost * gen_sum
                    total_opex += opex
                    operational_costs[f'generator_{gen_id}'] = opex
            
            # Calculate CAPEX for generators installed in this year
            if hasattr(network, 'generators_installed_by_year') and year in network.generators_installed_by_year:
                for gen_id in network.generators.index:
                    installed = network.generators_installed_by_year[year][gen_id]
                    
                    # Only count CAPEX in the year when it's first installed
                    is_newly_installed = installed > 0.5
                    if year_idx > 0:
                        prev_year = years[year_idx-1]
                        prev_installed = network.generators_installed_by_year[prev_year][gen_id]
                        is_newly_installed = installed > 0.5 and prev_installed < 0.5
                    
                    if is_newly_installed:
                        capacity = network.generators.loc[gen_id, 'capacity_mw']
                        capex_per_mw = network.generators.loc[gen_id].get('capex_per_mw', 0)
                        lifetime = network.generators.loc[gen_id].get('lifetime_years', 25)
                        capex = (capex_per_mw * capacity) / lifetime
                        total_capex += capex
                        capex_costs[f'generator_{gen_id}'] = capex
            
            # Calculate CAPEX for storage installed in this year
            if hasattr(network, 'storage_installed_by_year') and year in network.storage_installed_by_year:
                for storage_id in network.storage_units.index:
                    installed = network.storage_installed_by_year[year][storage_id]
                    
                    # Only count CAPEX in the year when it's first installed
                    is_newly_installed = installed > 0.5
                    if year_idx > 0:
                        prev_year = years[year_idx-1]
                        prev_installed = network.storage_installed_by_year[prev_year][storage_id]
                        is_newly_installed = installed > 0.5 and prev_installed < 0.5
                    
                    if is_newly_installed:
                        capacity = network.storage_units.loc[storage_id, 'p_mw']
                        capex_per_mw = network.storage_units.loc[storage_id].get('capex_per_mw', 0)
                        lifetime = network.storage_units.loc[storage_id].get('lifetime_years', 15)
                        capex = (capex_per_mw * capacity) / lifetime
                        total_capex += capex
                        capex_costs[f'storage_{storage_id}'] = capex
            
            # Calculate total year costs and discounted costs
            year_total_cost = total_opex + total_capex
            year_discounted_cost = year_total_cost * discount_factor
            
            # Add year data to the report
            report['years_data'][year] = {
                'total_cost': year_total_cost,
                'discounted_cost': year_discounted_cost,
                'discount_factor': discount_factor,
                'operational_cost': total_opex,
                'capital_cost': total_capex,
                'operational_costs_breakdown': operational_costs,
                'capital_costs_breakdown': capex_costs,
                'installations': {
                    'generators': {g: network.generators_installed_by_year[year][g] > 0.5 
                                   for g in network.generators.index},
                    'storage': {s: network.storage_installed_by_year[year][s] > 0.5 
                               for s in network.storage_units.index}
                }
            }
        
        # Add a summary of the cumulative installed capacity by the end of the planning horizon
        last_year = years[-1]
        report['final_installations'] = {
            'generators': {
                str(g): {
                    'installed': network.generators_installed_by_year[last_year][g] > 0.5,
                    'capacity_mw': network.generators.loc[g, 'capacity_mw'],
                    'annual_gen_mwh': network.generators_t_by_year[last_year]['p'][g].sum() 
                        if g in network.generators_t_by_year[last_year]['p'] else 0
                } for g in network.generators.index
            },
            'storage': {
                str(s): {
                    'installed': network.storage_installed_by_year[last_year][s] > 0.5,
                    'capacity_mw': network.storage_units.loc[s, 'p_mw'],
                    'energy_capacity_mwh': network.storage_units.loc[s, 'energy_mwh'],
                    'annual_charge_mwh': network.storage_units_t_by_year[last_year]['p_charge'][s].sum()
                        if s in network.storage_units_t_by_year[last_year]['p_charge'] else 0,
                    'annual_discharge_mwh': network.storage_units_t_by_year[last_year]['p_discharge'][s].sum()
                        if s in network.storage_units_t_by_year[last_year]['p_discharge'] else 0
                } for s in network.storage_units.index
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Multi-year cost report saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving multi-year cost report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Annual cost calculation from representative seasons')
    parser.add_argument('--winter-file', type=str,
                      help='Path to winter season result file')
    parser.add_argument('--summer-file', type=str,
                      help='Path to summer season result file')
    parser.add_argument('--spri-autu-file', type=str,
                      help='Path to spring/autumn season result file')
    parser.add_argument('--output-dir', type=str, default='results/annual',
                      help='Directory to store the annual cost report')
    
    args = parser.parse_args()
    
    # Check if at least one file is provided
    if not args.winter_file and not args.summer_file and not args.spri_autu_file:
        print("Error: At least one result file must be specified")
        import sys
        sys.exit(1)
    
    # Calculate annual cost
    analyze_annual_costs_from_files(
        args.winter_file,
        args.summer_file,
        args.spri_autu_file,
        args.output_dir
    ) 