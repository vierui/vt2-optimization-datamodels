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
                if bool(network.generators_installed.get(gen_id, 0) > 0.5):
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
                if bool(network.storage_installed.get(storage_id, 0) > 0.5):
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
                    is_newly_installed = bool(installed > 0.5)
                    if year_idx > 0:
                        prev_year = years[year_idx-1]
                        prev_installed = network.generators_installed_by_year[prev_year][gen_id]
                        is_newly_installed = bool(installed > 0.5 and prev_installed < 0.5)
                    
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
                    is_newly_installed = bool(installed > 0.5)
                    if year_idx > 0:
                        prev_year = years[year_idx-1]
                        prev_installed = network.storage_installed_by_year[prev_year][storage_id]
                        is_newly_installed = bool(installed > 0.5 and prev_installed < 0.5)
                    
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
                    'generators': {g: bool(network.generators_installed_by_year[year][g] > 0.5) 
                                   for g in network.generators.index},
                    'storage': {s: bool(network.storage_installed_by_year[year][s] > 0.5) 
                               for s in network.storage_units.index}
                }
            }
        
        # Add a summary of the cumulative installed capacity by the end of the planning horizon
        last_year = years[-1]
        report['final_installations'] = {
            'generators': {
                str(g): {
                    'installed': bool(network.generators_installed_by_year[last_year][g] > 0.5),
                    'capacity_mw': network.generators.loc[g, 'capacity_mw'],
                    'annual_gen_mwh': network.generators_t_by_year[last_year]['p'][g].sum() 
                        if g in network.generators_t_by_year[last_year]['p'] else 0
                } for g in network.generators.index
            },
            'storage': {
                str(s): {
                    'installed': bool(network.storage_installed_by_year[last_year][s] > 0.5),
                    'capacity_mw': network.storage_units.loc[s, 'p_mw'],
                    'energy_capacity_mwh': network.storage_units.loc[s, 'energy_mwh'],
                    'annual_charge_mwh': network.storage_units_t_by_year[last_year]['p_charge'][s].sum()
                        if s in network.storage_units_t_by_year[last_year]['p_charge'] else 0,
                    'annual_discharge_mwh': network.storage_units_t_by_year[last_year]['p_discharge'][s].sum()
                        if s in network.storage_units_t_by_year[last_year]['p_discharge'] else 0
                } for s in network.storage_units.index
            }
        }
        
        # Add installation history if available
        if hasattr(network, 'asset_installation_history'):
            report['asset_installation_history'] = {
                'generators': {str(g): history for g, history in network.asset_installation_history['generators'].items()},
                'storage': {str(s): history for s, history in network.asset_installation_history['storage'].items()}
            }
            
            # Map relative years to absolute years if available
            if hasattr(network, 'inverse_mapping') and network.inverse_mapping:
                # Create a utility function to convert years
                def convert_years_in_history(history_list):
                    for entry in history_list:
                        rel_year = entry['installation_year']
                        if rel_year in network.inverse_mapping:
                            entry['relative_year'] = rel_year
                            entry['absolute_year'] = network.inverse_mapping[rel_year]
                    return history_list
                
                # Apply conversions
                for g, history in report['asset_installation_history']['generators'].items():
                    report['asset_installation_history']['generators'][g] = convert_years_in_history(history)
                    
                for s, history in report['asset_installation_history']['storage'].items():
                    report['asset_installation_history']['storage'][s] = convert_years_in_history(history)
        
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

def generate_implementation_plan(network, output_file):
    """
    Generate a detailed implementation plan showing which assets have been installed and when
    
    Args:
        network: Network object with multi-year results
        output_file: Path to save the implementation plan
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if we're dealing with multi-year results
        if not (hasattr(network, 'years') and len(network.years) > 0):
            print("Cannot generate implementation plan: no multi-year data found.")
            return False
        
        years = network.years
        
        # Create a structure to track installation status and changes through years
        implementation_plan = {
            'base_year': getattr(network, 'base_year', min(network.years)),
            'planning_horizon': {
                'relative_years': years,
                'absolute_years': [network.inverse_mapping.get(y, y) for y in years] 
                    if hasattr(network, 'inverse_mapping') else years
            },
            'generators': {},
            'storage_units': {},
            'timeline': {},
            'installation_plan': []
        }
        
        # Initialize the timeline for each year
        for year in years:
            implementation_plan['timeline'][year] = {
                'generators': {'new': [], 'active': [], 'decommissioned': []},
                'storage': {'new': [], 'active': [], 'decommissioned': []},
                'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year
            }
        
        # Process generators
        for gen_id in network.generators.index:
            gen_info = {
                'id': gen_id,
                'name': network.generators.loc[gen_id, 'name'] if 'name' in network.generators.columns else f"Generator {gen_id}",
                'type': network.generators.loc[gen_id, 'type'] if 'type' in network.generators.columns else "Unknown",
                'capacity_mw': network.generators.loc[gen_id, 'capacity_mw'],
                'years_active': [],
                'installation_history': []
            }
            
            # Track active years and installation details
            prev_status = False
            for year_idx, year in enumerate(years):
                is_active = network.generators_installed_by_year[year][gen_id] > 0.5
                
                if is_active:
                    gen_info['years_active'].append(year)
                    implementation_plan['timeline'][year]['generators']['active'].append(gen_id)
                
                # Detect newly installed
                is_new = is_active and (year_idx == 0 or network.generators_installed_by_year[years[year_idx-1]][gen_id] < 0.5)
                
                if is_new:
                    installation_info = {
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'action': 'install',
                        'reason': 'capacity_expansion'
                    }
                    gen_info['installation_history'].append(installation_info)
                    
                    # Add to timeline
                    implementation_plan['timeline'][year]['generators']['new'].append(gen_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'generator',
                        'asset_id': gen_id,
                        'asset_name': gen_info['name'],
                        'action': 'install',
                        'capacity_mw': gen_info['capacity_mw'],
                        'asset_type_specific': gen_info['type']
                    })
                
                # Detect decommissioned (previously active, now inactive)
                is_decommissioned = not is_active and (year_idx > 0 and network.generators_installed_by_year[years[year_idx-1]][gen_id] > 0.5)
                
                if is_decommissioned:
                    decommission_info = {
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'action': 'decommission'
                    }
                    gen_info['installation_history'].append(decommission_info)
                    
                    # Add to timeline
                    implementation_plan['timeline'][year]['generators']['decommissioned'].append(gen_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'generator',
                        'asset_id': gen_id,
                        'asset_name': gen_info['name'],
                        'action': 'decommission',
                        'capacity_mw': gen_info['capacity_mw'],
                        'asset_type_specific': gen_info['type']
                    })
                
                prev_status = is_active
            
            # Add detailed info from asset installation history if available
            if hasattr(network, 'asset_installation_history') and 'generators' in network.asset_installation_history:
                if gen_id in network.asset_installation_history['generators']:
                    gen_info['detailed_installation_history'] = network.asset_installation_history['generators'][gen_id]
            
            # Add generator info to implementation plan
            implementation_plan['generators'][gen_id] = gen_info
        
        # Process storage units
        for storage_id in network.storage_units.index:
            storage_info = {
                'id': storage_id,
                'name': network.storage_units.loc[storage_id, 'name'] if 'name' in network.storage_units.columns else f"Storage {storage_id}",
                'p_mw': network.storage_units.loc[storage_id, 'p_mw'],
                'energy_mwh': network.storage_units.loc[storage_id, 'energy_mwh'],
                'years_active': [],
                'installation_history': []
            }
            
            # Track active years and installation details
            prev_status = False
            for year_idx, year in enumerate(years):
                is_active = network.storage_installed_by_year[year][storage_id] > 0.5
                
                if is_active:
                    storage_info['years_active'].append(year)
                    implementation_plan['timeline'][year]['storage']['active'].append(storage_id)
                
                # Detect newly installed
                is_new = is_active and (year_idx == 0 or network.storage_installed_by_year[years[year_idx-1]][storage_id] < 0.5)
                
                if is_new:
                    installation_info = {
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'action': 'install',
                        'reason': 'flexibility_requirement'
                    }
                    storage_info['installation_history'].append(installation_info)
                    
                    # Add to timeline
                    implementation_plan['timeline'][year]['storage']['new'].append(storage_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'storage',
                        'asset_id': storage_id,
                        'asset_name': storage_info['name'],
                        'action': 'install',
                        'capacity_mw': storage_info['p_mw'],
                        'energy_capacity_mwh': storage_info['energy_mwh']
                    })
                
                # Detect decommissioned (previously active, now inactive)
                is_decommissioned = not is_active and (year_idx > 0 and network.storage_installed_by_year[years[year_idx-1]][storage_id] > 0.5)
                
                if is_decommissioned:
                    decommission_info = {
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'action': 'decommission'
                    }
                    storage_info['installation_history'].append(decommission_info)
                    
                    # Add to timeline
                    implementation_plan['timeline'][year]['storage']['decommissioned'].append(storage_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'storage',
                        'asset_id': storage_id,
                        'asset_name': storage_info['name'],
                        'action': 'decommission',
                        'capacity_mw': storage_info['p_mw'],
                        'energy_capacity_mwh': storage_info['energy_mwh']
                    })
                
                prev_status = is_active
            
            # Add detailed info from asset installation history if available
            if hasattr(network, 'asset_installation_history') and 'storage' in network.asset_installation_history:
                if storage_id in network.asset_installation_history['storage']:
                    storage_info['detailed_installation_history'] = network.asset_installation_history['storage'][storage_id]
            
            # Add storage info to implementation plan
            implementation_plan['storage_units'][storage_id] = storage_info
        
        # Sort the installation plan chronologically
        implementation_plan['installation_plan'] = sorted(
            implementation_plan['installation_plan'], 
            key=lambda x: (x['year'], x['asset_type'], x['asset_id'])
        )
        
        # Generate a human-readable summary
        summary = ["# Implementation Plan Summary\n"]
        summary.append(f"## Planning Horizon\n")
        
        # Add planning horizon details
        abs_years = implementation_plan['planning_horizon']['absolute_years']
        rel_years = implementation_plan['planning_horizon']['relative_years']
        summary.append("| Relative Year | Absolute Year |\n")
        summary.append("|--------------|---------------|\n")
        for i in range(len(rel_years)):
            summary.append(f"| {rel_years[i]} | {abs_years[i]} |\n")
        
        summary.append("\n## Installation Timeline\n")
        for year in years:
            abs_year = implementation_plan['timeline'][year]['absolute_year']
            summary.append(f"\n### Year {year} (Absolute: {abs_year})\n")
            
            # New generators
            new_gens = implementation_plan['timeline'][year]['generators']['new']
            if new_gens:
                summary.append("\n#### New Generators\n")
                summary.append("| ID | Name | Type | Capacity (MW) |\n")
                summary.append("|-------|------|------|-------------|\n")
                for gen_id in new_gens:
                    gen = implementation_plan['generators'][gen_id]
                    summary.append(f"| {gen_id} | {gen['name']} | {gen['type']} | {gen['capacity_mw']} |\n")
            
            # New storage
            new_storage = implementation_plan['timeline'][year]['storage']['new']
            if new_storage:
                summary.append("\n#### New Storage Units\n")
                summary.append("| ID | Name | Power (MW) | Energy (MWh) |\n")
                summary.append("|-------|------|-----------|-------------|\n")
                for storage_id in new_storage:
                    storage = implementation_plan['storage_units'][storage_id]
                    summary.append(f"| {storage_id} | {storage['name']} | {storage['p_mw']} | {storage['energy_mwh']} |\n")
            
            # Decommissioned assets
            decom_gens = implementation_plan['timeline'][year]['generators']['decommissioned']
            decom_storage = implementation_plan['timeline'][year]['storage']['decommissioned']
            
            if decom_gens or decom_storage:
                summary.append("\n#### Decommissioned Assets\n")
                summary.append("| Type | ID | Name |\n")
                summary.append("|------|-------|------|\n")
                for gen_id in decom_gens:
                    gen = implementation_plan['generators'][gen_id]
                    summary.append(f"| Generator | {gen_id} | {gen['name']} |\n")
                for storage_id in decom_storage:
                    storage = implementation_plan['storage_units'][storage_id]
                    summary.append(f"| Storage | {storage_id} | {storage['name']} |\n")
        
        # Save the implementation plan to JSON file
        with open(output_file, 'w') as f:
            json.dump(implementation_plan, f, indent=2)
        
        # Save the human-readable summary as markdown
        summary_file = output_file.replace('.json', '.md')
        with open(summary_file, 'w') as f:
            f.writelines(summary)
        
        print(f"Implementation plan saved to: {output_file}")
        print(f"Human-readable summary saved to: {summary_file}")
        return True
        
    except Exception as e:
        print(f"Error generating implementation plan: {e}")
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