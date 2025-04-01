#!/usr/bin/env python3
"""
Post-processing module for calculating annual costs from representative season results
"""
import os
import json
import pandas as pd
from datetime import datetime
from pre import SEASON_WEIGHTS
import matplotlib.pyplot as plt
import numpy as np

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

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
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        # Generate plots
        generate_cost_plots(report, output_file.replace('.json', '.png'))
        
        print(f"Annual cost report saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving annual cost report: {e}")
        return False

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
            json.dump(report, f, indent=2, cls=NumpyEncoder)
            
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
            
            # Track generation and load for this year
            total_generation = 0
            total_load = 0
            
            if hasattr(network, 'generators_t_by_year') and year in network.generators_t_by_year:
                for gen_id in network.generators.index:
                    if gen_id in network.generators_t_by_year[year]['p']:
                        total_generation += network.generators_t_by_year[year]['p'][gen_id].sum()
                        
            if hasattr(network, 'loads_t_by_year') and year in network.loads_t_by_year:
                for load_id in network.loads.index:
                    if load_id in network.loads_t_by_year[year]['p']:
                        total_load += network.loads_t_by_year[year]['p'][load_id].sum()
            
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
                },
                'total_generation': total_generation,
                'total_load': total_load,
                'mismatch': total_generation - total_load
            }
            
            # Store year summary in network for use in other functions
            if not hasattr(network, 'year_summaries'):
                network.year_summaries = {}
            
            network.year_summaries[year] = {
                'total_generation': total_generation,
                'total_load': total_load,
                'total_cost': year_total_cost,
                'discounted_cost': year_discounted_cost
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
            json.dump(report, f, indent=2, cls=NumpyEncoder)
            
        # Generate generation-load mismatch plot
        plot_file = output_file.replace('.json', '_mismatch.png')
        plot_generation_load_mismatch(network, plot_file)
            
        print(f"Multi-year cost report saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving multi-year cost report: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_implementation_plan(implementation_plan, output_file):
    """
    Generate a visual representation of the implementation plan showing asset installations, 
    replacements, and decommissionings over the planning horizon
    
    Args:
        implementation_plan: Implementation plan dictionary or path to implementation plan JSON file
        output_file: Path to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # If a file path is provided, load the implementation plan
        if isinstance(implementation_plan, str):
            with open(implementation_plan, 'r') as f:
                implementation_plan = json.load(f)
        
        # Extract years and assets
        rel_years = implementation_plan['planning_horizon']['relative_years']
        abs_years = implementation_plan['planning_horizon']['absolute_years']
        
        # Get all generators and storage units
        generators = list(implementation_plan['generators'].keys())
        storage_units = list(implementation_plan['storage_units'].keys())
        
        # Prepare data for plotting
        all_assets = generators + storage_units
        asset_y_positions = {asset: i for i, asset in enumerate(all_assets)}
        
        # Create figure and axis
        fig = plt.figure(figsize=(12, max(8, len(all_assets) * 0.4)), dpi=100)
        ax = fig.add_subplot(111)
        
        # Set up plot parameters
        ax.set_title("Implementation Plan - Asset Timeline", fontsize=14, fontweight='bold')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Assets", fontsize=12)
        
        # Set y-axis labels
        ax.set_yticks(list(range(len(all_assets))))
        
        # Create labels with asset type prefix
        y_labels = []
        for asset in all_assets:
            if asset in generators:
                asset_info = implementation_plan['generators'][asset]
                label = f"G: {asset_info['name']} ({asset_info['type']})"
            else:
                asset_info = implementation_plan['storage_units'][asset]
                label = f"S: {asset_info['name']}"
            y_labels.append(label)
            
        ax.set_yticklabels(y_labels)
        
        # Create x-axis with relative and absolute years
        ax.set_xticks(rel_years)
        if rel_years != abs_years:
            ax.set_xticklabels([f"{rel}\n({abs})" for rel, abs in zip(rel_years, abs_years)])
        
        # Draw active asset periods
        for asset in all_assets:
            if asset in generators:
                asset_info = implementation_plan['generators'][asset]
                color = 'lightgreen'
            else:
                asset_info = implementation_plan['storage_units'][asset]
                color = 'lightblue'
            
            active_years = asset_info['years_active']
            
            if active_years:
                # Group consecutive years
                year_groups = []
                current_group = [active_years[0]]
                
                for year in active_years[1:]:
                    if year == current_group[-1] + 1:
                        current_group.append(year)
                    else:
                        year_groups.append(current_group)
                        current_group = [year]
                
                year_groups.append(current_group)
                
                # Draw rectangles for active periods
                for group in year_groups:
                    ax.add_patch(plt.Rectangle(
                        (min(group) - 0.4, asset_y_positions[asset] - 0.3),
                        len(group), 0.6, alpha=0.7, color=color
                    ))
        
        # Mark installation and replacement events
        for item in implementation_plan['installation_plan']:
            year = item['year']
            asset_type = item['asset_type']
            asset_id = str(item['asset_id'])
            action = item['action']
            
            y_pos = asset_y_positions[asset_id]
            
            if action == 'install':
                is_replacement = 'is_replacement' in item and item['is_replacement']
                if is_replacement:
                    # Replacement: add diamond marker
                    ax.scatter([year], [y_pos], marker='D', s=120, color='orange', 
                              zorder=10, label='Replacement' if y_pos == 0 else "")
                else:
                    # New installation: add triangle marker
                    ax.scatter([year], [y_pos], marker='^', s=120, color='green', 
                              zorder=10, label='New Installation' if y_pos == 0 else "")
            
            elif action == 'decommission':
                # Decommissioning: add X marker
                ax.scatter([year], [y_pos], marker='x', s=120, color='red', 
                          zorder=10, label='Decommission' if y_pos == 0 else "")
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Create a custom legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color='lightgreen', alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, color='lightblue', alpha=0.7),
            plt.Line2D([0], [0], marker='^', color='white', markerfacecolor='green', markersize=12),
            plt.Line2D([0], [0], marker='D', color='white', markerfacecolor='orange', markersize=12),
            plt.Line2D([0], [0], marker='x', color='red', markersize=12)
        ]
        labels = ['Generator Active', 'Storage Active', 'New Installation', 'Replacement', 'Decommission']
        
        # Place legend outside plot area
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                 shadow=True, ncol=5)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Implementation plan visualization saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error generating implementation plan visualization: {e}")
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
                'generators': {'new': [], 'active': [], 'decommissioned': [], 'replaced': []},
                'storage': {'new': [], 'active': [], 'decommissioned': [], 'replaced': []},
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
                is_newly_installed = network.generators_first_install_by_year[year][gen_id] > 0.5
                
                # Check if this is a replacement
                is_replacement = False
                if hasattr(network, 'generators_replacement_by_year') and year in network.generators_replacement_by_year:
                    if gen_id in network.generators_replacement_by_year[year]:
                        replacement_val = network.generators_replacement_by_year[year][gen_id]
                        is_replacement = replacement_val is not None and replacement_val > 0.5
                
                if is_active:
                    gen_info['years_active'].append(year)
                    implementation_plan['timeline'][year]['generators']['active'].append(gen_id)
                
                if is_newly_installed:
                    installation_info = {
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'action': 'install',
                        'reason': 'replacement' if is_replacement else 'capacity_expansion',
                        'is_replacement': is_replacement
                    }
                    gen_info['installation_history'].append(installation_info)
                    
                    # Add to timeline in the appropriate category
                    if is_replacement:
                        implementation_plan['timeline'][year]['generators']['replaced'].append(gen_id)
                    else:
                        implementation_plan['timeline'][year]['generators']['new'].append(gen_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'generator',
                        'asset_id': gen_id,
                        'asset_name': gen_info['name'],
                        'action': 'install',
                        'is_replacement': is_replacement,
                        'capacity_mw': gen_info['capacity_mw'],
                        'type': gen_info['type'],
                        'cost': network.generators.loc[gen_id].get('capex_per_mw', 0) * gen_info['capacity_mw'],
                        'lifetime_years': network.generators.loc[gen_id].get('lifetime_years', 25)
                    })
                
                # Detect decommissioned generators (were active in previous year but not in current year)
                if year_idx > 0 and not is_active and network.generators_installed_by_year[years[year_idx-1]][gen_id] > 0.5:
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
                        'type': gen_info['type']
                    })
            
            # Add to generators list
            implementation_plan['generators'][str(gen_id)] = gen_info
        
        # Process storage units
        for storage_id in network.storage_units.index:
            storage_info = {
                'id': storage_id,
                'name': network.storage_units.loc[storage_id, 'name'] if 'name' in network.storage_units.columns else f"Storage {storage_id}",
                'capacity_mw': network.storage_units.loc[storage_id, 'p_mw'],
                'energy_capacity_mwh': network.storage_units.loc[storage_id, 'energy_mwh'],
                'years_active': [],
                'installation_history': []
            }
            
            # Track active years and installation details
            for year_idx, year in enumerate(years):
                is_active = network.storage_installed_by_year[year][storage_id] > 0.5
                is_newly_installed = network.storage_first_install_by_year[year][storage_id] > 0.5
                
                # Check if this is a replacement
                is_replacement = False
                if hasattr(network, 'storage_replacement_by_year') and year in network.storage_replacement_by_year:
                    if storage_id in network.storage_replacement_by_year[year]:
                        replacement_val = network.storage_replacement_by_year[year][storage_id]
                        is_replacement = replacement_val is not None and replacement_val > 0.5
                
                if is_active:
                    storage_info['years_active'].append(year)
                    implementation_plan['timeline'][year]['storage']['active'].append(storage_id)
                
                if is_newly_installed:
                    installation_info = {
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'action': 'install',
                        'reason': 'replacement' if is_replacement else 'capacity_expansion',
                        'is_replacement': is_replacement
                    }
                    storage_info['installation_history'].append(installation_info)
                    
                    # Add to timeline in the appropriate category
                    if is_replacement:
                        implementation_plan['timeline'][year]['storage']['replaced'].append(storage_id)
                    else:
                        implementation_plan['timeline'][year]['storage']['new'].append(storage_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'storage',
                        'asset_id': storage_id,
                        'asset_name': storage_info['name'],
                        'action': 'install',
                        'is_replacement': is_replacement,
                        'capacity_mw': storage_info['capacity_mw'],
                        'energy_capacity_mwh': storage_info['energy_capacity_mwh'],
                        'cost': network.storage_units.loc[storage_id].get('capex_per_mw', 0) * storage_info['capacity_mw'],
                        'lifetime_years': network.storage_units.loc[storage_id].get('lifetime_years', 15)
                    })
                
                # Detect decommissioned storage units
                if year_idx > 0 and not is_active and network.storage_installed_by_year[years[year_idx-1]][storage_id] > 0.5:
                    implementation_plan['timeline'][year]['storage']['decommissioned'].append(storage_id)
                    
                    # Add to installation plan chronology
                    implementation_plan['installation_plan'].append({
                        'year': year,
                        'absolute_year': network.inverse_mapping.get(year, year) if hasattr(network, 'inverse_mapping') else year,
                        'asset_type': 'storage',
                        'asset_id': storage_id,
                        'asset_name': storage_info['name'],
                        'action': 'decommission',
                        'capacity_mw': storage_info['capacity_mw'],
                        'energy_capacity_mwh': storage_info['energy_capacity_mwh']
                    })
            
            # Add to storage units list
            implementation_plan['storage_units'][str(storage_id)] = storage_info
        
        # Sort installation plan by year
        implementation_plan['installation_plan'].sort(key=lambda x: (x['year'], x['asset_type'], x['asset_id']))
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(implementation_plan, f, indent=2, cls=NumpyEncoder)
        
        # Generate a human-readable summary
        summary_file = output_file.replace('.json', '.md')
        generate_plan_summary(implementation_plan, summary_file)
        
        # Generate a visual implementation plan
        plot_file = output_file.replace('.json', '_visual.png')
        plot_implementation_plan(implementation_plan, plot_file)
        
        print(f"Implementation plan saved to: {output_file}")
        print(f"Human-readable summary saved to: {summary_file}")
        print(f"Visual implementation plan saved to: {plot_file}")
        return True
        
    except Exception as e:
        print(f"Error generating implementation plan: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_plan_summary(implementation_plan, output_file):
    """
    Generate a human-readable markdown summary of the implementation plan
    
    Args:
        implementation_plan: Implementation plan dictionary
        output_file: Path to save the summary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate markdown summary
        summary = ["# Implementation Plan Summary\n"]
        summary.append("## Planning Horizon\n")
        
        # Add planning horizon details
        abs_years = implementation_plan['planning_horizon']['absolute_years']
        rel_years = implementation_plan['planning_horizon']['relative_years']
        summary.append("| Relative Year | Absolute Year |\n")
        summary.append("|--------------|---------------|\n")
        for i in range(len(rel_years)):
            summary.append(f"| {rel_years[i]} | {abs_years[i]} |\n")
        
        summary.append("\n## Installation Timeline\n")
        for year in rel_years:
            abs_year = implementation_plan['timeline'][year]['absolute_year']
            summary.append(f"\n### Year {year} (Absolute: {abs_year})\n")
            
            # New generators
            new_gens = implementation_plan['timeline'][year]['generators']['new']
            if new_gens:
                summary.append("\n#### New Generators\n")
                summary.append("| ID | Name | Type | Capacity (MW) | Cost |\n")
                summary.append("|-------|------|------|-------------|------|\n")
                for gen_id in new_gens:
                    gen = implementation_plan['generators'][str(gen_id)]
                    cost = 0
                    # Find the cost from the installation plan
                    for item in implementation_plan['installation_plan']:
                        if item['year'] == year and item['asset_type'] == 'generator' and item['asset_id'] == gen_id:
                            cost = item.get('cost', 0)
                            break
                    summary.append(f"| {gen_id} | {gen['name']} | {gen['type']} | {gen['capacity_mw']} | {cost:,.0f} |\n")
            
            # Replaced generators
            replaced_gens = implementation_plan['timeline'][year]['generators']['replaced']
            if replaced_gens:
                summary.append("\n#### Replaced Generators\n")
                summary.append("| ID | Name | Type | Capacity (MW) | Cost |\n")
                summary.append("|-------|------|------|-------------|------|\n")
                for gen_id in replaced_gens:
                    gen = implementation_plan['generators'][str(gen_id)]
                    cost = 0
                    # Find the cost from the installation plan
                    for item in implementation_plan['installation_plan']:
                        if item['year'] == year and item['asset_type'] == 'generator' and item['asset_id'] == gen_id:
                            cost = item.get('cost', 0)
                            break
                    summary.append(f"| {gen_id} | {gen['name']} | {gen['type']} | {gen['capacity_mw']} | {cost:,.0f} |\n")
            
            # New storage
            new_storage = implementation_plan['timeline'][year]['storage']['new']
            if new_storage:
                summary.append("\n#### New Storage Units\n")
                summary.append("| ID | Name | Power (MW) | Energy (MWh) | Cost |\n")
                summary.append("|-------|------|-----------|-------------|------|\n")
                for storage_id in new_storage:
                    storage = implementation_plan['storage_units'][str(storage_id)]
                    cost = 0
                    # Find the cost from the installation plan
                    for item in implementation_plan['installation_plan']:
                        if item['year'] == year and item['asset_type'] == 'storage' and item['asset_id'] == storage_id:
                            cost = item.get('cost', 0)
                            break
                    summary.append(f"| {storage_id} | {storage['name']} | {storage['capacity_mw']} | {storage['energy_capacity_mwh']} | {cost:,.0f} |\n")
            
            # Replaced storage
            replaced_storage = implementation_plan['timeline'][year]['storage']['replaced']
            if replaced_storage:
                summary.append("\n#### Replaced Storage Units\n")
                summary.append("| ID | Name | Power (MW) | Energy (MWh) | Cost |\n")
                summary.append("|-------|------|-----------|-------------|------|\n")
                for storage_id in replaced_storage:
                    storage = implementation_plan['storage_units'][str(storage_id)]
                    cost = 0
                    # Find the cost from the installation plan
                    for item in implementation_plan['installation_plan']:
                        if item['year'] == year and item['asset_type'] == 'storage' and item['asset_id'] == storage_id:
                            cost = item.get('cost', 0)
                            break
                    summary.append(f"| {storage_id} | {storage['name']} | {storage['capacity_mw']} | {storage['energy_capacity_mwh']} | {cost:,.0f} |\n")
            
            # Decommissioned assets
            decom_gens = implementation_plan['timeline'][year]['generators']['decommissioned']
            decom_storage = implementation_plan['timeline'][year]['storage']['decommissioned']
            
            if decom_gens or decom_storage:
                summary.append("\n#### Decommissioned Assets\n")
                summary.append("| Type | ID | Name | Reason |\n")
                summary.append("|------|-------|------|-------|\n")
                for gen_id in decom_gens:
                    gen = implementation_plan['generators'][str(gen_id)]
                    # Try to find the decommissioning reason
                    reason = "End of lifetime"
                    for item in implementation_plan['installation_plan']:
                        if item['year'] == year and item['asset_type'] == 'generator' and item['asset_id'] == gen_id and item['action'] == 'decommission':
                            reason = item.get('reason', "End of lifetime")
                            break
                    summary.append(f"| Generator | {gen_id} | {gen['name']} | {reason} |\n")
                for storage_id in decom_storage:
                    storage = implementation_plan['storage_units'][str(storage_id)]
                    # Try to find the decommissioning reason
                    reason = "End of lifetime"
                    for item in implementation_plan['installation_plan']:
                        if item['year'] == year and item['asset_type'] == 'storage' and item['asset_id'] == storage_id and item['action'] == 'decommission':
                            reason = item.get('reason', "End of lifetime")
                            break
                    summary.append(f"| Storage | {storage_id} | {storage['name']} | {reason} |\n")
        
        # Add summary of active assets at the end of planning horizon
        summary.append("\n## End of Planning Horizon\n")
        last_year = rel_years[-1]
        summary.append(f"\n### Active Assets in Year {last_year}\n")
        
        # Active generators
        active_gens = implementation_plan['timeline'][last_year]['generators']['active']
        if active_gens:
            summary.append("\n#### Generators\n")
            summary.append("| ID | Name | Type | Capacity (MW) | Installation Year(s) |\n")
            summary.append("|-------|------|------|-------------|--------------------|\n")
            for gen_id in active_gens:
                gen = implementation_plan['generators'][str(gen_id)]
                install_years = [str(history['year']) for history in gen['installation_history']]
                summary.append(f"| {gen_id} | {gen['name']} | {gen['type']} | {gen['capacity_mw']} | {', '.join(install_years)} |\n")
        
        # Active storage
        active_storage = implementation_plan['timeline'][last_year]['storage']['active']
        if active_storage:
            summary.append("\n#### Storage Units\n")
            summary.append("| ID | Name | Power (MW) | Energy (MWh) | Installation Year(s) |\n")
            summary.append("|-------|------|-----------|-------------|--------------------|\n")
            for storage_id in active_storage:
                storage = implementation_plan['storage_units'][str(storage_id)]
                install_years = [str(history['year']) for history in storage['installation_history']]
                summary.append(f"| {storage_id} | {storage['name']} | {storage['capacity_mw']} | {storage['energy_capacity_mwh']} | {', '.join(install_years)} |\n")
        
        # Save the summary to a markdown file
        with open(output_file, 'w') as f:
            f.writelines(summary)
        
        return True
        
    except Exception as e:
        print(f"Error generating plan summary: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_cost_plots(report, output_file):
    """
    Generate cost breakdown plots for a cost report
    
    Args:
        report: Cost report dictionary
        output_file: Path to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Season cost breakdown
        season_costs = []
        season_names = []
        
        for season, data in report['calculation'].items():
            season_costs.append(data['subtotal'])
            season_names.append(season.capitalize())
        
        ax1.bar(season_names, season_costs)
        ax1.set_title('Cost by Season')
        ax1.set_ylabel('Cost')
        ax1.set_ylim(bottom=0)
        
        # Add cost values on top of bars
        for i, cost in enumerate(season_costs):
            ax1.text(i, cost + (max(season_costs) * 0.01), f"{cost:.2f}", 
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Pie chart of relative costs
        ax2.pie(season_costs, labels=season_names, autopct='%1.1f%%')
        ax2.set_title('Relative Cost Distribution')
        
        # Add total cost as text
        fig.text(0.5, 0.01, f"Total Annual Cost: {report['annual_cost']:.2f}", ha='center', fontsize=12)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"Error generating cost plots: {e}")
        return False

def plot_generation_load_mismatch(network, output_file):
    """
    Create a visualization of generation vs load and mismatch across the planning horizon
    
    Args:
        network: Network object with multi-year results
        output_file: Path to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not hasattr(network, 'years') or len(network.years) == 0:
            print("Cannot generate mismatch plot: no multi-year data found.")
            return False
        
        years = network.years
        
        # Collect data
        generation = []
        load = []
        mismatch = []
        
        for year in years:
            # If the network has year summary data, use it
            if hasattr(network, 'year_summaries') and year in network.year_summaries:
                gen = network.year_summaries[year].get('total_generation', 0)
                ld = network.year_summaries[year].get('total_load', 0)
                
                generation.append(gen)
                load.append(ld)
                mismatch.append(gen - ld)
            # Otherwise calculate from time series if available
            elif hasattr(network, 'generators_t_by_year') and year in network.generators_t_by_year:
                # Sum up generation across all generators
                gen_total = sum(network.generators_t_by_year[year]['p'][gen_id].sum() 
                              for gen_id in network.generators.index 
                              if gen_id in network.generators_t_by_year[year]['p'])
                
                # Sum up load across all load components
                load_total = 0
                if hasattr(network, 'loads_t_by_year') and year in network.loads_t_by_year:
                    load_total = sum(network.loads_t_by_year[year]['p'][load_id].sum() 
                                   for load_id in network.loads.index 
                                   if load_id in network.loads_t_by_year[year]['p'])
                
                generation.append(gen_total)
                load.append(load_total)
                mismatch.append(gen_total - load_total)
            else:
                # Skip this year if we don't have data
                continue
        
        # Create the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: Generation vs Load
        ax1.bar(years, generation, label='Generation', alpha=0.7, color='green')
        ax1.bar(years, load, label='Load', alpha=0.7, color='blue')
        ax1.set_title('Generation vs Load by Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Energy (MWh)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Mismatch
        ax2.bar(years, mismatch, label='Mismatch (Gen-Load)', 
                color=['green' if m >= 0 else 'red' for m in mismatch])
        ax2.set_title('Generation-Load Mismatch by Year')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Energy Mismatch (MWh)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add the mismatch percentage
        for i, m in enumerate(mismatch):
            percentage = (m / load[i]) * 100 if load[i] > 0 else 0
            ax2.text(years[i], m + (max(mismatch) * 0.05 if m >= 0 else min(mismatch) * 0.05), 
                    f"{percentage:.1f}%", ha='center', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"Error generating generation-load mismatch plot: {e}")
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