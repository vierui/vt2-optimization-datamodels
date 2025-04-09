#!/usr/bin/env python3
"""
Post-processing module for calculating annual costs and generating implementation plans
from the integrated optimization model
"""
import os
import json
import pandas as pd
from datetime import datetime
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
    annual_cost = 0
    for season, cost in season_costs.items():
        weeks = SEASON_WEIGHTS.get(season, 0)
        annual_cost += weeks * cost
    
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

def generate_implementation_plan(integrated_network):
    """
    Generate an implementation plan based on integrated optimization results
    
    This creates a year-by-year plan of asset deployments that ensures
    consistency across all seasonal models.
    
    Args:
        integrated_network: IntegratedNetwork object with optimization results
        
    Returns:
        Implementation plan dictionary
    """
    if not hasattr(integrated_network, 'asset_installation'):
        print("Error: No asset installation data available in the integrated network.")
        return {}
    
    try:
        # Extract years and installation decisions
        years = integrated_network.years
        asset_installation = integrated_network.asset_installation
        
        # Use the first network to get component data
        first_network = list(integrated_network.season_networks.values())[0]
        
        implementation_plan = {
            'years': list(years),
            'generators': {},
            'storage': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Process generators
        for year in years:
            for gen_id, installed in asset_installation['generators'][year].items():
                if installed > 0.5:  # Binary decision variable > 0.5 means installed
                    if gen_id not in implementation_plan['generators']:
                        # Get generator data from the network
                        if gen_id in first_network.generators.index:
                            gen_data = first_network.generators.loc[gen_id]
                            implementation_plan['generators'][gen_id] = {
                                'id': gen_id,
                                'name': gen_data.get('name', f"Generator_{gen_id}"),
                                'capacity_mw': float(gen_data.get('p_nom', 0)),
                                'cost_mwh': float(gen_data.get('marginal_cost', 0)),
                                'type': gen_data.get('type', 'unknown'),
                                'years_installed': [],
                                'capex_per_mw': float(gen_data.get('capex_per_mw', 0)),
                                'lifetime_years': float(gen_data.get('lifetime_years', 25))
                            }
                        else:
                            # Minimal data if generator not found in network
                            implementation_plan['generators'][gen_id] = {
                                'id': gen_id,
                                'name': f"Generator_{gen_id}",
                                'years_installed': []
                            }
                    
                    implementation_plan['generators'][gen_id]['years_installed'].append(year)
        
        # Process storage units
        for year in years:
            for storage_id, installed in asset_installation['storage'][year].items():
                if installed > 0.5:  # Binary decision variable > 0.5 means installed
                    if storage_id not in implementation_plan['storage']:
                        # Get storage data from the network
                        if storage_id in first_network.storage_units.index:
                            storage_data = first_network.storage_units.loc[storage_id]
                            implementation_plan['storage'][storage_id] = {
                                'id': storage_id,
                                'name': storage_data.get('name', f"Storage_{storage_id}"),
                                'capacity_mw': float(storage_data.get('p_nom', 0)),
                                'energy_mwh': float(storage_data.get('energy_mwh', 0)),
                                'years_installed': [],
                                'capex_per_mw': float(storage_data.get('capex_per_mw', 0)),
                                'lifetime_years': float(storage_data.get('lifetime_years', 15))
                            }
                        else:
                            # Minimal data if storage not found in network
                            implementation_plan['storage'][storage_id] = {
                                'id': storage_id,
                                'name': f"Storage_{storage_id}",
                                'years_installed': []
                            }
                    
                    implementation_plan['storage'][storage_id]['years_installed'].append(year)
        
        # Add first installation info if available
        if hasattr(integrated_network, 'asset_installation_history'):
            # Process generator installation history
            for gen_id, history in integrated_network.asset_installation_history.get('generators', {}).items():
                if gen_id in implementation_plan['generators']:
                    implementation_plan['generators'][gen_id]['installation_history'] = history
            
            # Process storage installation history
            for storage_id, history in integrated_network.asset_installation_history.get('storage', {}).items():
                if storage_id in implementation_plan['storage']:
                    implementation_plan['storage'][storage_id]['installation_history'] = history
        
        # Add asset utilization by season
        implementation_plan['seasonal_utilization'] = {
            'generators': {},
            'storage': {}
        }
        
        for season, network in integrated_network.season_networks.items():
            # Skip if no results available
            if not hasattr(network, 'generators_t_by_year'):
                continue
            
            # Last year (for simplicity)
            last_year = years[-1]
            
            # Generator utilization
            for gen_id in network.generators.index:
                # Skip if not installed in the final year
                if gen_id not in implementation_plan['generators']:
                    continue
                
                # Calculate utilization (capacity factor)
                if 'p' in network.generators_t_by_year.get(last_year, {}) and gen_id in network.generators_t_by_year[last_year]['p']:
                    gen_output = network.generators_t_by_year[last_year]['p'][gen_id].sum()
                    capacity = network.generators.loc[gen_id].get('p_nom', 0)
                    hours = len(network.generators_t_by_year[last_year]['p'])
                    
                    # Capacity factor = actual output / potential output
                    capacity_factor = gen_output / (capacity * hours) if capacity * hours > 0 else 0
                    
                    # Store in implementation plan
                    if gen_id not in implementation_plan['seasonal_utilization']['generators']:
                        implementation_plan['seasonal_utilization']['generators'][gen_id] = {}
                    
                    implementation_plan['seasonal_utilization']['generators'][gen_id][season] = float(capacity_factor)
            
            # Storage utilization
        for storage_id in network.storage_units.index:
                # Skip if not installed in the final year
                if storage_id not in implementation_plan['storage']:
                    continue
                
                # Calculate utilization metrics
                if 'p_charge' in network.storage_units_t_by_year.get(last_year, {}) and storage_id in network.storage_units_t_by_year[last_year]['p_charge']:
                    charge_sum = network.storage_units_t_by_year[last_year]['p_charge'][storage_id].sum()
                    discharge_sum = network.storage_units_t_by_year[last_year]['p_discharge'][storage_id].sum()
                    capacity = network.storage_units.loc[storage_id].get('p_nom', 0)
                    hours = len(network.storage_units_t_by_year[last_year]['p_charge'])
                    
                    # Cycling rate = sum of charge / storage capacity
                    cycling_rate = (charge_sum + discharge_sum) / (2 * capacity * hours) if capacity * hours > 0 else 0
                    
                    # Store in implementation plan
                    if storage_id not in implementation_plan['seasonal_utilization']['storage']:
                        implementation_plan['seasonal_utilization']['storage'][storage_id] = {}
                    
                    implementation_plan['seasonal_utilization']['storage'][storage_id][season] = float(cycling_rate)
        
        # Add seasonal cost breakdown
        implementation_plan['seasonal_costs'] = {}
        if hasattr(integrated_network, 'seasons_total_cost'):
            implementation_plan['seasonal_costs'] = {
                season: float(cost) for season, cost in integrated_network.seasons_total_cost.items()
            }
        
        # Add total annual cost
        annual_cost = 0
        if hasattr(integrated_network, 'seasons_total_cost'):
            for season, cost in integrated_network.seasons_total_cost.items():
                weeks = integrated_network.season_weights.get(season, 0)
                annual_cost += cost * weeks
        
        implementation_plan['annual_cost'] = float(annual_cost)
        
        return implementation_plan
        
    except Exception as e:
        print(f"Error generating implementation plan: {e}")
        import traceback
        traceback.print_exc()
        return {}

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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Annual cost calculation from integrated model results')
    parser.add_argument('--network-file', type=str, required=True,
                      help='Path to integrated network pickle file')
    parser.add_argument('--output-dir', type=str, default='results/annual',
                      help='Directory to store the annual cost report')
    
    args = parser.parse_args()
    
    # Check if network file exists
    if not os.path.exists(args.network_file):
        print(f"Error: Network file not found: {args.network_file}")
        import sys
        sys.exit(1)
    
    # Load the integrated network
    from network import IntegratedNetwork
    
    try:
        integrated_network = IntegratedNetwork.load_from_pickle(args.network_file)
        
        if not integrated_network:
            print("Error loading integrated network")
            import sys
            sys.exit(1)
        
        # Generate implementation plan
        plan_file = os.path.join(args.output_dir, "implementation_plan.json")
        generate_implementation_plan(integrated_network)
        
        # Generate annual cost report
        report_file = os.path.join(args.output_dir, "annual_cost_report.json")
        
        if hasattr(integrated_network, 'seasons_total_cost'):
            save_annual_cost_report(integrated_network.seasons_total_cost, report_file)
            print(f"Annual cost report saved to: {report_file}")
        else:
            print("Error: No season costs found in the integrated network")
            import sys
            sys.exit(1)
        
    except Exception as e:
        print(f"Error processing integrated network: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1) 