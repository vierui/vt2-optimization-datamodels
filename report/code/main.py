#!/usr/bin/env python3

"""
multi_scenario.py

- Loads scenarios from scenarios_parameters.csv
- Runs DCOPF for each scenario across winter, summer, autumn_spring
- Saves results and plots in /data/results/<scenario_name>/
- Summarizes costs in scenario_results.csv
"""

import os
import sys

# Add the scripts directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import ast

from dcopf import dcopf
from dotenv import load_dotenv
from scenario_critic import ScenarioCritic
from update_readme import update_readme_with_scenarios, create_readme_template, get_project_root
from create_master_invest import InvestmentAnalysis
from visualization.summary_plots import create_annual_summary_plots, create_scenario_comparison_plot
from visualization.scenario_plots import plot_scenario_results
from core.time_series import build_gen_time_series, build_demand_time_series
from core.helpers import ask_user_confirmation
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from visualization.report_plots import create_scenario_plots

# Paths
project_root = get_project_root()
working_dir = os.path.join(project_root, "data", "working")
results_root = os.path.join(project_root, "data", "results")

bus_file = os.path.join(working_dir, "bus.csv")
branch_file = os.path.join(working_dir, "branch.csv")
master_gen_file = os.path.join(working_dir, "master_gen.csv")
master_load_file = os.path.join(working_dir, "master_load.csv")
scenarios_params_file = os.path.join(working_dir, "scenarios_parameters.csv")

# Season weights
season_weights = {
    "winter": 13,
    "summer": 13,
    "autumn_spring": 26
}

# Load environment variables
load_dotenv('../.env.local')
api_key = os.getenv('OPENAPI_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in .env.local file")

# Initialize critic
critic = ScenarioCritic(api_key)

# Data classes
@dataclass
class SeasonalData:
    generation: Dict[str, float]  # Asset type -> generation amount
    cost: float
    capacity_factors: Dict[str, float]

@dataclass
class ScenarioVariant:
    scenario_name: str
    variant_type: str  # 'nominal', 'high', 'low'
    load_factor: float
    annual_cost: float
    seasonal_data: Dict[str, SeasonalData]  # season -> data
    generation_by_asset: Dict[str, float]
    generation_costs: Dict[str, float]
    available_capacity: Dict[str, float]
    capacity_factors: Dict[str, float]
    
    @property
    def full_name(self) -> str:
        return f"{self.scenario_name}_{self.variant_type}"
    
    def to_dict(self) -> Dict: # Convert to flat dictionary for DataFrame
        """Convert to flat dictionary for DataFrame"""
        result = {
            "scenario_name": self.full_name,
            "base_scenario": self.scenario_name,
            "variant": self.variant_type,
            "load_factor": self.load_factor,
            "annual_cost": self.annual_cost
        }
        
        # Add seasonal data
        for season, data in self.seasonal_data.items():
            for asset, gen in data.generation.items():
                result[f"{season}_gen_{asset}"] = gen
            result[f"{season}_cost"] = data.cost
        
        # Add annual metrics
        for asset, gen in self.generation_by_asset.items():
            result[f"gen_{asset}"] = gen
            result[f"gen_cost_{asset}"] = self.generation_costs.get(asset, 0)
            result[f"avail_gen_{asset}"] = self.available_capacity.get(asset, 0)
            result[f"capacity_factor_{asset}"] = self.capacity_factors.get(asset, 0)
            
        return result

# Run a single scenario variant (nominal, high, or low load)
def run_scenario_variant(
    scenario_name: str,
    gen_positions: Dict[int, int],
    storage_positions: Dict[int, int],
    load_factor: float,
    variant: str,
    data_context: Dict[str, Any]
) -> Optional[ScenarioVariant]:
    """Run a single scenario variant (nominal, high, or low load)"""
    
    seasonal_data = {}
    total_gen_year = {}
    total_gen_cost_year = {}
    total_avail_gen_year = {}
    
    print(f"\nProcessing {scenario_name} ({variant} load) with:")
    print(f"  Load factor: {load_factor}")

    for season in ["winter", "summer", "autumn_spring"]:
        print(f"  Running {season}...")
        season_result = run_single_season(
            season=season,
            gen_positions=gen_positions,
            storage_positions=storage_positions,
            load_factor=load_factor,
            data_context=data_context
        )
        
        if season_result is None:
            return None
            
        # Convert SeasonResult to SeasonalData
        generation_dict = {
            asset: metrics.generation 
            for asset, metrics in season_result.metrics.items()
        }
        capacity_factors = {
            asset: (metrics.generation / metrics.available if metrics.available > 0 else 0)
            for asset, metrics in season_result.metrics.items()
        }
        
        seasonal_data[season] = SeasonalData(
            generation=generation_dict,
            cost=season_result.cost,
            capacity_factors=capacity_factors
        )
        
        # Accumulate annual metrics
        weight = data_context['season_weights'][season]
        for asset, metrics in season_result.metrics.items():
            total_gen_year[asset] = total_gen_year.get(asset, 0) + metrics.generation * weight
            total_gen_cost_year[asset] = total_gen_cost_year.get(asset, 0) + metrics.cost * weight
            total_avail_gen_year[asset] = total_avail_gen_year.get(asset, 0) + metrics.available * weight

    # Calculate capacity factors
    capacity_factors = {
        asset: total_gen_year[asset] / total_avail_gen_year[asset]
        if total_avail_gen_year.get(asset, 0) > 0 else 0
        for asset in total_gen_year
    }

    annual_cost = sum(
        data.cost * data_context['season_weights'][season] 
        for season, data in seasonal_data.items()
    )

    return ScenarioVariant(
        scenario_name=scenario_name,
        variant_type=variant,
        load_factor=load_factor,
        annual_cost=annual_cost,
        seasonal_data=seasonal_data,
        generation_by_asset=total_gen_year,
        generation_costs=total_gen_cost_year,
        available_capacity=total_avail_gen_year,
        capacity_factors=capacity_factors
    )


def load_data_context() -> Dict[str, Any]:
    """
    Load and preprocess all required data for scenario analysis.
    Returns a context dictionary containing all necessary data and mappings.
    """
    # Load base data files
    bus = pd.read_csv(bus_file)
    branch = pd.read_csv(branch_file)
    master_gen = pd.read_csv(master_gen_file, parse_dates=["time"]).sort_values("time")
    master_load = pd.read_csv(master_load_file, parse_dates=["time"]).sort_values("time")
    scenarios_df = pd.read_csv(scenarios_params_file)

    # Process branch data
    branch.rename(columns={"rateA": "ratea"}, inplace=True, errors="ignore")
    branch["sus"] = 1 / branch["x"]
    branch["id"] = np.arange(1, len(branch) + 1)

    # Create mappings
    id_to_type = master_gen.drop_duplicates(subset=['id'])[['id', 'type']].set_index('id')['type'].to_dict()
    type_to_id = master_gen.drop_duplicates(subset=['type'])[['type', 'id']].set_index('type')['id'].to_dict()
    id_to_gencost = master_gen.drop_duplicates(subset=['id'])[['id', 'gencost']].set_index('id')['gencost'].to_dict()
    id_to_pmax = master_gen.drop_duplicates(subset=['id'])[['id', 'pmax']].set_index('id')['pmax'].to_dict()

    return {
        # Raw data
        'bus': bus,
        'branch': branch,
        'master_gen': master_gen,
        'master_load': master_load,
        'scenarios_df': scenarios_df,
        
        # Mappings
        'id_to_type': id_to_type,
        'type_to_id': type_to_id,
        'id_to_gencost': id_to_gencost,
        'id_to_pmax': id_to_pmax,
        
        # Constants
        'season_weights': season_weights,
        
        # Paths
        'results_root': results_root
    }

def parse_positions(positions_str: str, type_to_id: Dict[str, int]) -> Dict[int, int]:
    """
    Parse positions string from scenarios file and convert types to IDs.
    
    Args:
        positions_str: String representation of positions dictionary
        type_to_id: Mapping from generator type to ID
    
    Returns:
        Dictionary mapping bus numbers to generator IDs
    """
    try:
        positions_raw = ast.literal_eval(positions_str)
        return {
            int(bus): type_to_id[gen_type]
            for bus, gen_type in positions_raw.items()
        }
    except (ValueError, KeyError) as e:
        print(f"Error parsing positions: {e}")
        return {}

@dataclass
class SeasonMetrics:
    generation: float
    cost: float
    available: float

@dataclass
class SeasonResult:
    metrics: Dict[str, SeasonMetrics]
    cost: float
    storage_data: Optional[pd.DataFrame] = None

def run_single_season(
    season: str,
    gen_positions: Dict[int, int],
    storage_positions: Dict[int, int],
    load_factor: float,
    data_context: Dict[str, Any]
) -> Optional[SeasonResult]:
    """Run DCOPF for a single season and collect results"""
    # Build time series
    gen_ts = build_gen_time_series(
        data_context['master_gen'], 
        gen_positions, 
        storage_positions,
        season
    )
    
    # Build demand time series
    demand_ts = build_demand_time_series(
        data_context['master_load'],
        load_factor,
        season
    )
    
    # Print debug info
    print(f"\nAssets in {season}:")
    print("Generators:", gen_positions)
    print("Storage:", storage_positions)
    print("Types in time series:", gen_ts['type'].unique())
    print(f"Load factor: {load_factor}")
    
    # Run DCOPF
    results = dcopf(
        gen_ts, 
        data_context['branch'], 
        data_context['bus'], 
        demand_ts, 
        delta_t=1
    )
    
    # Debug prints to check DCOPF results structure
    print("\nDCOPF Results Structure:")
    print("Available keys:", results.keys())
    if 'storage' in results:
        print("\nStorage data found!")
        print("Storage data columns:", results['storage'].columns.tolist())
        print("First few rows of storage data:")
        print(results['storage'].head())
        print("\nStorage data shape:", results['storage'].shape)
        print("Storage data types:", results['storage'].dtypes)
    else:
        print("\nNo storage data in DCOPF results")
    
    if not results or results.get("status") != "Optimal":
        print(f"Failed to find optimal solution for {season}")
        return None
    
    # Process generation metrics
    metrics_by_type = {}
    
    # Group generation by type
    for _, gen_row in results['generation'].iterrows():
        gen_type = data_context['id_to_type'].get(gen_row['id'])
        if gen_type:
            if gen_type not in metrics_by_type:
                metrics_by_type[gen_type] = SeasonMetrics(
                    generation=0,
                    cost=0,
                    available=0
                )
            
            # Add generation
            metrics_by_type[gen_type].generation += gen_row['gen']
            
            # Add cost
            if gen_row['id'] in data_context['id_to_gencost']:
                cost = gen_row['gen'] * data_context['id_to_gencost'][gen_row['id']]
                metrics_by_type[gen_type].cost += cost
            
            # Calculate available capacity
            if gen_row['id'] in data_context['id_to_pmax']:
                metrics_by_type[gen_type].available += data_context['id_to_pmax'][gen_row['id']]
    
    # Extract storage data if available
    storage_data = None
    if 'storage' in results:
        storage_data = results['storage'].copy()
        # Use 'E' column as Storage_SoC (as in your PULP code)
        if 'E' in storage_data.columns:
            storage_data['Storage_SoC'] = storage_data['E']
            storage_data.set_index('time', inplace=True)
    
    return SeasonResult(
        metrics=metrics_by_type,
        cost=results.get("cost", 0.0),
        storage_data=storage_data
    )

class MultiScenario:
    """Main class to handle multiple scenario analysis for power system investments"""
    
    #######################
    # 1. INITIALIZATION
    #######################
    def __init__(self, plot_gen_mix=False):
        """Initialize parameters, paths, and configurations"""
        # Setup paths
        self.setup_paths()
        
        # Load scenario parameters
        self.load_scenario_parameters()
        
        # Initialize analysis parameters
        self.init_analysis_parameters()
        
        # Setup output directories
        self.create_output_dirs()
    
    #######################
    # 2. NETWORK CREATION
    #######################
    def create_network(self):
        """Create and configure PyPSA network for scenarios"""
        # Load component data (generators, buses, branches)
        self.load_component_data()
        
        # Configure network parameters
        self.setup_network_parameters()
        
        # Add components to network
        self.add_network_components()
        
        return network

    #######################
    # 3. SCENARIO SOLVING
    #######################
    def solve(self):
        """Main solving function for all scenarios"""
        # Iterate through scenarios
        for scenario in self.scenarios:
            # Create network for scenario
            network = self.create_network()
            
            # Run OPF
            self.run_opf(network)
            
            # Store results
            self.store_scenario_results(network)
            
            # Calculate metrics
            self.calculate_scenario_metrics()
    
    #######################
    # 4. METRICS CALCULATION
    #######################
    def calculate_metrics(self):
        """Calculate investment and performance metrics"""
        # Financial calculations
        self.calculate_financial_metrics()
        
        # Sensitivity analysis
        self.perform_sensitivity_analysis()
        
        # Store metric results
        self.store_metrics()
    
    #######################
    # 5. VISUALIZATION
    #######################
    def generate_plots(self):
        """Generate all required plots"""
        # Generation mix plots
        self.plot_generation_mix()
        
        # Investment metric plots
        self.plot_investment_metrics()
        
        # Sensitivity analysis plots
        self.plot_sensitivity_results()
        
        # Add AI comments to plots
        self.add_plot_comments()
    
    #######################
    # 6. REPORTING
    #######################
    def create_summary(self):
        """Create summary reports and analysis"""
        # Generate summary statistics
        self.calculate_summary_stats()
        
        # Create summary file
        self.write_summary_file()
        
        # Generate AI analysis
        self.generate_ai_analysis()
    
    #######################
    # 7. UTILITY FUNCTIONS
    #######################
    def setup_paths(self):
        """Setup directory paths"""
        pass
    
    def load_scenario_parameters(self):
        """Load and validate scenario parameters"""
        pass
    
    def store_scenario_results(self, network):
        """Store results for a specific scenario"""
        pass

def main():
    # Load all data
    data_context = load_data_context()
    
    # Ask for sensitivity analysis
    run_sensitivity = ask_user_confirmation(
        "Do you want to run sensitivity analysis ?"
    )

    scenario_variants: List[ScenarioVariant] = []
    
    # Dictionary to collect storage data from all scenarios
    all_scenarios_storage = {}
    
    for _, row in data_context['scenarios_df'].iterrows():
        scenario_name = row["scenario_name"]
        gen_positions = parse_positions(row["gen_positions"], data_context['type_to_id'])
        storage_positions = parse_positions(row["storage_units"], data_context['type_to_id'])
        base_load_factor = float(row["load_factor"])

        # Storage data collection for nominal load only
        storage_data = pd.DataFrame()
        
        # Run variants
        variants_to_run = [("nominal", base_load_factor)]
        if run_sensitivity:
            variants_to_run.extend([
                ("high", base_load_factor * 1.2),
                ("low", base_load_factor * 0.8)
            ])
        
        for variant_name, load_factor in variants_to_run:
            # Run scenario and collect results
            for season in ["winter", "summer", "autumn_spring"]:
                season_result = run_single_season(
                    season=season,
                    gen_positions=gen_positions,
                    storage_positions=storage_positions,
                    load_factor=load_factor,
                    data_context=data_context
                )
                
                # Collect storage data for nominal load case
                if variant_name == "nominal" and season_result and season_result.storage_data is not None:
                    storage_data = pd.concat([storage_data, season_result.storage_data])
            
            result = run_scenario_variant(
                scenario_name=scenario_name,
                gen_positions=gen_positions,
                storage_positions=storage_positions,
                load_factor=load_factor,
                variant=variant_name,
                data_context=data_context
            )
            if result:
                scenario_variants.append(result)
        
        # Store storage data if available
        if not storage_data.empty:
            storage_data = storage_data.sort_index()
            all_scenarios_storage[scenario_name] = storage_data
            
            # Create plots with this scenario's data
            if 'Storage_SoC' in storage_data.columns:
                create_scenario_plots({scenario_name: storage_data})
                print(f"Created storage plots for scenario {scenario_name}")
            else:
                print(f"Warning: No Storage_SoC column found in scenario {scenario_name}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        variant.to_dict() for variant in scenario_variants
    ])

    # Save initial results
    results_df.to_csv(os.path.join(results_root, "scenario_results.csv"), index=False)
    print("Initial results saved to CSV.")

    # Then perform investment analysis
    print("\nPerforming investment analysis...")
    analysis = InvestmentAnalysis()
    investment_results = analysis.analyze_scenario(
        os.path.join(results_root, "scenario_results.csv"),
        master_gen_file
    )
    
    print("Investment analysis columns:", investment_results.columns.tolist())

    # Check if base_scenario is in the index
    if 'base_scenario' in investment_results.index.names:
        # Reset index only if base_scenario is not already a column
        if 'base_scenario' not in investment_results.columns:
            investment_results = investment_results.reset_index()
    
    # Filter for nominal variants
    nominal_results = results_df[results_df['variant'] == 'nominal'].copy()

    # Get the actual columns that exist in the DataFrame
    available_columns = nominal_results.columns.tolist()
    print("\nAvailable columns in nominal_results:", available_columns)

    # Define essential columns based on what's available
    base_essential_columns = [
        'base_scenario',
        'variant',
        'load_factor',
        'annual_cost'
    ]

    # Add generation columns that exist
    gen_columns = [col for col in available_columns if col.startswith('gen_')]
    essential_columns = base_essential_columns + gen_columns

    print("\nSelected essential columns:", essential_columns)

    # Filter columns
    nominal_results = nominal_results[essential_columns]

    # Merge the results
    final_results = nominal_results.merge(
        investment_results,
        on='base_scenario',
        how='left'
    )

    # Add a clean scenario identifier
    final_results['scenario_id'] = final_results.apply(
        lambda x: f"{x['base_scenario']}_{x['variant']}", axis=1
    )

    # Define base columns that we want first
    base_columns = [
        'scenario_id',
        'base_scenario',
        'variant',
        'load_factor'
    ]

    # Define investment-related columns
    investment_columns = [
        'installed_capacity',
        'initial_investment',
        'annual_cost',
        'annual_costs'
    ]

    # Define NPV and annuity columns
    financial_columns = [
        'npv_10y', 'npv_20y', 'npv_30y',
        'annuity_10y', 'annuity_20y', 'annuity_30y'
    ]

    # Get generation-related columns
    generation_columns = [col for col in final_results.columns 
                         if col.startswith('gen_') or 
                            col.startswith('winter_') or 
                            col.startswith('summer_') or 
                            col.startswith('autumn_')]

    # Combine all columns in desired order
    column_order = (base_columns + 
                    investment_columns + 
                    financial_columns + 
                    generation_columns)

    # Add any remaining columns that we haven't explicitly ordered
    remaining_columns = [col for col in final_results.columns 
                        if col not in column_order]
    column_order.extend(remaining_columns)

    # Reorder columns, but only include ones that exist
    final_columns = [col for col in column_order 
                     if col in final_results.columns]
    final_results = final_results[final_columns]

    # Save the final results
    final_results.to_csv(os.path.join(results_root, "scenario_results_with_investment.csv"), 
                         index=False)

    # Ask user for generation preferences
    generate_plots = ask_user_confirmation("Do you want to generate plots?")
    generate_individual = ask_user_confirmation("Do you want to generate individual scenario reports?")
    generate_global = ask_user_confirmation("Do you want to generate a global comparison report?")

    if generate_plots:
        print("\nGenerating plots...")
        # Group scenarios by base scenario
        scenario_groups = final_results.groupby('base_scenario')
        
        for base_scenario, group in scenario_groups:
            print(f"\nProcessing scenario: {base_scenario}")
            
            # Get variants with debug printing
            nominal_data = group[group['variant'] == 'nominal'].iloc[0].to_dict()
            
            # Get high variant
            high_data = {}
            high_variant = group[group['variant'] == 'high']
            if not high_variant.empty:
                high_data = high_variant.iloc[0].to_dict()
                print(f"Found high variant for {base_scenario}")
            else:
                print(f"No high variant for {base_scenario}")
                
            # Get low variant
            low_data = {}
            low_variant = group[group['variant'] == 'low']
            if not low_variant.empty:
                low_data = low_variant.iloc[0].to_dict()
                print(f"Found low variant for {base_scenario}")
            else:
                print(f"No low variant for {base_scenario}")
            
            # Add sensitivity data to nominal data
            nominal_data['high_variant'] = high_data
            nominal_data['low_variant'] = low_data
            
            # Create plots
            create_annual_summary_plots(nominal_data, results_root)

    if generate_individual or generate_global:
        print("\nGenerating requested reports...")
        
        # Generate individual reports if requested
        if generate_individual:
            print("\nGenerating individual scenario reports...")
            # Only process nominal variants for reports
            nominal_results = final_results[final_results['variant'] == 'nominal']
            for _, row in nominal_results.iterrows():
                if row['annual_cost'] is not None:
                    critic.analyze_scenario(row.to_dict(), results_root)
            print("Individual reports completed.")
        
        # Generate global report if requested
        if generate_global:
            print("\nGenerating global comparison report...")
            # Use only nominal variants for global comparison
            nominal_results = final_results[final_results['variant'] == 'nominal']
            critic.create_global_comparison_report(nominal_results, results_root)
            print("Global report completed.")
        
        print("All requested reports generated.")
    else:
        print("\nSkipping report generation.")

    # Update README with scenario links
    project_root = get_project_root()
    readme_path = os.path.join(project_root, 'README.md')
    create_readme_template(readme_path)  # Create/update the full README
    update_readme_with_scenarios()       # Update the scenario links

if __name__ == "__main__":
    main()
