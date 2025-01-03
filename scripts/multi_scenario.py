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
from utils.time_series import build_gen_time_series, build_demand_time_series
from typing import Dict, Any

# Paths
working_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/working"
results_root = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/results"

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

def ask_user_confirmation(message: str) -> bool:
    """Ask user for confirmation before proceeding"""
    while True:
        response = input(f"\n{message} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please answer with 'y' or 'n'")

def main():
    # Load Data
    bus = pd.read_csv(bus_file)
    branch = pd.read_csv(branch_file)
    branch.rename(columns={"rateA": "ratea"}, inplace=True, errors="ignore")
    branch["sus"] = 1 / branch["x"]
    branch["id"] = np.arange(1, len(branch) + 1)

    master_gen = pd.read_csv(master_gen_file, parse_dates=["time"]).sort_values("time")
    master_load = pd.read_csv(master_load_file, parse_dates=["time"]).sort_values("time")

    # Create mappings from master_gen.csv
    id_to_type = master_gen.drop_duplicates(subset=['id'])[['id', 'type']].set_index('id')['type'].to_dict()
    type_to_id = master_gen.drop_duplicates(subset=['type'])[['type', 'id']].set_index('type')['id'].to_dict()
    id_to_gencost = master_gen.drop_duplicates(subset=['id'])[['id', 'gencost']].set_index('id')['gencost'].to_dict()
    id_to_pmax = master_gen.drop_duplicates(subset=['id'])[['id', 'pmax']].set_index('id')['pmax'].to_dict()

    # Load Scenarios
    scenarios_df = pd.read_csv(scenarios_params_file)
    scenario_results = []

    # Initialize seasonal generation data
    seasonal_generation = {
        'winter': {},
        'summer': {},
        'autumn_spring': {}
    }

    for _, row in scenarios_df.iterrows():
        scenario_name = row["scenario_name"]
        gen_pos_raw = ast.literal_eval(row["gen_positions"])
        storage_pos_raw = ast.literal_eval(row["storage_units"])
        load_factor = float(row["load_factor"])

        # Initialize result entry with seasonal generation data
        result_entry = {
            "scenario_name": scenario_name,
            "annual_cost": None,
            "winter_gen": {},  # Initialize empty dictionaries for each season
            "summer_gen": {},
            "autumn_spring_gen": {}
        }

        # Map types to IDs
        try:
            gen_positions = {bus: type_to_id[gen_type] for bus, gen_type in gen_pos_raw.items()}
        except KeyError as e:
            print(f"Error: Generator type '{e.args[0]}' not found in type_to_id mapping.")
            scenario_results.append(result_entry)
            continue

        try:
            storage_positions = {bus: type_to_id[gen_type] for bus, gen_type in storage_pos_raw.items()}
        except KeyError as e:
            print(f"Error: Storage type '{e.args[0]}' not found in type_to_id mapping.")
            scenario_results.append(result_entry)
            continue

        print(f"\nProcessing {scenario_name} with Generators: {gen_pos_raw} and Storage: {storage_pos_raw}")

        season_costs = {}
        all_ok = True

        # Initialize accumulators for metrics
        total_gen_year = {}
        total_gen_cost_year = {}
        total_avail_gen_year = {}

        # Create scenario folder early
        scenario_folder = os.path.join(results_root, scenario_name)
        os.makedirs(scenario_folder, exist_ok=True)

        for season in ["winter", "summer", "autumn_spring"]:
            print(f"  Running {season}...")
            gen_ts = build_gen_time_series(master_gen, gen_positions, storage_positions, season)
            demand_ts = build_demand_time_series(master_load, load_factor, season)
            results = dcopf(gen_ts, branch, bus, demand_ts, delta_t=1)

            if results and results.get("status") == "Optimal":
                # Sum up total generation for each asset type in this season
                season_gen = {}
                for _, gen_row in results['generation'].iterrows():
                    gen_type = id_to_type.get(gen_row['id'])
                    if gen_type:
                        season_gen[gen_type] = season_gen.get(gen_type, 0) + gen_row['gen']
                
                # Store the seasonal generation data directly in result_entry
                result_entry[f"{season}_gen"] = season_gen
                season_costs[season] = results.get("cost", 0.0)
                print(f"    Optimal. Cost: {results['cost']}")
            else:
                print(f"    Not Optimal.")
                all_ok = False
                break

            # Plotting and metric extraction
            if all_ok:
                # Plotting and metric extraction
                plot_gen, plot_gen_cost, plot_avail, gen_vs_demand = plot_scenario_results(
                    results, 
                    demand_ts, 
                    branch, 
                    bus, 
                    scenario_folder=scenario_folder,  # Now always providing a valid path
                    season_key=season, 
                    id_to_type=id_to_type, 
                    id_to_gencost=id_to_gencost, 
                    id_to_pmax=id_to_pmax
                )

                # Accumulate metrics weighted by season
                weight = season_weights[season]
                for asset, gen in plot_gen.items():
                    total_gen_year[asset] = total_gen_year.get(asset, 0) + gen * weight
                for asset, cost in plot_gen_cost.items():
                    total_gen_cost_year[asset] = total_gen_cost_year.get(asset, 0) + cost * weight
                for asset, avail in plot_avail.items():
                    total_avail_gen_year[asset] = total_avail_gen_year.get(asset, 0) + avail * weight

        if all_ok:
            # Calculate Annual Cost
            annual_cost = sum(season_costs[season] * season_weights[season] for season in season_costs)
            result_entry["annual_cost"] = round(annual_cost, 1)

            # Add Total Generation per Asset
            for asset, gen in total_gen_year.items():
                result_entry[f"gen_{asset}"] = round(gen, 1)

            # Add Total Generation Cost per Asset
            for asset, cost in total_gen_cost_year.items():
                result_entry[f"gen_cost_{asset}"] = round(cost, 1)

            # Add Total Available Generation per Asset
            for asset, avail in total_avail_gen_year.items():
                result_entry[f"avail_gen_{asset}"] = round(avail, 1)

            # Optional: Add Capacity Factor per Asset
            capacity_factor_year = {}
            for asset, gen in total_gen_year.items():
                avail = total_avail_gen_year.get(asset, 0)
                if avail > 0:
                    capacity_factor_year[asset] = round(gen / avail, 2)
                else:
                    capacity_factor_year[asset] = 0.0
            for asset, cf in capacity_factor_year.items():
                result_entry[f"capacity_factor_{asset}"] = cf

            print(f"  Annual Cost: {round(annual_cost, 1)}")

        scenario_results.append(result_entry)

    # Save Results to CSV first
    results_df = pd.DataFrame(scenario_results)
    results_df.to_csv(os.path.join(results_root, "scenario_results.csv"), index=False)
    print("Initial results saved to CSV.")

    # Then perform investment analysis
    print("\nPerforming investment analysis...")
    analysis = InvestmentAnalysis()
    investment_results = analysis.analyze_scenario(
        os.path.join(results_root, "scenario_results.csv"),
        master_gen_file
    )
    
    # Ensure we have the right column names from investment analysis
    print("Investment analysis columns:", investment_results.columns.tolist())
    
    # Merge investment results with scenario results using scenario name
    results_df = results_df.merge(
        investment_results,
        left_on='scenario_name',
        right_index=True,
        how='left'
    )
    
    # Ensure we have the expected columns
    expected_columns = ['npv', 'annuity', 'initial_investment']
    actual_columns = results_df.columns.tolist()
    print("\nAvailable columns after merge:", actual_columns)
    
    # Rename columns if necessary
    column_mapping = {
        'NPV': 'npv_30y',
        'Annuity': 'annuity_30y',
        'Initial Investment': 'initial_investment',
        'annual_costs': 'annual_costs'
    }
    
    results_df = results_df.rename(columns=column_mapping)
    
    # Convert NPV and cost columns to numeric
    numeric_columns = [
        'npv_10y', 'npv_20y', 'npv_30y',
        'annuity_10y', 'annuity_20y', 'annuity_30y',
        'initial_investment', 'annual_cost', 'annual_costs'
    ]
    
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Sort by 30-year NPV for ranking
    if 'npv_30y' in results_df.columns:
        results_df = results_df.sort_values('npv_30y', ascending=False)
    else:
        print("Warning: NPV column not found in results")

    # Ask user for generation preferences
    generate_plots = ask_user_confirmation("Do you want to generate plots?")
    generate_individual = ask_user_confirmation("Do you want to generate individual scenario reports?")
    generate_global = ask_user_confirmation("Do you want to generate a global comparison report?")

    if generate_plots:
        print("\nGenerating plots...")
        for _, row in results_df.iterrows():
            scenario_data = row.to_dict()
            print(f"\nProcessing scenario: {scenario_data['scenario_name']}")
            print(f"Available data keys: {list(scenario_data.keys())}")
            create_annual_summary_plots(scenario_data, results_root)
            create_scenario_comparison_plot(scenario_data, results_root)
        print("Plot generation completed.")

    if generate_individual or generate_global:
        print("\nGenerating requested reports...")
        
        # Generate individual reports if requested
        if generate_individual:
            print("\nGenerating individual scenario reports...")
            for _, row in results_df.iterrows():
                if row['annual_cost'] is not None:
                    critic.analyze_scenario(row.to_dict(), results_root)
            print("Individual reports completed.")
        
        # Generate global report if requested
        if generate_global:
            print("\nGenerating global comparison report...")
            critic.create_global_comparison_report(results_df, results_root)
            print("Global report completed.")
        
        print("All requested reports generated.")
    else:
        print("\nSkipping report generation.")

    # Save the final results CSV with investment analysis
    results_df.to_csv(os.path.join(results_root, "scenario_results_with_investment.csv"), index=False)
    print("Final results saved with investment analysis.")
    
    # Update README with scenario links
    project_root = get_project_root()
    readme_path = os.path.join(project_root, 'README.md')
    create_readme_template(readme_path)  # Create/update the full README
    update_readme_with_scenarios()       # Update the scenario links

if __name__ == "__main__":
    main()
