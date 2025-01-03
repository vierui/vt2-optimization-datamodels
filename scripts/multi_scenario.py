#!/usr/bin/env python3

"""
multi_scenario.py

- Loads scenarios from scenarios_parameters.csv
- Runs DCOPF for each scenario across winter, summer, autumn_spring
- Saves results and plots in /data/results/<scenario_name>/
- Summarizes costs in scenario_results.csv
"""
# %%
import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
import networkx as nx
from dcopf import dcopf  # Ensure this imports your updated DCOPF code
from dotenv import load_dotenv
from scenario_critic import ScenarioCritic
from update_readme import update_readme_with_scenarios, create_readme_template, get_project_root
from create_master_invest import InvestmentAnalysis
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

# %%
def ask_user_confirmation(message: str) -> bool:
    """Ask user for confirmation before proceeding"""
    while True:
        response = input(f"\n{message} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please answer with 'y' or 'n'")

# %%
def build_gen_time_series(master_gen, gen_positions_dict, storage_positions_dict, season_key):
    """Build generation time series for the scenario."""
    scenario_gen = master_gen[master_gen["season"] == season_key].copy()
    merged_positions = {**gen_positions_dict, **storage_positions_dict}
    selected_ids = list(merged_positions.values())
    scenario_gen = scenario_gen[scenario_gen["id"].isin(selected_ids)].copy()
    
    for bus_i, gen_id in merged_positions.items():
        scenario_gen.loc[scenario_gen["id"] == gen_id, "bus"] = bus_i
    
    scenario_gen.drop_duplicates(subset=['time', 'id'], inplace=True)
    scenario_gen.sort_values(["time", "id"], inplace=True)
    scenario_gen.reset_index(drop=True, inplace=True)
    return scenario_gen

# %%
def build_demand_time_series(master_load, load_factor, season_key):
    """Build demand time series for the scenario."""
    scenario_load = master_load[master_load["season"] == season_key].copy()
    scenario_load["pd"] *= load_factor
    scenario_load.drop_duplicates(subset=['time', 'bus'], inplace=True)
    scenario_load.sort_values(["time", "bus"], inplace=True)
    scenario_load.reset_index(drop=True, inplace=True)
    return scenario_load

# %%
def plot_scenario_results(results, demand_time_series, branch, bus, scenario_folder, season_key, id_to_type, id_to_gencost, id_to_pmax):
    """Plot and analyze scenario results"""
    # Get unique time points
    time_points = sorted(demand_time_series['time'].unique())
    
    # Create scenario folder
    os.makedirs(scenario_folder, exist_ok=True)
    
    # Initialize data structures
    gen_by_type = [{} for _ in time_points]
    total_demand = np.zeros(len(time_points))
    total_gen_per_asset = {}
    total_gen_cost_per_asset = {}
    remaining_capacity_series = {}
    
    # Process generation and demand data
    for t, time in enumerate(time_points):
        # Get demand for this time point
        demand = demand_time_series[demand_time_series['time'] == time]['pd'].sum()
        total_demand[t] = demand
        
        # Get generation for this time point
        gen_at_t = results['generation'][results['generation']['time'] == time]
        
        # Aggregate generation by type
        for _, gen_row in gen_at_t.iterrows():
            gen_type = id_to_type.get(gen_row['id'])
            if gen_type:
                gen_by_type[t][gen_type] = gen_by_type[t].get(gen_type, 0) + gen_row['gen']
                total_gen_per_asset[gen_type] = total_gen_per_asset.get(gen_type, 0) + gen_row['gen']
                
                # Calculate generation cost
                if gen_row['id'] in id_to_gencost:
                    cost = gen_row['gen'] * id_to_gencost[gen_row['id']]
                    total_gen_cost_per_asset[gen_type] = total_gen_cost_per_asset.get(gen_type, 0) + cost
        
        # Calculate remaining capacity
        for gen_type in set(id_to_type.values()):
            gen_ids = [id for id, typ in id_to_type.items() if typ == gen_type]
            total_pmax = sum(id_to_pmax.get(id, 0) for id in gen_ids)
            used_capacity = total_gen_per_asset.get(gen_type, 0)
            remaining = total_pmax - used_capacity if total_pmax > used_capacity else 0
            remaining_capacity_series[gen_type] = remaining

    # Store generation vs demand data for comparison plot
    gen_vs_demand_data = {}
    for t in range(len(time_points)):
        gen_vs_demand_data[t] = {
            gen_type: gen_by_type[t].get(gen_type, 0) 
            for gen_type in set(id_to_type.values())
        }
        gen_vs_demand_data[t]['demand'] = total_demand[t]
    
    # Convert Series to dict if needed
    total_gen_per_asset_dict = (total_gen_per_asset.to_dict() 
                               if hasattr(total_gen_per_asset, 'to_dict') 
                               else total_gen_per_asset)
    
    total_gen_cost_per_asset_dict = (total_gen_cost_per_asset.to_dict() 
                                    if hasattr(total_gen_cost_per_asset, 'to_dict') 
                                    else total_gen_cost_per_asset)
    
    remaining_capacity_dict = (remaining_capacity_series.to_dict() 
                             if hasattr(remaining_capacity_series, 'to_dict') 
                             else remaining_capacity_series)
    
    # Convert to pandas Series for easier plotting
    total_gen_per_asset_series = pd.Series(total_gen_per_asset)
    remaining_capacity_series = pd.Series(remaining_capacity_dict)
    
    # Ensure both series have the same index
    all_types = sorted(set(total_gen_per_asset_series.index) | set(remaining_capacity_series.index))
    total_gen_per_asset_series = total_gen_per_asset_series.reindex(all_types, fill_value=0)
    remaining_capacity_series = remaining_capacity_series.reindex(all_types, fill_value=0)

    # Create figure directory if it doesn't exist
    figure_dir = os.path.join(scenario_folder, "figure")
    os.makedirs(figure_dir, exist_ok=True)

    # Plot stacked histogram
    plt.figure(figsize=(10, 6))
    plt.bar(total_gen_per_asset_series.index, total_gen_per_asset_series, 
            color='skyblue', label='Total Generation (MW)')
    plt.bar(remaining_capacity_series.index, remaining_capacity_series, 
            bottom=total_gen_per_asset_series, color='lightgray', 
            label='Remaining Capacity (MW)')
    plt.xlabel('Asset')
    plt.ylabel('Power (MW)')
    plt.title(f'Total Generation and Remaining Capacity per Asset ({season_key.capitalize()})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"hist_total_gen_with_capacity_{season_key}.png"))
    plt.close()
    print(f"Saved Generation with Capacity histogram => hist_total_gen_with_capacity_{season_key}.png")

    return (total_gen_per_asset_dict, 
            total_gen_cost_per_asset_dict,
            remaining_capacity_dict,
            gen_vs_demand_data)

# %%
def create_annual_summary_plots(scenario_data: Dict[str, Any], results_root: str) -> None:
    """Create annual generation and cost mix plots"""
    scenario_name = scenario_data['scenario_name']
    
    # Create figure directory if it doesn't exist
    figure_dir = os.path.join(results_root, scenario_name, "figure")
    os.makedirs(figure_dir, exist_ok=True)

    # Create a figure with 3 subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Annual Generation (Pie Chart)
    generation_data = {k.replace('gen_', ''): v for k, v in scenario_data.items() 
                      if k.startswith('gen_') and not k.startswith('gen_cost_')}
    if generation_data:
        # Filter out zero/negative values and take absolute values
        generation_data = {k: abs(v) for k, v in generation_data.items() 
                         if v and not pd.isna(v)}
        
        if generation_data:  # Check if we still have data after filtering
            # Calculate total generation for percentage calculation
            total_gen = sum(generation_data.values())
            # Create pie chart
            wedges, texts, autotexts = ax1.pie(
                list(generation_data.values()),
                labels=list(generation_data.keys()),
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*total_gen):,} MWh)',
                textprops={'fontsize': 8}
            )
            # Ensure text doesn't overlap
            plt.setp(autotexts, size=7, weight="bold")
            plt.setp(texts, size=8)
        else:
            ax1.text(0.5, 0.5, 'No valid generation data', 
                    ha='center', va='center')
        ax1.set_title('Annual Generation Mix')

    # Plot 2: Winter vs Summer Generation
    winter_data = {k.replace('winter_gen_', ''): v for k, v in scenario_data.items() 
                  if k.startswith('winter_gen_')}
    summer_data = {k.replace('summer_gen_', ''): v for k, v in scenario_data.items() 
                  if k.startswith('summer_gen_')}
    
    # Get all unique assets
    all_assets = sorted(set(winter_data.keys()) | set(summer_data.keys()))
    
    if all_assets:
        x = np.arange(len(all_assets))
        width = 0.35  # Width of the bars
        
        # Create bars
        winter_values = [winter_data.get(asset, 0) for asset in all_assets]
        summer_values = [summer_data.get(asset, 0) for asset in all_assets]
        
        # Plot bars
        ax2.bar(x - width/2, winter_values, width, label='Winter', color='lightblue')
        ax2.bar(x + width/2, summer_values, width, label='Summer', color='orange')
        
        # Customize plot
        ax2.set_ylabel('Generation (MWh)')
        ax2.set_title('Winter vs Summer Generation')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_assets, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(winter_values):
            if v != 0:
                ax2.text(i - width/2, v, f'{v:,.0f}', 
                        ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(summer_values):
            if v != 0:
                ax2.text(i + width/2, v, f'{v:,.0f}', 
                        ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No seasonal generation data', 
                ha='center', va='center')

    # Plot 3: NPV Comparison
    npv_data = {
        '10y': scenario_data.get('npv_10y', 0),
        '20y': scenario_data.get('npv_20y', 0),
        '30y': scenario_data.get('npv_30y', 0)
    }
    
    # Filter out NaN values
    npv_data = {k: v for k, v in npv_data.items() 
                if not pd.isna(v)}
    
    if npv_data:
        # Create bars with colors based on NPV values (red for negative, green for positive)
        bars = ax3.bar(npv_data.keys(), npv_data.values(), 
                      color=['red' if v < 0 else 'lightgreen' for v in npv_data.values()])
        ax3.set_title('Net Present Value (NPV) Comparison')
        ax3.set_ylabel('NPV (€)')
        ax3.grid(True, alpha=0.3)
        
        # Add zero line for NPV reference
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on top/bottom of bars
        for bar in bars:
            height = bar.get_height()
            # Position labels above bar for positive values, below for negative
            va = 'bottom' if height >= 0 else 'top'
            ax3.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'€{height:,.0f}',
                ha='center',
                va=va,
                rotation=0,
                fontsize=8
            )
    else:
        ax3.text(0.5, 0.5, 'No valid NPV data', 
                ha='center', va='center')

    # Add overall title
    plt.suptitle(f'Annual Summary - {scenario_name}', fontsize=16, y=1.05)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(figure_dir, "annual_summary.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved annual summary plots => {plot_path}")

# %%
def create_scenario_comparison_plot(scenario_data: Dict[str, Any], results_root: str) -> None:
    """Create a three-panel comparison plot for a scenario"""
    scenario_name = scenario_data.get('scenario_name', 'Unknown')
    scenario_folder = os.path.join(results_root, scenario_name, "figure")
    os.makedirs(scenario_folder, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Costs Distribution Plot
    cost_data = {k.replace('gen_cost_', ''): v for k, v in scenario_data.items() 
                if k.startswith('gen_cost_')}
    # Filter out zero or NaN values
    cost_data = {k: v for k, v in cost_data.items() if v and not pd.isna(v)}
    
    if cost_data:  # Only create pie chart if we have valid data
        costs = list(cost_data.values())
        labels = list(cost_data.keys())
        ax1.pie(costs, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Cost Distribution')

    # 2. Annual Generation Mix
    gen_data = {k.replace('gen_', ''): v for k, v in scenario_data.items() 
                if k.startswith('gen_') and not k.startswith('gen_cost_')}
    avail_data = {k.replace('avail_gen_', ''): v for k, v in scenario_data.items() 
                  if k.startswith('avail_gen_')}
    
    # Filter out NaN values
    gen_data = {k: v for k, v in gen_data.items() if not pd.isna(v)}
    avail_data = {k: v for k, v in avail_data.items() if not pd.isna(v)}
    
    if gen_data:  # Only create bar chart if we have valid data
        x = np.arange(len(gen_data))
        width = 0.35
        ax2.bar(x, list(gen_data.values()), width, label='Actual Generation', color='skyblue')
        if avail_data:
            ax2.bar(x, [avail_data.get(k, 0) for k in gen_data.keys()], width, 
                   label='Available Capacity', color='lightgray', alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(gen_data.keys()), rotation=45)
        ax2.legend()
    ax2.set_title('Annual Generation Mix')
    ax2.set_ylabel('Generation (MW)')

    # 3. Summer-Winter Generation Difference
    summer_gen = {k.replace('summer_gen_', ''): v 
                 for k, v in scenario_data.items() 
                 if k.startswith('summer_gen_')}
    winter_gen = {k.replace('winter_gen_', ''): v 
                 for k, v in scenario_data.items() 
                 if k.startswith('winter_gen_')}
    
    # Filter out NaN values and calculate differences
    gen_diff = {}
    for k in set(summer_gen.keys()) | set(winter_gen.keys()):
        if not pd.isna(summer_gen.get(k, 0)) and not pd.isna(winter_gen.get(k, 0)):
            gen_diff[k] = summer_gen.get(k, 0) - winter_gen.get(k, 0)
    
    if gen_diff:
        x = np.arange(len(gen_diff))
        colors = ['red' if v >= 0 else 'blue' for v in gen_diff.values()]
        
        bars = ax3.bar(x, list(gen_diff.values()), color=colors)
        ax3.set_xticks(x)
        ax3.set_xticklabels(list(gen_diff.keys()), rotation=45)
        ax3.set_title('Summer-Winter Generation Difference')
        ax3.set_ylabel('Generation Difference (MW)\n(Summer - Winter)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add zero line
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on top of bars
        for i, v in enumerate(gen_diff.values()):
            va = 'bottom' if v >= 0 else 'top'
            ax3.text(i, v, f'{v:,.0f}', ha='center', va=va)

    # Add overall title and adjust layout
    fig.suptitle(f'Scenario Analysis: {scenario_name}', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(scenario_folder, 'scenario_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

# %%
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

# %%
if __name__ == "__main__":
    main()
# %%
