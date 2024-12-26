#!/usr/bin/env python3

# %%
"""
multi_scenario.py

Example script that:
- Loads a list of scenarios from scenarios_parameters.csv
- For each scenario, runs DCOPF 3 times (winter, summer, autumn_spring).
- If any season fails, marks scenario as "no result."
- Otherwise, sums up annual cost from the 3 runs.
- Saves each scenario's results and plots in an individual folder under /data/results.
- Finally, appends/updates scenario_results.csv with final costs (rounded to 1 decimal).
"""

import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
import networkx as nx
from dcopf import dcopf  # Ensure this imports your updated DCOPF code

# Paths
working_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/working"
results_root = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/results"   # Where each scenario's folder is created

bus_file = os.path.join(working_dir, "bus.csv")
branch_file = os.path.join(working_dir, "branch.csv")
master_gen_file = os.path.join(working_dir, "master_gen.csv")
master_load_file = os.path.join(working_dir, "master_load.csv")
scenarios_params_file = os.path.join(working_dir, "scenarios_parameters.csv")

# Example: 13 weeks for winter, 13 for summer, 26 for autumn_spring
season_weights = {
    "winter": 13,
    "summer": 13,
    "autumn_spring": 26
}

# Map each gen/storage type to a numeric ID:
type_to_id = {
    "nuclear": 1,
    "gas": 2,
    "wind": 3,
    "solar": 4,
    "battery1": 101,
    "battery2": 102
}

# %%
def main():
    # 1) Load base network
    bus = pd.read_csv(bus_file)
    branch = pd.read_csv(branch_file)
    branch.rename(columns={"rateA": "ratea"}, inplace=True, errors="ignore")
    branch["sus"] = 1 / branch["x"]
    branch["id"] = np.arange(1, len(branch) + 1)

    master_gen = pd.read_csv(master_gen_file, parse_dates=["time"]).sort_values("time")
    master_load = pd.read_csv(master_load_file, parse_dates=["time"]).sort_values("time")

    # 2) Load all scenarios
    scenarios_df = pd.read_csv(scenarios_params_file)

    # We'll store scenario-level results in a list of dict (to build scenario_results.csv)
    scenario_results = []

    # 3) For each scenario, run 3 seasons
    for idx, row in scenarios_df.iterrows():
        scenario_name = row["scenario_name"]
        gen_positions_str = row["gen_positions"]
        storage_positions_str = row["storage_units"]
        load_factor = float(row["load_factor"])

        # Convert from str to Python dict
        # gen_positions_dict and storage_positions_dict are now {bus_i: 'type'}
        gen_positions_dict_raw = ast.literal_eval(gen_positions_str)       # e.g. {1:'nuclear', 4:'solar'}
        storage_positions_dict_raw = ast.literal_eval(storage_positions_str) # e.g. {2:'battery1'}

        # Convert type strings to gen_ids using type_to_id
        gen_positions_dict = {bus: type_to_id[gen_type] for bus, gen_type in gen_positions_dict_raw.items()}
        storage_positions_dict = {bus: type_to_id[gen_type] for bus, gen_type in storage_positions_dict_raw.items()}

        print(f"\n[MultiScenario] Processing {scenario_name} with gen_positions={gen_positions_dict_raw} (mapped to IDs {gen_positions_dict}), storage_positions={storage_positions_dict_raw} (mapped to IDs {storage_positions_dict}), load_factor={load_factor}")

        # We'll store season->cost in a dict
        season_costs = {}
        all_ok = True

        # Attempt the 3 seasons
        for season_key in ["winter", "summer", "autumn_spring"]:
            print(f"\n[MultiScenario] Running scenario={scenario_name}, season={season_key}...")

            # Build gen_time_series for this scenario+season
            gen_ts = build_gen_time_series(master_gen, gen_positions_dict, storage_positions_dict, season_key)
            # Build demand_time_series for this scenario+season
            demand_ts = build_demand_time_series(master_load, load_factor, season_key)

            # Run DCOPF
            results = dcopf(gen_ts, branch, bus, demand_ts, delta_t=1)

            # Debug prints to check results
            print(f"[DEBUG] scenario={scenario_name}, season={season_key} => DCOPF returned:")
            print(f"[DEBUG] results = {results}")
            if results is not None:
                print(f"[DEBUG] results['status'] = {results.get('status')}")
                print(f"[DEBUG] results['cost']   = {results.get('cost')}")

            # Check status
            if (results is None) or (results["status"] != "Optimal"):
                print(f"[MultiScenario] {scenario_name}, season={season_key} => NOT Optimal.")
                all_ok = False
                break
            else:
                # Store cost
                season_costs[season_key] = results["cost"]
                print(f"[MultiScenario] {scenario_name}, season={season_key} => Optimal with cost={results['cost']:.1f}")

        # 4) If all seasons are good, compute annual cost & plot
        if all_ok:
            # Weighted annual cost
            annual_cost = 0.0
            for s_key, cost_val in season_costs.items():
                weight = season_weights.get(s_key, 1)
                annual_cost += cost_val * weight

            # Round the costs
            winter_cost = round(season_costs.get("winter", 0.0), 1)
            summer_cost = round(season_costs.get("summer", 0.0), 1)
            autumn_cost = round(season_costs.get("autumn_spring", 0.0), 1)
            annual_cost_rounded = round(annual_cost, 1)

            # Create scenario folder: e.g. /data/results/scenario_1/
            scenario_folder = os.path.join(results_root, scenario_name)
            os.makedirs(scenario_folder, exist_ok=True)

            # Generate plots for each season
            for season_key in ["winter", "summer", "autumn_spring"]:
                print(f"\n[MultiScenario] Generating plots for scenario={scenario_name}, season={season_key}...")

                # Build gen_time_series and demand_time_series again for plotting
                gen_ts_plot = build_gen_time_series(master_gen, gen_positions_dict, storage_positions_dict, season_key)
                demand_ts_plot = build_demand_time_series(master_load, load_factor, season_key)

                # Run DCOPF again to get results for plotting
                plot_results = dcopf(gen_ts_plot, branch, bus, demand_ts_plot, delta_t=1)

                if (plot_results is not None) and (plot_results["status"] == "Optimal"):
                    # Make plots with plot_results
                    plot_scenario_results(plot_results, demand_ts_plot, branch, bus, scenario_folder, season_key)
                else:
                    print(f"[MultiScenario] {scenario_name}, season={season_key} plotting step => final solve not optimal, skipping plots")

            # Append scenario-level result
            scenario_results.append({
                "scenario_name": scenario_name,
                "winter_cost": winter_cost,
                "summer_cost": summer_cost,
                "autumn_spring_cost": autumn_cost,
                "annual_cost": annual_cost_rounded
            })
            print(f"[MultiScenario] {scenario_name} => Annual Cost: {annual_cost_rounded}")
        else:
            # Mark scenario as fail
            scenario_results.append({
                "scenario_name": scenario_name,
                "winter_cost": None,
                "summer_cost": None,
                "autumn_spring_cost": None,
                "annual_cost": None
            })
            print(f"[MultiScenario] {scenario_name} => Marked as NOT Optimal due to season failure.")

    def build_gen_time_series(master_gen, gen_positions_dict, storage_positions_dict, season_key):
        """
        Builds the scenario-specific generation time series.

        :param master_gen: DataFrame from create_master_gen.py, 
                           with columns [time, id, type, pmax, pmin, gencost, emax, einitial, eta, season].
        :param gen_positions_dict: dictionary {bus_i: gen_id, ...} for non-storage assets
        :param storage_positions_dict: dictionary {bus_i: gen_id, ...} for storage assets
        :param season_key: e.g. "winter", "summer", or "autumn_spring"
        :return: DataFrame with the same columns as master_gen, 
                 but only for the selected IDs, with the bus column overridden as specified.
        """

        # 1) Filter master_gen by 'season'
        scenario_gen = master_gen[master_gen["season"] == season_key].copy()

        # 2) Combine both dicts if you have storage
        merged_positions = {}
        merged_positions.update(gen_positions_dict)       # e.g. {1:1, 4:4}
        merged_positions.update(storage_positions_dict)   # e.g. {2:101}

        # 3) Extract all gen_ids from the merged dict
        selected_ids = list(merged_positions.values())

        # 4) Filter scenario_gen to keep only those IDs
        scenario_gen = scenario_gen[scenario_gen["id"].isin(selected_ids)].copy()

        # 5) For each (bus_i -> gen_id), override scenario_gen["bus"] to bus_i
        for bus_i, gen_id in merged_positions.items():
            scenario_gen.loc[scenario_gen["id"] == gen_id, "bus"] = bus_i

        # 6) Ensure no duplicate (time, id) pairs
        before_dedup = len(scenario_gen)
        scenario_gen.drop_duplicates(subset=['time', 'id'], keep='first', inplace=True)
        after_dedup = len(scenario_gen)
        if before_dedup != after_dedup:
            print(f"[build_gen_time_series] Dropped {before_dedup - after_dedup} duplicate (time, id) entries.")

        # 7) Sort by time and id
        scenario_gen.sort_values(["time", "id"], inplace=True)
        scenario_gen.reset_index(drop=True, inplace=True)

        return scenario_gen

    def build_demand_time_series(master_load, load_factor, season_key):
        """
        Filter master_load for the chosen season, scale load if needed.

        :param master_load: DataFrame with columns ['time', 'bus', 'pd', 'season']
        :param load_factor: float to scale the demand
        :param season_key: e.g. "winter", "summer", or "autumn_spring"
        :return: DataFrame with columns ['time', 'bus', 'pd'], sorted
        """
        # 1) Filter by season
        scenario_load = master_load[master_load["season"] == season_key].copy()

        # 2) Scale the demand
        scenario_load["pd"] *= load_factor

        # 3) Ensure no duplicate (time, bus) pairs
        before_dedup = len(scenario_load)
        scenario_load.drop_duplicates(subset=['time', 'bus'], keep='first', inplace=True)
        after_dedup = len(scenario_load)
        if before_dedup != after_dedup:
            print(f"[build_demand_time_series] Dropped {before_dedup - after_dedup} duplicate (time, bus) entries.")

        # 4) Sort by time and bus
        scenario_load.sort_values(["time", "bus"], inplace=True)
        scenario_load.reset_index(drop=True, inplace=True)

        return scenario_load

    def plot_scenario_results(results, demand_time_series, branch, bus, scenario_folder, season_key):
        """
        Function to produce and save plots 
        (Generation vs. Demand, line flows, network shape, Total Loads) 
        in scenario_folder.

        :param results: dict returned by dcopf function
        :param demand_time_series: DataFrame with columns ['time', 'bus', 'pd']
        :param branch: DataFrame with branch information
        :param bus: DataFrame with bus information
        :param scenario_folder: Path to save the plots
        :param season_key: Current season being plotted
        """
        generation_over_time = results['generation'].copy()
        storage_over_time = results['storage'].copy()
        flows_over_time = results['flows'].copy()

        # 1) Pivot the Generation Data
        # Ensure no duplicate (time, id) by aggregating if necessary
        gen_pivot = generation_over_time.groupby(['time', 'id']).sum().reset_index()
        gen_pivot = gen_pivot.pivot(index='time', columns='id', values='gen')

        # Check for duplicates in the pivot index
        if gen_pivot.index.duplicated().any():
            print("[plot_scenario_results] Duplicate times found in gen_pivot.index. Aggregating by summing.")
            gen_pivot = gen_pivot.groupby(level=0).sum()

        # Now sort by time
        gen_pivot = gen_pivot.sort_index()

        # 2) Prepare the Total Demand Series
        # Remove duplicates by time and aggregate if necessary
        demand_series = demand_time_series.groupby('time')['pd'].sum()
        # Reindex to match gen_pivot index
        try:
            demand_ts = demand_series.reindex(gen_pivot.index, fill_value=0)
        except ValueError as e:
            print(f"[plot_scenario_results] Reindexing failed: {e}")
            print("[plot_scenario_results] Handling duplicate labels by aggregating.")
            demand_series = demand_time_series.groupby('time')['pd'].sum()
            demand_ts = demand_series.reindex(gen_pivot.index, fill_value=0)

        # 3) Map generator IDs to names (example)
        id_map = {
            1: "Nuclear",
            2: "Gas",
            3: "Wind",
            4: "Solar",
            101: "Battery1",
            102: "Battery2"
        }
        gen_pivot.rename(columns=id_map, inplace=True, errors="ignore")

        # 4) Calculate Total Loads
        total_loads = demand_ts.sum()
        print(f"[plot_scenario_results] Total Load for {season_key}: {total_loads:.1f} MW")

        # 5) Plot Generation vs. Demand
        fig, ax = plt.subplots(figsize=(12, 6))
        # stackplot needs the transposed 2D array
        ax.stackplot(gen_pivot.index, gen_pivot.T, labels=gen_pivot.columns, alpha=0.8)
        ax.plot(demand_ts.index, demand_ts.values, label='Total Demand', color='black', linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Power (MW)')
        ax.set_title(f'Generation Distribution Over Time vs. Demand ({season_key.capitalize()})')
        ax.legend(loc='upper left')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        gen_demand_plot = os.path.join(scenario_folder, f"generation_vs_demand_{season_key}.png")
        plt.savefig(gen_demand_plot)
        plt.close(fig)
        print(f"[plot_scenario_results] Saved Generation vs Demand plot => {gen_demand_plot}")

        # 6) Plot Sum of All Loads
        # Assuming 'Total Demand' is the sum across all buses at each time step
        fig_load, ax_load = plt.subplots(figsize=(12, 6))
        ax_load.plot(demand_ts.index, demand_ts.values, label='Total Demand', color='red', linewidth=2)

        # Overlay generation sum to check matching
        generation_sum = gen_pivot.sum(axis=1)
        ax_load.plot(gen_pivot.index, generation_sum, label='Total Generation', color='blue', linewidth=2)

        ax_load.set_xlabel('Time')
        ax_load.set_ylabel('Power (MW)')
        ax_load.set_title(f'Total Generation vs. Total Demand ({season_key.capitalize()})')
        ax_load.legend(loc='upper left')
        ax_load.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        total_load_plot = os.path.join(scenario_folder, f"total_generation_vs_total_demand_{season_key}.png")
        plt.savefig(total_load_plot)
        plt.close(fig_load)
        print(f"[plot_scenario_results] Saved Total Generation vs. Total Demand plot => {total_load_plot}")

        # 7) Check flow limits
        flows_with_limits = flows_over_time.merge(
            branch[['fbus', 'tbus', 'ratea']],
            left_on=['from_bus', 'to_bus'],
            right_on=['fbus', 'tbus'],
            how='left'
        )
        flows_with_limits['abs_flow'] = flows_with_limits['flow'].abs()
        flows_with_limits['within_limits'] = flows_with_limits['abs_flow'] <= flows_with_limits['ratea']

        flows_exceeding_limits = flows_with_limits[~flows_with_limits['within_limits']]
        exceed_csv = os.path.join(scenario_folder, f"flows_exceeding_limits_{season_key}.csv")

        if not flows_exceeding_limits.empty:
            flows_exceeding_limits.to_csv(exceed_csv, index=False)
            print(f"[plot_scenario_results] Some lines exceed limits. See CSV => {exceed_csv}")
        else:
            print(f"[plot_scenario_results] All line flows are within limits for {season_key}.")

        # 8) Plot flows over time for each line
        unique_lines = flows_with_limits[['from_bus', 'to_bus']].drop_duplicates()
        fig2, ax2 = plt.subplots(figsize=(12,6))
        for idx, row in unique_lines.iterrows():
            line_flows = flows_with_limits[
                (flows_with_limits['from_bus'] == row['from_bus']) & 
                (flows_with_limits['to_bus'] == row['to_bus'])
            ]
            ax2.plot(
                line_flows['time'],
                line_flows['flow'],
                label=f"Line {row['from_bus']}→{row['to_bus']}"
            )
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Flow (MW)')
        ax2.set_title(f'Line Flows Over Time ({season_key.capitalize()})')
        ax2.legend()
        ax2.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        flow_plot = os.path.join(scenario_folder, f"line_flows_{season_key}.png")
        plt.savefig(flow_plot)
        plt.close(fig2)
        print(f"[plot_scenario_results] Saved line flows plot => {flow_plot}")

        # 9) Visualize the network at a specific time
        # Choose the median time to visualize a representative snapshot
        median_time = gen_pivot.index[len(gen_pivot) // 2]
        flows_at_time = flows_over_time[flows_over_time['time'] == median_time]

        G = nx.DiGraph()
        for _, rowB in bus.iterrows():
            G.add_node(rowB['bus_i'])
        for _, rowB in branch.iterrows():
            G.add_edge(rowB['fbus'], rowB['tbus'], capacity=rowB['ratea'])

        for _, rowF in flows_at_time.iterrows():
            if (rowF['from_bus'], rowF['to_bus']) in G.edges():
                G[rowF['from_bus']][rowF['to_bus']]['flow'] = rowF['flow']
            else:
                # If the flow exists in flows_over_time but not in branch, add it
                G.add_edge(rowF['from_bus'], rowF['to_bus'], capacity=rowF['ratea'], flow=rowF['flow'])

        pos = nx.spring_layout(G, seed=42)
        edge_flows = [abs(G[u][v].get('flow',0)) for u,v in G.edges()]
        max_flow = max(edge_flows) if edge_flows else 1
        edge_widths = [5*(f/max_flow) for f in edge_flows]

        fig3, ax3 = plt.subplots(figsize=(12,8))
        nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax3)
        nx.draw_networkx_labels(G, pos, ax=ax3)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=edge_widths, ax=ax3)

        edge_labels = {(u,v): f"{G[u][v].get('flow',0):.1f} MW" for u,v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax3)

        plt.title(f"Network Flows at {median_time} ({season_key.capitalize()})")
        plt.axis('off')

        flow_network_png = os.path.join(scenario_folder, f"network_flows_{season_key}.png")
        plt.savefig(flow_network_png)
        plt.close(fig3)
        print(f"[plot_scenario_results] Saved network flow diagram => {flow_network_png}")

    if __name__ == "__main__":
        main()