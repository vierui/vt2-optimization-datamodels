# %%
#!/usr/bin/env python3

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
# %%
import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
import networkx as nx
from dcopf import dcopf  # Make sure this is your updated DCOPF code

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
        storage_units_str = row["storage_units"]
        load_factor = float(row["load_factor"])

        # Convert from str to Python dict
        gen_positions_dict = ast.literal_eval(gen_positions_str)
        storage_units_dict = ast.literal_eval(storage_units_str)  # if needed
        # (You might or might not use storage_units_dict depending on your setup)

        # We'll store season->cost in a dict
        season_costs = {}
        all_ok = True

        # Attempt the 3 seasons
        for season_key in ["winter", "summer", "autumn_spring"]:
            print(f"\n[MultiScenario] Running scenario={scenario_name}, season={season_key}...")

            # Build gen_time_series for this scenario+season
            gen_ts = build_gen_time_series(master_gen, gen_positions_dict, storage_units_dict, season_key)
            # Build demand_time_series for this scenario+season
            demand_ts = build_demand_time_series(master_load, load_factor, season_key)

            # Run DCOPF
            results = dcopf(gen_ts, branch, bus, demand_ts, delta_t=1)

            # Check status
            if (results is None) or (results["status"] != "Optimal"):
                print(f"[MultiScenario] {scenario_name}, season={season_key} => NOT Optimal.")
                all_ok = False
                break
            else:
                # Store cost
                season_costs[season_key] = results["cost"]
                # If you want to save partial results for each season, do it here
                # or wait until after we confirm all seasons are OK
                # e.g. save or plot partial results
                # pass

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

            # Optionally run final DCOPF again for a "combined" approach OR just plot e.g. last season
            # More commonly, you might do your 3 solves, pick the final or store the results from each.
            # For demonstration, let's just do the "summer" again to produce a sample plot:
            # (In real usage, you'd do the same for each season or pick a representative.)

            # We'll produce one final solve for "summer" to generate plots:
            final_gen_ts = build_gen_time_series(master_gen, gen_positions_dict, storage_units_dict, "summer")
            final_demand_ts = build_demand_time_series(master_load, load_factor, "summer")
            final_results = dcopf(final_gen_ts, branch, bus, final_demand_ts, delta_t=1)
            if (final_results is not None) and (final_results["status"] == "Optimal"):
                # Make plots with final_results
                plot_scenario_results(final_results, final_demand_ts, branch, bus, scenario_folder)
            else:
                print(f"[MultiScenario] {scenario_name} plotting step => final solve not optimal, skipping plots")

            # Append scenario-level result
            scenario_results.append({
                "scenario_name": scenario_name,
                "winter_cost": winter_cost,
                "summer_cost": summer_cost,
                "autumn_spring_cost": autumn_cost,
                "annual_cost": annual_cost_rounded
            })
        else:
            # Mark scenario as fail
            scenario_results.append({
                "scenario_name": scenario_name,
                "winter_cost": None,
                "summer_cost": None,
                "autumn_spring_cost": None,
                "annual_cost": None
            })

    # 5) Create/Update a DataFrame of scenario results
    results_df = pd.DataFrame(scenario_results)
    # Save or print
    out_csv = os.path.join(results_root, "scenario_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[MultiScenario] scenario_results saved to: {out_csv}")

# %%
def build_gen_time_series(master_gen, gen_positions_dict, storage_units_dict, season_key):
    """
    Example function that filters master_gen for the chosen season
    and generator assignments, then assigns a numeric 'id' and a 'bus' 
    for each type. Adapt if your master_gen already has these columns.
    """

    # (1) Filter by season if your master_gen has a 'season' column
    scenario_gen = master_gen[master_gen["season"] == season_key].copy()

    # (2) Filter by 'type' in gen_positions_dict (e.g., {1:'nuclear',4:'solar'} => ['nuclear','solar'])
    selected_types = list(gen_positions_dict.values())
    scenario_gen = scenario_gen[scenario_gen["type"].isin(selected_types)].copy()

    # (3) Create a mapping from 'type' -> bus, using gen_positions_dict
    #  e.g. gen_positions_dict = {1:'nuclear',4:'solar'}
    #  => 'nuclear' -> 1, 'solar' -> 4
    type_to_bus_map = {v: k for k, v in gen_positions_dict.items()}

    def assign_bus(gen_type):
        # If generator type is in the dictionary, return the bus
        # Otherwise, fallback to a default bus or NaN
        return type_to_bus_map.get(gen_type, float('nan'))

    # Because 'bus' isn't in master_gen, we define it from scenario dict
    scenario_gen["bus"] = scenario_gen["type"].apply(assign_bus)

    # If you want to ensure no NaNs remain, check for them
    # scenario_gen.dropna(subset=["bus"], inplace=True)

    # (4) If you also want a numeric 'id' for DCOPF, map each 'type' to an integer
    # Example dictionary:
    type_to_id = {
        "nuclear": 1,
        "solar": 2,
        "wind": 3,
        "gas": 4,
        "battery1": 5,
        "battery2": 6
    }
    scenario_gen["id"] = scenario_gen["type"].map(type_to_id)

    # (5) Sort if needed
    # Now that 'id' and 'bus' exist, you can sort by time, id, etc.
    scenario_gen.sort_values(["time","id"], inplace=True)

    return scenario_gen

# %%
def build_demand_time_series(master_load, load_factor, season_key):
    """
    Filter master_load for the chosen season, scale load if needed.
    """
    if 'season' in master_load.columns:
        scenario_load = master_load[master_load["season"] == season_key].copy()
    else:
        scenario_load = master_load.copy()

    scenario_load["pd"] *= load_factor
    scenario_load.sort_values(["time","bus"], inplace=True)
    return scenario_load

# %%
# Plots generation
'''
def plot_scenario_results(results, demand_time_series, branch, bus, scenario_folder):
    """
    Revised function to produce and save plots 
    (Generation vs. Demand, line flows, network shape) 
    in scenario_folder. Handles duplicate time indices in gen_pivot.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import os
    
    generation = results['generation'].copy()
    # Remove duplicates if any (time, id) combos
    generation.drop_duplicates(subset=['time','id'], keep='first', inplace=True)
    
    # Pivot
    gen_pivot = generation.pivot(index='time', columns='id', values='gen')
    
    # If pivot has duplicates in the index (time)
    if gen_pivot.index.duplicated().any():
        # sum duplicates or drop them
        gen_pivot = gen_pivot.groupby(level=0).sum()
    
    gen_pivot = gen_pivot.sort_index()

    # demand side
    # (1) remove duplicates by time
    demand_time_series = demand_time_series.drop_duplicates(subset=['time'], keep='first')
    # (2) set index
    demand_series = demand_time_series.set_index('time')['pd']
    # if demand_series has duplicates in index, you can similarly groupby
    if demand_series.index.duplicated().any():
        demand_series = demand_series.groupby(level=0).sum()

    # reindex
    demand_series = demand_series.reindex(gen_pivot.index, fill_value=0)
    
    # (Optional) If demand_time_series has duplicates as well, do a similar approach
    
    # 3) Map generator IDs to names (example)
    id_map = {
        1: "Nuclear",
        2: "Gas",
        3: "Wind",
        4: "Solar",
        5: "Wind Batt",   # if you used battery1=5
        6: "Battery1",    # or some naming
        7: "Battery2"
    }
    gen_pivot.rename(columns=id_map, inplace=True, errors="ignore")
    
    # 4) Plot Generation vs. Demand
    fig, ax = plt.subplots(figsize=(12,6))
    # stackplot needs the transposed 2D array
    ax.stackplot(gen_pivot.index, gen_pivot.T, labels=gen_pivot.columns, alpha=0.8)
    ax.plot(demand_ts.index, demand_ts.values, label='Total Demand', color='black', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Generation Distribution Over Time vs. Demand')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    gen_demand_plot = os.path.join(scenario_folder, "generation_vs_demand.png")
    plt.savefig(gen_demand_plot)
    plt.close(fig)
    print(f"[MultiScenario] Saved Generation vs Demand plot => {gen_demand_plot}")
    
    # 5) Check flow limits & plot line flows ...
    #    (the same code as before)
    flows_with_limits = flows_over_time.merge(
        branch[['fbus','tbus','ratea']],
        left_on=['from_bus','to_bus'],
        right_on=['fbus','tbus'],
        how='left'
    )
    flows_with_limits['abs_flow'] = flows_with_limits['flow'].abs()
    flows_with_limits['within_limits'] = flows_with_limits['abs_flow'] <= flows_with_limits['ratea']
    
    flows_exceeding_limits = flows_with_limits[~flows_with_limits['within_limits']]
    if not flows_exceeding_limits.empty:
        exceed_csv = os.path.join(scenario_folder, "flows_exceeding_limits.csv")
        flows_exceeding_limits.to_csv(exceed_csv, index=False)
        print("[MultiScenario] Some lines exceed limits. See CSV =>", exceed_csv)
    else:
        print("[MultiScenario] All line flows are within limits.")
    
    # Plot flows over time
    unique_lines = flows_with_limits[['from_bus','to_bus']].drop_duplicates()
    fig2, ax2 = plt.subplots(figsize=(12,6))
    for idx, row in unique_lines.iterrows():
        line_flows = flows_with_limits[
            (flows_with_limits['from_bus'] == row['from_bus']) & 
            (flows_with_limits['to_bus'] == row['to_bus'])
        ]
        ax2.plot(line_flows['time'], line_flows['flow'], label=f"Line {row['from_bus']}->{row['to_bus']}")
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Flow (MW)')
    ax2.set_title('Line Flows Over Time')
    ax2.legend()
    ax2.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    flow_plot = os.path.join(scenario_folder, "line_flows.png")
    plt.savefig(flow_plot)
    plt.close(fig2)
    print(f"[MultiScenario] Saved line flows plot => {flow_plot}")
    
    # 6) Visualize the network at a specific time
    #   (same logic as before)
    time_to_visualize = gen_pivot.index[0]  # or pick a time
    flows_at_time = flows_over_time[flows_over_time['time'] == time_to_visualize]
    
    G = nx.DiGraph()
    for _, rowB in bus.iterrows():
        G.add_node(rowB['bus_i'])
    for _, rowB in branch.iterrows():
        G.add_edge(rowB['fbus'], rowB['tbus'], capacity=rowB['ratea'])
    
    for idxF, rowF in flows_at_time.iterrows():
        if (rowF['from_bus'], rowF['to_bus']) in G.edges():
            G[rowF['from_bus']][rowF['to_bus']]['flow'] = rowF['flow']
        else:
            G.add_edge(rowF['from_bus'], rowF['to_bus'], flow=rowF['flow'])
    
    pos = nx.spring_layout(G, seed=42)
    edge_flows = [abs(G[u][v].get('flow',0)) for u,v in G.edges()]
    max_flow = max(edge_flows) if edge_flows else 1
    edge_widths = [5*(f/max_flow) for f in edge_flows]
    
    fig3, ax3 = plt.subplots(figsize=(12,8))
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=edge_widths)
    
    edge_labels = {(u,v): f"{G[u][v].get('flow',0):.1f} MW" for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(f"Network Flows at {time_to_visualize}")
    plt.axis('off')
    
    flow_network_png = os.path.join(scenario_folder, "network_flows.png")
    plt.savefig(flow_network_png)
    plt.close(fig3)
    print(f"[MultiScenario] Saved network flow diagram => {flow_network_png}")

'''
# %%
if __name__ == "__main__":
    main()
# %%
