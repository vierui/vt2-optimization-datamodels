# %%
#!/usr/bin/env python3

"""
run_scenario.py

Example script to demonstrate how to:
  1) Load bus.csv, branch.csv, master_gen.csv, master_load.csv, scenarios_parameters.csv
  2) Parse a chosen scenario (scenario_1) from scenarios_parameters
  3) Build the final gen_time_series & demand_time_series
  4) Run dcopf
  5) Print results
"""
# %%
import pandas as pd
import numpy as np
import ast  # for safely evaluating dictionary strings
import os
from dcopf import dcopf  # <-- Adjust import to wherever your dcopf function resides

# Adjust these paths to your local setup
working_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/working"

bus_file = os.path.join(working_dir, "bus.csv")
branch_file = os.path.join(working_dir, "branch.csv")
master_gen_file = os.path.join(working_dir, "master_gen.csv")
master_load_file = os.path.join(working_dir, "master_load.csv")
scenario_params_file = os.path.join(working_dir, "scenarios_parameters.csv")

# %%
def main():
    # 1) Load bus & branch
    bus = pd.read_csv(bus_file)
    branch = pd.read_csv(branch_file)
    # If your DCOPF expects 'ratea', rename if needed:
    branch.rename(columns={"rateA": "ratea"}, inplace=True, errors="ignore")
    # Create 'sus' = 1/x for DC power flow
    branch["sus"] = 1 / branch["x"]
    # Optionally assign ID
    branch["id"] = np.arange(1, len(branch)+1)

    # 2) Load master_gen & master_load
    master_gen = pd.read_csv(master_gen_file, parse_dates=["time"]).sort_values("time")
    master_load = pd.read_csv(master_load_file, parse_dates=["time"]).sort_values("time")

    # 3) Load scenarios_parameters to find scenario_1
    scenarios_params = pd.read_csv(scenario_params_file)
    # We'll look for "scenario_1"
    scenario_row = scenarios_params[scenarios_params["scenario_name"] == "scenario_1"].iloc[0]
    # e.g. scenario_1,"{1:'nuclear',4:'solar'}","{}",1.0

    gen_positions_str = scenario_row["gen_positions"]      # "{1:'nuclear',4:'solar'}"
    storage_units_str = scenario_row["storage_units"]      # "{}"
    load_factor = float(scenario_row["load_factor"])

    # Convert dictionary-like strings to Python dict
    gen_positions_dict = ast.literal_eval(gen_positions_str)   # e.g. {1:'nuclear', 4:'solar'}
    storage_units_dict = ast.literal_eval(storage_units_str)    # e.g. {}

    # 4) Build final gen_time_series for this scenario
    gen_ts = build_gen_time_series(master_gen, gen_positions_dict, storage_units_dict)

    # 5) Build final demand_time_series for this scenario
    demand_ts = build_demand_time_series(master_load, load_factor)

    # 6) Call DCOPF
    results = dcopf(
        gen_time_series=gen_ts,
        branch=branch,
        bus=bus,
        demand_time_series=demand_ts,
        delta_t=1
    )

    # 7) Check results
    if results and results["status"] == "Optimal":
        print("[simulate_scenario] Optimization succeeded for scenario_1.")
        print("Total Cost:", results["cost"])
        print("\nGeneration sample:")
        print(results["generation"].head())
        print("\nFlows sample:")
        print(results["flows"].head())
        print("\nStorage sample:")
        print(results["storage"].head())
    else:
        print("[simulate_scenario] Optimization NOT optimal or returned None.")

# %%
def build_gen_time_series(master_gen, gen_positions_dict, storage_units_dict):
    """
    Filters master_gen so that only the selected generator types appear,
    assigns bus IDs as needed, sets 'id' from type_to_id, etc.
    """

    # Convert scenario dict (like {1:'nuclear',4:'solar'}) -> a list of 'type's
    selected_types = list(gen_positions_dict.values())

    # Filter master_gen so we only keep rows whose 'type' is in selected_types
    scenario_gen = master_gen[master_gen["type"].isin(selected_types)].copy()

    # Map each type -> bus from gen_positions_dict if you want to reassign
    # e.g. gen_positions_dict = {1:'nuclear', 4:'solar'}
    # Then 'nuclear' -> bus=1, 'solar' -> bus=4
    type_to_bus_map = {v: k for k,v in gen_positions_dict.items()}

    def assign_bus(row):
        t = row["type"]
        if t in type_to_bus_map:
            return type_to_bus_map[t]
        else:
            return row["bus"]  # fallback to whatever is in master_gen if type not found
    scenario_gen["bus"] = scenario_gen.apply(assign_bus, axis=1)

    # Optional: Add an integer 'id' for DCOPF
    type_to_id = {
        "nuclear": 1,
        "solar": 2,
        "wind": 3,
        "gas": 4,
        "battery1": 5,
        "battery2": 6
    }
    scenario_gen["id"] = scenario_gen["type"].map(type_to_id)

    # Sort by time and the newly added id
    scenario_gen.sort_values(["time","id"], inplace=True)

    return scenario_gen

# %%
def build_demand_time_series(master_load, load_factor):
    """
    For a quick example, let's assume you want
    all loads from your master_load (which might have bus 5 & 6).
    Then apply the load_factor.

    If you only want certain buses or certain time slices, you can filter further.
    """
    scenario_load = master_load.copy()
    # scale
    scenario_load["pd"] = scenario_load["pd"] * load_factor

    # must have columns: [time, bus, pd]
    scenario_load.sort_values(["time","bus"], inplace=True)
    return scenario_load

# %%
if __name__ == "__main__":
    main()
# %%
