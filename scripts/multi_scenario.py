# %%
"""
multi_scenario.py

Example script that:
- Loads a list of scenarios from scenarios_parameters.csv
- For each scenario, runs DCOPF 3 times (winter, summer, autumn_spring).
- If any season fails, marks scenario as "no result."
- Otherwise, sums up annual cost from the 3 runs.
- Finally, can plot or print the results.
"""

import pandas as pd
import numpy as np
import ast
import os
from dcopf import dcopf

# Adjust these to your directories
working_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/working"
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

    # We'll store results in a list of dict
    scenario_results = []

    # 3) For each scenario, run 3 seasons
    for idx, row in scenarios_df.iterrows():
        scenario_name = row["scenario_name"]
        gen_positions_str = row["gen_positions"]
        storage_units_str = row["storage_units"]
        load_factor = float(row["load_factor"])

        # Convert from str to Python dict
        gen_positions_dict = ast.literal_eval(gen_positions_str)
        storage_units_dict = ast.literal_eval(storage_units_str)

        # We'll store season->cost in a dict
        season_costs = {}

        # Attempt the 3 seasons
        all_ok = True
        for season_key in ["winter", "summer", "autumn_spring"]:
            print(f"\n[DEBUG] Starting season={season_key} for scenario={scenario_name}...")

            gen_ts = build_gen_time_series(master_gen, gen_positions_dict, storage_units_dict, season_key)
            demand_ts = build_demand_time_series(master_load, load_factor, season_key)

            results = dcopf(gen_ts, branch, bus, demand_ts, delta_t=1)

            print(f"[DEBUG] scenario={scenario_name}, season={season_key} => DCOPF returned:")
            print(f"[DEBUG] results = {results}")
            if results is not None:
                print(f"[DEBUG] results['status'] = {results.get('status')}")
                print(f"[DEBUG] results['cost']   = {results.get('cost')}")

            if results is None or results["status"] != "Optimal":
                print(f"[MultiScenario] {scenario_name}, season={season_key} => NOT Optimal.")
                all_ok = False
                break
            else:
                season_costs[season_key] = results["cost"]

        # If all seasons are good, compute annual cost
        if all_ok:
            # e.g. annual_cost = sum( season_costs[s] * weight_s )
            annual_cost = 0.0
            for s_key, cost_val in season_costs.items():
                weight = season_weights.get(s_key, 1)
                annual_cost += cost_val * weight

            scenario_results.append({
                "scenario_name": scenario_name,
                "winter_cost": season_costs["winter"],
                "summer_cost": season_costs["summer"],
                "autumn_spring_cost": season_costs["autumn_spring"],
                "annual_cost": annual_cost
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


    # 4) Create a DataFrame of scenario results
    results_df = pd.DataFrame(scenario_results)
    # Save or print
    out_csv = os.path.join(working_dir, "scenario_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[MultiScenario] Results saved to: {out_csv}")

    # 5) Optionally plot a bar chart of annual cost
    # Let's filter out scenarios with None
    plot_df = results_df.dropna(subset=["annual_cost"])
    if not plot_df.empty:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,5))
            plt.bar(plot_df["scenario_name"], plot_df["annual_cost"], color='skyblue')
            plt.xlabel("Scenario")
            plt.ylabel("Annual Cost")
            plt.title("Annual Cost by Scenario")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "annual_cost_bar.png"))
            print("[MultiScenario] Plot saved as annual_cost_bar.png")
        except ImportError:
            print("[MultiScenario] Matplotlib not installed, skipping plot.")
    else:
        print("[MultiScenario] No scenario had valid annual cost, skipping plot.")

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
    # If master_load has a 'season' column with 'winter','summer','autumn_spring'
    scenario_load = master_load[master_load["season"] == season_key].copy()
    # scale
    scenario_load["pd"] *= load_factor

    # must have [time,bus,pd]
    scenario_load.sort_values(["time","bus"], inplace=True)
    return scenario_load

# %%
if __name__ == "__main__":
    main()
# %%
