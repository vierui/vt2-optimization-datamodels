#%%
# run_scenario.py
import argparse
import pandas as pd
import numpy as np
from dcopf import dcopf  # Make sure this points to your updated DCOPF file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True, help='Path to scenario folder')
    args = parser.parse_args()
    scenario_dir = args.scenario

    # 1. Load bus data
    bus = pd.read_csv(f"{scenario_dir}/bus.csv")
    # (Optional) If you want uniform lower-case columns:
    # bus.columns = bus.columns.str.lower()

    # 2. Load branch data
    branch = pd.read_csv(f"{scenario_dir}/branch.csv")
    # Ensure column names match what DCOPF references
    # If your CSV has 'rateA', rename it to 'ratea' or vice versa.
    branch.rename(columns={'rateA':'ratea'}, inplace=True, errors='ignore')
    # Create susceptance if the code needs 'sus'
    branch['sus'] = 1 / branch['x']
    # Possibly create an 'id' column (not strictly used by DCOPF, but sometimes helpful)
    branch['id'] = np.arange(1, len(branch)+1)

    # 3. Load generation time series
    gen_time_series = pd.read_csv(f"{scenario_dir}/gen.csv", parse_dates=['time'])
    gen_time_series = gen_time_series.fillna(0).sort_values('time')

    # 4. Load demand time series
    demand_time_series = pd.read_csv(f"{scenario_dir}/demand.csv", parse_dates=['time'])
    demand_time_series = demand_time_series.sort_values('time')

    # 5. Run DCOPF
    results = dcopf(
        gen_time_series=gen_time_series,
        branch=branch,
        bus=bus,
        demand_time_series=demand_time_series,
        delta_t=1
    )

    # 6. Check and Print Results
    if results and results['status'] == 'Optimal':
        print(f"Optimization was successful for scenario: {scenario_dir}")
        print(f"Total Cost: {results['cost']}")
    else:
        print(f"Optimization failed or no optimal solution for scenario: {scenario_dir}")
# %%
