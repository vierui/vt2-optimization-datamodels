# run_scenario.py
import argparse
import pandas as pd
import numpy as np
from dcopf import dcopf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True, help='Path to scenario folder')
    args = parser.parse_args()
    scenario_dir = args.scenario

    # Load data
    bus = pd.read_csv(f"{scenario_dir}/bus.csv")
    branch = pd.read_csv(f"{scenario_dir}/branch.csv")
    branch['sus'] = 1 / branch['x']
    branch['id'] = np.arange(1, len(branch)+1)

    gen_time_series = pd.read_csv(f"{scenario_dir}/gen.csv", parse_dates=['time'])
    gen_time_series = gen_time_series.fillna(0).sort_values('time')

    demand_time_series = pd.read_csv(f"{scenario_dir}/demand.csv", parse_dates=['time'])
    demand_time_series = demand_time_series.sort_values('time')

    # Run DCOPF
    results = dcopf(gen_time_series=gen_time_series, branch=branch, bus=bus, demand_time_series=demand_time_series, delta_t=1)

    if results and results['status'] == 'Optimal':
        print(f"Optimization was successful for scenario: {scenario_dir}")
        print(f"Total Weekly Cost: {results['cost']}")
    else:
        print(f"Optimization failed or no optimal solution for scenario: {scenario_dir}")