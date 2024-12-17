# %%
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import os

from dcopf import dcopf

scenario_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/scenarios/base_scenario/"

# %%
# 1. Load Bus and Branch Data
bus = pd.read_csv(os.path.join(scenario_dir, "bus.csv"))
branch = pd.read_csv(os.path.join(scenario_dir, "branch.csv"))

# Calculate susceptance if not done already
branch['sus'] = 1 / branch['x']
branch['id'] = np.arange(1, len(branch) + 1)

# %%
# 2. Load gen and load data
gen_time_series = pd.read_csv(os.path.join(scenario_dir, "gen.csv"), parse_dates=['time'])
gen_time_series = gen_time_series.fillna(0)
gen_time_series = gen_time_series.sort_values('time').reset_index(drop=True)

demand_time_series = pd.read_csv(os.path.join(scenario_dir, "demand.csv"), parse_dates=['time'])
demand_time_series = demand_time_series.sort_values('time').reset_index(drop=True)

# %%
# 4. Run DCOPF
# Adjust delta_t if your time steps are different (e.g., if 1 hour steps are correct)
results = dcopf(
    gen_time_series=gen_time_series,
    branch=branch,
    bus=bus,
    demand_time_series=demand_time_series,
    delta_t=1
)

# 5. Check Results
if results and results['status'] == 'Optimal':
    print("Optimization was successful.")
    print(f"Total Cost: {results['cost']}")

    # Extract Results
   
# %%
print(branch.columns)
# %%
