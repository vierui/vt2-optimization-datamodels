# %%
# 1. Initialization
# ===========================

import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import networkx as nx

# Working directory
datadir = "/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/processed/"

# %%
# 2. Data
# ===========================

# Load demand data
demand_data = pd.read_csv(
    '/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/vt1-energy-investment-model/data/raw/data-load-becc.csv',
    delimiter=';',
    names=['time', 'load'],
    parse_dates=['time'],
    dayfirst=True,
    header=0
)

demand_data['load'] = pd.to_numeric(demand_data['load']) * 100

# Load wind data
wind_data = pd.read_csv(
    '/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/raw/wind-sion-2023.csv',
    skiprows=3,
    parse_dates=['time'],
    delimiter=','
)

# Create generator time series DataFrame
wind_gen = pd.DataFrame({
    'time': wind_data['time'],
    'id': 1,
    'bus': 1,
    'pmax': wind_data['electricity'],
    'pmin': 0,
    'gencost': 5
})

solar_gen = pd.DataFrame({
    'time': wind_data['time'],
    'id': 2,
    'bus': 2,
    'pmax': 250,
    'pmin': 0,
    'gencost': 9
})

# Combine wind and solar generator data
gen_time_series = pd.concat([wind_gen, solar_gen], ignore_index=True)

# Create demand time series DataFrame
demand_time_series = pd.DataFrame({
    'time': demand_data['time'],
    'bus': 3,
    'pd': demand_data['load'],
})

# Ensure 'time' columns are datetime
gen_time_series['time'] = pd.to_datetime(gen_time_series['time'])
demand_time_series['time'] = pd.to_datetime(demand_time_series['time'])

# Load branch and bus data
branch = pd.read_csv(datadir + "branch.csv")
bus = pd.read_csv(datadir + "bus.csv")

# Rename all columns to lowercase
for df in [branch, bus]:
    df.columns = df.columns.str.lower()

# Create generator and line IDs
branch['id'] = np.arange(1, len(branch) + 1)

# Susceptance of each line
branch['sus'] = 1 / branch['x']

# %%
# 2.1 Add Storage
# ===========================

# Storage parameters
storage_id = 3  # Next available generator ID
storage_bus = 3  # Bus where storage is connected
Pmax_storage = 100  # Maximum charging/discharging power (MW)
E_max = 500  # Maximum energy capacity (MWh)
eta = 0.95  # Storage efficiency (95%)
storage_cost = 0  # Operational cost (adjust as needed)

# Create storage generator DataFrame
storage_gen = pd.DataFrame({
    'time': pd.to_datetime(wind_data['time']),
    'id': storage_id,
    'bus': storage_bus,
    'pmax': Pmax_storage,
    'pmin': -Pmax_storage,
    'gencost': storage_cost
})

# Add storage to gen_time_series
gen_time_series = pd.concat([gen_time_series, storage_gen], ignore_index=True)

# %%
# 3. Function
# ===========================

def dcopf_multi_period(gen_time_series, branch, bus, demand_time_series):
    # Create the optimization model
    DCOPF = pulp.LpProblem("DCOPF_Multi_Period", pulp.LpMinimize)
    
    # Define sets
    time_steps = sorted(gen_time_series['time'].unique())
    G = gen_time_series['id'].unique()  # Set of all generators
    N = bus['bus_i'].values             # Set of all buses

    # Decision variables
    GEN = {}
    THETA = {}
    FLOW = {}
    E = {}
    for t in time_steps:
        for g in G:
            pmin = gen_time_series.loc[
                (gen_time_series['id'] == g) & (gen_time_series['time'] == t),
                'pmin'].values[0]
            pmax = gen_time_series.loc[
                (gen_time_series['id'] == g) & (gen_time_series['time'] == t),
                'pmax'].values[0]
            GEN[g, t] = pulp.LpVariable(f"GEN_{g}_{t}",
                                        lowBound=pmin,
                                        upBound=pmax)
        for i in N:
            THETA[i, t] = pulp.LpVariable(f"THETA_{i}_{t}", lowBound=None)
        for idx, row in branch.iterrows():
            i = row['fbus']
            j = row['tbus']
            FLOW[i, j, t] = pulp.LpVariable(f"FLOW_{i}_{j}_{t}",
                                            lowBound=None)
        # Storage SoC variable
        E[t] = pulp.LpVariable(f"E_{t}", lowBound=0, upBound=E_max)
    
    # Initial SoC (can be set as a parameter)
    E_init = 0.5 * E_max
    DCOPF += E[time_steps[0]] == E_init, "Initial_SoC"

    # Objective function: Minimize generation costs
    DCOPF += pulp.lpSum(
        gen_time_series.loc[
            (gen_time_series['id'] == g) & (gen_time_series['time'] == t),
            'gencost'].values[0] * GEN[g, t]
        for t in time_steps for g in G
    ), "Total Generation Cost"

    # Constraints
    for t in time_steps:
        # Slack bus angle
        DCOPF += THETA[1, t] == 0, f"Slack_Bus_Angle_Time_{t}"
        
        # Power balance at each bus
        for i in N:
            gen_sum = pulp.lpSum(
                GEN[g, t]
                for g in gen_time_series[
                    (gen_time_series['bus'] == i) &
                    (gen_time_series['time'] == t)
                ]['id'].unique()
            )
            demand = demand_time_series.loc[
                (demand_time_series['bus'] == i) &
                (demand_time_series['time'] == t), 'pd']
            demand_value = demand.values[0] if not demand.empty else 0

            flow_out = pulp.lpSum(
                FLOW[i, j, t]
                for (i_, j, t_) in FLOW if i_ == i and t_ == t
            )
            flow_in = pulp.lpSum(
                FLOW[j, i, t]
                for (j, i_, t_) in FLOW if i_ == i and t_ == t
            )

            DCOPF += (
                gen_sum - demand_value + flow_in - flow_out == 0
            ), f"Power_Balance_at_Bus_{i}_Time_{t}"

        # DC power flow equations
        for idx, row in branch.iterrows():
            i = row['fbus']
            j = row['tbus']
            DCOPF += (
                FLOW[i, j, t] == row['sus'] * (THETA[i, t] - THETA[j, t])
            ), f"Flow_Constraint_{i}_{j}_Time_{t}"

        # Flow limits
        for idx, row in branch.iterrows():
            i = row['fbus']
            j = row['tbus']
            rate_a = row['ratea']
            DCOPF += (
                FLOW[i, j, t] <= rate_a
            ), f"Flow_Limit_{i}_{j}_Upper_Time_{t}"
            DCOPF += (
                FLOW[i, j, t] >= -rate_a
            ), f"Flow_Limit_{i}_{j}_Lower_Time_{t}"

    # Storage dynamics and constraints
    for idx, t in enumerate(time_steps[:-1]):
        next_t = time_steps[idx + 1]
        DCOPF += (
            E[next_t] == eta * E[t] + GEN[storage_id, t]
        ), f"Storage_Dynamics_Time_{t}"
    
    # Cyclic condition
    DCOPF += E[time_steps[0]] == E[time_steps[-1]], "Cyclic_Condition"

    # Solve the optimization problem
    DCOPF.solve(pulp.GLPK(msg=True))

    # Check solver status
    status = pulp.LpStatus[DCOPF.status]
    if status != 'Optimal':
        print(f"Solver Status: {status}")
        return None

    # Extract results
    generation = []
    flows = []
    storage_soc = []
    for t in time_steps:
        gen_t = pd.DataFrame({
            'time': t,
            'id': [g for g in G],
            'gen': [pulp.value(GEN[g, t]) for g in G]
        })
        generation.append(gen_t)

        flows_t = pd.DataFrame({
            'time': t,
            'from_bus': [i for (i, j, t_) in FLOW if t_ == t],
            'to_bus': [j for (i, j, t_) in FLOW if t_ == t],
            'flow': [pulp.value(FLOW[i, j, t]) for (i, j, t_) in FLOW if t_ == t]
        })
        flows.append(flows_t)

        storage_soc.append({'time': t, 'E': pulp.value(E[t])})

    # Combine results
    generation = pd.concat(generation, ignore_index=True)
    flows = pd.concat(flows, ignore_index=True)
    storage_soc = pd.DataFrame(storage_soc)

    # Return the solution
    return {
        'generation': generation,
        'flows': flows,
        'storage_soc': storage_soc,
        'cost': pulp.value(DCOPF.objective),
        'status': status
    }

# %%
# 4. Solve
# ===========================

# Define the time steps to consider (e.g., a single day)
selected_day = '2023-01-01'
gen_time_series_day = gen_time_series[
    gen_time_series['time'].dt.date == pd.to_datetime(selected_day).date()
]
demand_time_series_day = demand_time_series[
    demand_time_series['time'].dt.date == pd.to_datetime(selected_day).date()
]

# Run the multi-period DCOPF
result = dcopf_multi_period(
    gen_time_series_day,
    branch,
    bus,
    demand_time_series_day
)

# Check if result is not None
if result is not None:
    generation = result['generation']
    flows = result['flows']
    storage_soc = result['storage_soc']
    total_cost = result['cost']

    # Display results
    print(f"Total Generation Cost: {total_cost}")
    print("\nGeneration:")
    print(generation)
    print("\nStorage State of Charge:")
    print(storage_soc)
else:
    print("Optimization did not find an optimal solution.")

# %%
# 5. Visualization
# ===========================

# Plot Storage State of Charge over time
plt.figure(figsize=(12, 6))
plt.plot(storage_soc['time'], storage_soc['E'], marker='o')
plt.xlabel('Time')
plt.ylabel('Energy Stored (MWh)')
plt.title('Storage State of Charge Over Time')
plt.grid(True)
plt.show()

# Plot Generation Mix Over Time
# Merge generation data with generator types
gen_info = pd.DataFrame({
    'id': [1, 2, 3],
    'type': ['Wind', 'Solar', 'Storage']
})

# Merge with generation data
generation_with_types = pd.merge(generation, gen_info, on='id')

# Pivot the DataFrame
generation_pivot = generation_with_types.pivot_table(
    index='time',
    columns='type',
    values='gen',
    aggfunc='sum'
).reset_index()

generation_pivot.fillna(0, inplace=True)

# Plot stacked area chart
plt.figure(figsize=(12, 6))
plt.stackplot(
    generation_pivot['time'],
    [generation_pivot['Wind'], generation_pivot['Solar'], generation_pivot['Storage']],
    labels=['Wind', 'Solar', 'Storage']
)
plt.xlabel('Time')
plt.ylabel('Generation (MW)')
plt.title('Generation Mix Over Time')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot Line Flows Over Time with Limits
flows_with_limits = pd.merge(
    flows,
    branch[['fbus', 'tbus', 'ratea']],
    left_on=['from_bus', 'to_bus'],
    right_on=['fbus', 'tbus'],
    how='left'
)

unique_lines = flows_with_limits[['from_bus', 'to_bus']].drop_duplicates()

plt.figure(figsize=(12, 6))

for idx, row in unique_lines.iterrows():
    line_flows = flows_with_limits[
        (flows_with_limits['from_bus'] == row['from_bus']) &
        (flows_with_limits['to_bus'] == row['to_bus'])
    ]
    plt.plot(
        line_flows['time'],
        line_flows['flow'],
        label=f"Line {row['from_bus']}->{row['to_bus']}"
    )

# Assuming the line limit is the same for all lines
line_limit = branch['ratea'].iloc[0]

# Plot horizontal lines for the upper and lower limits
plt.axhline(y=line_limit, color='red', linestyle='--', label='Line Limit')
plt.axhline(y=-line_limit, color='red', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Flow (MW)')
plt.title('Line Flows Over Time with Limits')
plt.legend()
plt.grid(True)
plt.show()
