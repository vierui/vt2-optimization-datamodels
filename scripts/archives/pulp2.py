# %%
# 1. Initialization
# ===========================

import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

# Working directory
datadir = "/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/vt1-energy-investment-model/data/processed/"
# %%
# 2. Data
# ===========================

demand_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/vt1-energy-investment-model/data/raw/data-load-becc.csv',
    delimiter=';',
    names=['time', 'load'],
    parse_dates=['time'],
    dayfirst=True,
    header=0
)

demand_data['load'] = pd.to_numeric(demand_data['load']) * 100

wind_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/vt1-energy-investment-model/data/raw/wind-sion-2023.csv', skiprows=3, parse_dates=['time'], delimiter=',')

# Load into DataFrames
wind_gen = pd.DataFrame({
    'time': wind_data['time'],
    'id': 1,  # Assuming wind generator has ID 1
    'bus': 1,  # Assuming wind generator is connected to bus 1
    'pmax': wind_data['electricity'],  # Time-varying maximum generation
    'pmin': 0,  # Assuming minimum generation is 0
    'gencost': 5  # Cost coefficient for wind generator
})
solar_gen = pd.DataFrame({
    'time': pd.to_datetime(wind_data['time']),  # Ensure the time column matches
    'id': 2,  # Assuming solar generator has ID 2
    'bus': 2,  # Assuming solar generator is connected to bus 2
    'pmax': 250,  # Constant maximum generation
    'pmin': 0,
    'gencost': 9  # Cost coefficient for solar generator
})
# Define storage parameters
storage_id = 3  # Next available ID
storage_bus = 3  # Bus where storage is connected
Pmax_storage = 100  # Maximum charging/discharging power in MW
storage_cost = 0  # Cost coefficient (adjust as needed)

# Create storage generator DataFrame
storage_gen = pd.DataFrame({
    'time': pd.to_datetime(wind_data['time']),
    'id': storage_id,
    'bus': storage_bus,
    'pmax': Pmax_storage,
    'pmin': -Pmax_storage,
    'gencost': storage_cost
})

# Combine wind and solar generator data
gen_time_series = pd.concat([wind_gen, solar_gen, storage_gen], ignore_index=True)

# Create a DataFrame for time-varying demand at bus 3
demand_time_series = pd.DataFrame({
    'time': demand_data['time'],
    'bus': 3,  # Demand is at bus 3
    'pd': demand_data['load'],
})

# Ensure 'time' column is datetime
demand_time_series['time'] = pd.to_datetime(demand_time_series['time'])

# Load branch and bus data
branch = pd.read_csv(datadir + "branch.csv")
bus = pd.read_csv(datadir + "bus.csv")

# Rename all columns to lowercase
for df in [branch, bus]:
    df.columns = df.columns.str.lower()

# Create generator and line IDs
branch['id'] = np.arange(1, len(branch) + 1)

# Susceptance of each line based on reactance
# Assuming reactance >> resistance, susceptance ≈ 1 / reactance
branch['sus'] = 1 / branch['x']

# Display the bus DataFrame as an example
# print(bus)
# print(gen_time_series)
# print(branch)

# %%
# 3. Function
# ===========================

def dcopf_multi_period(gen_time_series, branch, bus, demand_time_series, time_steps, storage_id, E_max, eta):
    # Create the optimization model
    DCOPF = pulp.LpProblem("DCOPF_Multi_Period", pulp.LpMinimize)

    # Define sets
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
                (gen_time_series['id'] == g) & (gen_time_series['time'] == t), 'pmin'
            ].values[0]
            pmax = gen_time_series.loc[
                (gen_time_series['id'] == g) & (gen_time_series['time'] == t), 'pmax'
            ].values[0]
            GEN[g, t] = pulp.LpVariable(f"GEN_{g}_{t}", lowBound=pmin, upBound=pmax)

        for i in N:
            THETA[i, t] = pulp.LpVariable(f"THETA_{i}_{t}", lowBound=None)

        for idx, row in branch.iterrows():
            i_ = row['fbus']
            j_ = row['tbus']
            FLOW[i_, j_, t] = pulp.LpVariable(f"FLOW_{i_}_{j_}_{t}", lowBound=None)

        # Storage SoC variable
        E[t] = pulp.LpVariable(f"E_{t}", lowBound=0, upBound=E_max)

    # Objective function
    DCOPF += pulp.lpSum(
        gen_time_series.loc[
            (gen_time_series['id'] == g) & (gen_time_series['time'] == t), 'gencost'
        ].values[0] * GEN[g, t]
        for t in time_steps for g in G
    ), "Total Generation Cost"

    # Constraints
    try:
        # Storage Dynamics
        for idx, t in enumerate(time_steps[:-1]):
            next_t = time_steps[idx + 1]
            DCOPF += E[next_t] == eta * E[t] + GEN[storage_id, t], f"Storage_Dynamics_{t}"

        # Cyclic condition
        DCOPF += E[time_steps[0]] == E[time_steps[-1]], "Cyclic_Condition"

        # Slack bus angle
        for t in time_steps:
            DCOPF += THETA[1, t] == 0, f"Slack_Bus_Angle_Time_{t}"

        # Power balance constraints
        for t in time_steps:
            for i in N:
                gen_ids = gen_time_series[
                    (gen_time_series['bus'] == i) & (gen_time_series['time'] == t)
                ]['id']
                gen_sum = pulp.lpSum(GEN[g, t] for g in gen_ids)

                demand = demand_time_series.loc[
                    (demand_time_series['bus'] == i) & (demand_time_series['time'] == t), 'pd'
                ]
                demand_value = demand.values[0] if not demand.empty else 0

                flow_in = pulp.lpSum(
                    FLOW[k, i, t] for (k, i_, t_) in FLOW.keys() if i_ == i and t_ == t
                )
                flow_out = pulp.lpSum(
                    FLOW[i, k, t] for (i_, k, t_) in FLOW.keys() if i_ == i and t_ == t
                )

                DCOPF += (
                    gen_sum - demand_value + flow_in - flow_out == 0,
                    f"Power_Balance_at_Bus_{i}_Time_{t}"
                )

        # DC power flow equations
        for t in time_steps:
            for idx, row in branch.iterrows():
                i_ = row['fbus']
                j_ = row['tbus']
                DCOPF += FLOW[i_, j_, t] == row['sus'] * (THETA[i_, t] - THETA[j_, t]), f"Flow_Constraint_{i_}_{j_}_Time_{t}"

        # Flow limits
        for t in time_steps:
            for idx, row in branch.iterrows():
                i_ = row['fbus']
                j_ = row['tbus']
                rate_a = row['ratea']
                DCOPF += FLOW[i_, j_, t] <= rate_a, f"Flow_Limit_{i_}_{j_}_Upper_Time_{t}"
                DCOPF += FLOW[i_, j_, t] >= -rate_a, f"Flow_Limit_{i_}_{j_}_Lower_Time_{t}"

    except Exception as e:
        print(f"Error adding constraints: {e}")
        return None

    # Solve the optimization problem
    DCOPF.solve(pulp.GLPK(msg=True))

    # Check if the problem was solved optimally
    if pulp.LpStatus[DCOPF.status] != 'Optimal':
        print(f"Optimization failed: {pulp.LpStatus[DCOPF.status]}")
        return None

    # Extract results
    # ... (as before)

    return {
        'generation': generation,
        'storage_soc': storage_soc,
        'flows': flows,
        'cost': pulp.value(DCOPF.objective),
        'status': pulp.LpStatus[DCOPF.status]
    }

# %%
# 4. Solve
# ===========================

# Set storage parameters
E_max = 500  # Maximum storage capacity in MWh (adjust as needed)
eta = 1.0    # Storage efficiency (assuming perfect efficiency for simplicity)

# Get the unique time steps sorted in chronological order
selected_day = '2023-01-01'  # Example date
gen_time_series = gen_time_series[gen_time_series['time'].dt.date == pd.to_datetime(selected_day).date()]
demand_time_series = demand_time_series[demand_time_series['time'].dt.date == pd.to_datetime(selected_day).date()]
time_steps = sorted(gen_time_series['time'].unique())

# Run the DCOPF multi-period function
result = dcopf_multi_period(gen_time_series, branch, bus, demand_time_series, time_steps, storage_id, E_max, eta)

# Check optimization status
if result is not None and result['status'] == 'Optimal':
    generation_over_time = result['generation']
    storage_soc_over_time = result['storage_soc']
    flows_over_time = result['flows']
    total_cost = result['cost']
else:
    print("Optimization failed or no optimal solution found.")

selected_day = '2023-01-01'  # Example date
gen_time_series = gen_time_series[gen_time_series['time'].dt.date == pd.to_datetime(selected_day).date()]

# Get the unique time steps sorted in chronological order
time_steps = sorted(gen_time_series['time'].unique())

# Initialize lists to store results
generation_results = []
flow_results = []
cost_results = []

for t in time_steps:
    # Filter generator data for time t
    gen_t = gen_time_series[gen_time_series['time'] == t]
    
    # Filter demand data for time t
    demand_t = demand_time_series[demand_time_series['time'] == t]
    
    # Run the DCOPF for time t
    result_t = dcopf(gen_t, branch, bus, demand_t)
    
    # Check optimization status
    if result_t['status'] == 'Optimal':
        # Add time to the generation DataFrame
        result_t['generation']['time'] = t
        generation_results.append(result_t['generation'])
        
        # Store flows with time
        flows_t = pd.DataFrame({
            'time': t,
            'from_bus': [k[0] for k in result_t['flows'].keys()],
            'to_bus': [k[1] for k in result_t['flows'].keys()],
            'flow': list(result_t['flows'].values())
        })
        flow_results.append(flows_t)
        
        # Store total cost
        cost_results.append({'time': t, 'cost': result_t['cost']})
    else:
        print(f"Optimization failed at time {t}: {result_t['status']}")

#Combine generations results, flow and costs results
generation_over_time = pd.concat(generation_results, ignore_index=True)
flows_over_time = pd.concat(flow_results, ignore_index=True)
costs_over_time = pd.DataFrame(cost_results)

# Generated LP File
DCOPF.writeLP("DCOPF_Multi_Period.lp")

# %%
# 5. Visualisation
# ===========================

# 1. Pivot the Generation Data
gen_pivot = generation_over_time.pivot(index='time', columns='id', values='gen')
gen_pivot = gen_pivot.sort_index()

# 2. Prepare the Total Demand Series
total_demand = demand_time_series.set_index('time')['pd'].reindex(gen_pivot.index)

# Map generator IDs to names (optional)
gen_pivot.rename(columns={
    1: 'Wind Generator',
    2: 'Solar Generator',
    # Add more mappings if necessary
}, inplace=True)

# 4. Plot the Stacked Area Chart
fig, ax = plt.subplots(figsize=(12, 6))

# Plot stacked generation
ax.stackplot(gen_pivot.index, gen_pivot.T, labels=gen_pivot.columns, alpha=0.8)

# Plot total demand
ax.plot(total_demand.index, total_demand.values, label='Total Demand', color='black', linewidth=2)

# Customize the plot
ax.set_xlabel('Time')
ax.set_ylabel('Power (MW)')
ax.set_title('Generation Distribution Over Time vs. Demand')
ax.legend(loc='upper left', title='Generators')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# Map generator IDs to names
gen_info = pd.DataFrame({
    'id': [1, 2, 3],
    'type': ['Wind Generator', 'Solar Generator', 'Storage']
})

# Merge generation data with generator types
generation_with_types = pd.merge(generation_over_time, gen_info, on='id')

# Pivot the DataFrame to get generation per type over time
gen_pivot = generation_with_types.pivot_table(index='time', columns='type', values='gen', aggfunc='sum').reset_index()
gen_pivot = gen_pivot.sort_values('time')

# Prepare the Total Demand Series
total_demand = demand_time_series.set_index('time')['pd'].reindex(gen_pivot['time'])

# Plot the Stacked Area Chart
fig, ax = plt.subplots(figsize=(12, 6))

# Plot stacked generation
ax.stackplot(gen_pivot['time'], [gen_pivot[col] for col in gen_info['type']], labels=gen_info['type'], alpha=0.8)

# Plot total demand
ax.plot(total_demand.index, total_demand.values, label='Total Demand', color='black', linewidth=2)

# Customize the plot
ax.set_xlabel('Time')
ax.set_ylabel('Power (MW)')
ax.set_title('Generation Distribution Over Time vs. Demand')
ax.legend(loc='upper left', title='Generators')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Plot Storage SoC over time
plt.figure(figsize=(12, 6))
plt.plot(storage_soc_over_time['time'], storage_soc_over_time['E'], marker='o')
plt.xlabel('Time')
plt.ylabel('Energy Stored (MWh)')
plt.title('Storage State of Charge Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Plot Storage SoC over time
plt.figure(figsize=(12, 6))
plt.plot(storage_soc_over_time['time'], storage_soc_over_time['E'], marker='o')
plt.xlabel('Time')
plt.ylabel('Energy Stored (MWh)')
plt.title('Storage State of Charge Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Merge flows with branch data to get line limits
flows_with_limits = pd.merge(
    flows_over_time,
    branch[['fbus', 'tbus', 'ratea']],
    left_on=['from_bus', 'to_bus'],
    right_on=['fbus', 'tbus'],
    how='left'
)

# Plot flows over time for each line
unique_lines = flows_with_limits[['from_bus', 'to_bus']].drop_duplicates()

plt.figure(figsize=(12, 6))

# Iterate over each unique line to plot its flow
for idx, row in unique_lines.iterrows():
    # Filter flows for the current line
    line_flows = flows_with_limits[
        (flows_with_limits['from_bus'] == row['from_bus']) & 
        (flows_with_limits['to_bus'] == row['to_bus'])
    ]
    # Plot the line flows over time
    plt.plot(
        line_flows['time'], 
        line_flows['flow'], 
        label=f"Line {row['from_bus']}->{row['to_bus']}"
    )

# Assuming the line limit is the same for all lines
line_limit = branch['ratea'].iloc[0]  # Get the limit from the first line

# Plot horizontal lines for the upper and lower limits
plt.axhline(y=line_limit, color='red', linestyle='--', label='Line Limit')
plt.axhline(y=-line_limit, color='red', linestyle='--')

# Set plot labels and title
plt.xlabel('Time')
plt.ylabel('Flow (MW)')
plt.title('Line Flows Over Time with Limits')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Choose a specific time to visualize
time_to_visualize = time_steps[0]  # Change as needed

# Get flows at that time
flows_at_time = flows_over_time[flows_over_time['time'] == time_to_visualize]

# Add flow data to edges
for idx, row in flows_at_time.iterrows():
    G[row['from_bus']][row['to_bus']]['flow'] = row['flow']

# Get edge widths based on flow
edge_flows = [abs(G[u][v]['flow']) for u, v in G.edges()]
max_flow = max(edge_flows) if edge_flows else 1
edge_widths = [5 * flow / max_flow for flow in edge_flows]  # Scale widths

# Plot the network
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_labels(G, pos)

# Draw edges with widths representing flows
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=edge_widths)

# Add edge labels for flows
edge_labels = {(u, v): f"{G[u][v]['flow']:.1f} MW" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title(f"Network Flows at {time_to_visualize}")
plt.axis('off')
plt.show()

# %%
