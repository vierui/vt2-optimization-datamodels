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
datadir = "/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/processed/"

# %%
# 2. Data
# ===========================

demand_data = pd.read_csv(
    '/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/vt1-energy-investment-model/data/raw/data-load-becc.csv',
    delimiter=';',
    names=['time', 'load'],  # Specify the column names
    parse_dates=['time'],
    dayfirst=True,
    header=0
)

demand_data['load'] = pd.to_numeric(demand_data['load']) * 100

wind_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/raw/wind-sion-2023.csv', skiprows=3, parse_dates=['time'], delimiter=',')

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
# Combine wind and solar generator data
gen_time_series = pd.concat([wind_gen, solar_gen], ignore_index=True)

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

# Optimal Power Flow Problem
def dcopf(gen_t, branch, bus, demand_t):
    # Create the optimization model
    DCOPF = pulp.LpProblem("DCOPF_Problem", pulp.LpMinimize)
    
     # Define sets
    G = gen_t['id'].unique()  # Set of generators at time t
    N = bus['bus_i'].values   # Set of buses

    # Define base MVA for power flow calculations (assuming first value for all)
    baseMVA = 1

    # Decision variables
    GEN = {g: pulp.LpVariable(f"GEN_{g}",
                              lowBound=gen_t.loc[gen_t['id'] == g, 'pmin'].values[0],
                              upBound=gen_t.loc[gen_t['id'] == g, 'pmax'].values[0])
           for g in G}
    THETA = {i: pulp.LpVariable(f"THETA_{i}", lowBound=None) for i in N}
    FLOW = {}
    for idx, row in branch.iterrows():
        i = row['fbus']
        j = row['tbus']
        FLOW[i, j] = pulp.LpVariable(f"FLOW_{i}_{j}", lowBound=None)
    
    # Set slack bus with reference angle = 0 (assuming bus 1 is the slack bus)
    DCOPF += THETA[1] == 0

    # Objective function: Minimize generation costs
    DCOPF += pulp.lpSum(gen_t.loc[gen_t['id'] == g, 'gencost'].values[0] * GEN[g] for g in G), "Total Generation Cost"
    
    # Power balance constraints at each bus
    for i in N:
        # Sum of generation at bus i
        gen_sum = pulp.lpSum(GEN[g] for g in gen_t[gen_t['bus'] == i]['id'])
        
        # Demand at bus i from demand_t
        demand = demand_t.loc[demand_t['bus'] == i, 'pd']
        demand_value = demand.values[0] if not demand.empty else 0

        # Net flow out of bus i
        flow_out = pulp.lpSum(FLOW[i, j] for (i_, j) in FLOW if i_ == i)
        flow_in = pulp.lpSum(FLOW[j, i] for (j, i_) in FLOW if i_ == i)
        
        # Power balance constraint
        DCOPF += gen_sum - demand_value + flow_in - flow_out == 0, f"Power_Balance_at_Bus_{i}"
    
    # DC power flow equations
    for idx, row in branch.iterrows():
        i = row['fbus']
        j = row['tbus']
        DCOPF += FLOW[i, j] == row['sus'] * (THETA[i] - THETA[j]), f"Flow_Constraint_{i}_{j}"
    
    # Flow limits
    for idx, row in branch.iterrows():
        i = row['fbus']
        j = row['tbus']
        rate_a = row['ratea']
        DCOPF += FLOW[i, j] <= rate_a, f"Flow_Limit_{i}_{j}_Upper"
        DCOPF += FLOW[i, j] >= -rate_a, f"Flow_Limit_{i}_{j}_Lower"
    
    # Solve the optimization problem using GLPK
    DCOPF.solve(pulp.GLPK(msg=True))

    # Extracting output variables after solving
    generation = pd.DataFrame({
        'id': gen_t['id'].values,
        'node': gen_t['bus'].values,
        'gen': [pulp.value(GEN[g]) for g in G]
    })

    angles = {i: pulp.value(THETA[i]) for i in N}
    
    # Extract flows after solving and store them in a dictionnary
    flows = {(i, j): pulp.value(FLOW[i, j]) for (i, j) in FLOW}
    
    # Extract nodal prices (shadow prices)
    # prices = {}
    # for i in N:
    #     constraint_name = f"Power_Balance_at_Bus_{i}"
    #     dual = DCOPF.constraints[constraint_name].pi  # Dual value (shadow price)
    #     prices[i] = dual
    
    # Return the solution and objective as a dictionary
    return {
        'generation': generation,
        'angles': angles,
        'flows': flows,
        # 'prices': prices_df,
        'cost': pulp.value(DCOPF.objective),
        'status': pulp.LpStatus[DCOPF.status]
    }

# %%
# 4. Solve
# ===========================
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

# Display the flow results
# print("\nFlows:")
# for (i, j), flow_value in result['flows'].items():
#     print(f"Flow from {i} to {j}: {flow_value}")


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
# Merge flows with branch data to get line limits
flows_with_limits = pd.merge(
    flows_over_time,
    branch[['fbus', 'tbus', 'ratea']],
    left_on=['from_bus', 'to_bus'],
    right_on=['fbus', 'tbus'],
    how='left'
)

# Calculate absolute flow and compare with limits
flows_with_limits['abs_flow'] = flows_with_limits['flow'].abs()
flows_with_limits['within_limits'] = flows_with_limits['abs_flow'] <= flows_with_limits['ratea']

# Identify any flows exceeding limits
flows_exceeding_limits = flows_with_limits[~flows_with_limits['within_limits']]

# Print any lines exceeding limits
if not flows_exceeding_limits.empty:
    print("Lines exceeding limits:")
    print(flows_exceeding_limits[['time', 'from_bus', 'to_bus', 'flow', 'ratea']])
else:
    print("All line flows are within limits.")

# Plot flows over time for each line
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

plt.xlabel('Time')
plt.ylabel('Flow (MW)')
plt.title('Line Flows Over Time')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Create a graph
G = nx.DiGraph()

# Add nodes
for idx, row in bus.iterrows():
    G.add_node(row['bus_i'])

# Add edges with attributes
for idx, row in branch.iterrows():
    G.add_edge(row['fbus'], row['tbus'], capacity=row['ratea'])

# Choose a specific time to visualize
time_to_visualize = time_steps[0]  # Change as needed

# Get flows at that time
flows_at_time = flows_over_time[flows_over_time['time'] == time_to_visualize]

# Add flow data to edges
flow_dict = {}
for idx, row in flows_at_time.iterrows():
    G[row['from_bus']][row['to_bus']]['flow'] = row['flow']

# Get positions for the nodes (you might need to define these)
pos = nx.spring_layout(G)  # Or define your own positions

# Get edge colors and widths based on flow
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
