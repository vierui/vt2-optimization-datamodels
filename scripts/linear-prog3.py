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

# Load the wind data into
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

# Test
selected_time = pd.Timestamp('2023-01-01 12:00')
gen_t = gen_time_series[gen_time_series['time'] == selected_time]


# Load branch and bus data
branch = pd.read_csv(datadir + "branch.csv")
bus = pd.read_csv(datadir + "bus.csv")

# Rename all columns to lowercase
for df in [branch, bus]:
    df.columns = df.columns.str.lower()

# Create generator and line IDs
branch['id'] = np.arange(1, len(branch) + 1)

# Add reverse direction rows in branch DataFrame
# branch2 = branch.copy()
# branch2['f'] = branch2['fbus']
# branch2['fbus'] = branch['tbus']
# branch2['tbus'] = branch2['f']
# branch2 = branch2[branch.columns]  # Ensure branch2 has the same column order as branch
# branch = pd.concat([branch, branch2], ignore_index=True)

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

# Transport problem (no physical limitation, no lines limitations)
# def transport(gen, branch, gencost, bus):
#     # Create the optimization model
#     Transport = pulp.LpProblem("TransportFlowProblem", pulp.LpMinimize)
    
#     # Define sets
#     G = gen['id'].values  # Set of all generators
#     N = bus['bus_i'].values  # Set of all nodes

#     # Decision variables
#     GEN = {g: pulp.LpVariable(f"GEN_{g}", lowBound=0) for g in G}  # Generation variable (GEN >= 0)
    
#     FLOW = {(i, j): pulp.LpVariable(f"FLOW_{i}_{j}", lowBound=None) for i in N for j in N}  # Flow variable (can be positive or negative)
    
#    # Objective function: Minimize generation costs
#     Transport += pulp.lpSum(gencost.loc[idx, 'x1'] * GEN[g] for idx, g in enumerate(G)), "Total Generation Cost"
    
#     # Supply/demand balance constraints
#     for i in N:
#         Transport += (
#             pulp.lpSum(GEN[g] for g in gen[gen['bus'] == i]['id']) 
#             - bus.loc[bus['bus_i'] == i, 'pd'].values[0]
#             == pulp.lpSum(FLOW[i, j] for j in branch[branch['tbus'] == i]['fbus'].values)
#         ), f"Balance_at_Node_{i}"
    
#     # Max generation constraints
#     for g in G:
#         Transport += GEN[g] <= gen.loc[gen['id'] == g, 'pmax'].values[0], f"MaxGen_{g}"
    
#     # Flow constraints on each branch
#     for l in range(len(branch)):
#         fbus = branch.iloc[l]['fbus']
#         tbus = branch.iloc[l]['tbus']
#         rate_a = branch.iloc[l]['ratea']
        
#         Transport += FLOW[fbus, tbus] <= rate_a, f"FlowLimit_{fbus}_{tbus}"
    
#     # Anti-symmetric flow constraints
#     for i in N:
#         for j in N:
#             Transport += FLOW[i, j] == -FLOW[j, i], f"AntiSymmetricFlow_{i}_{j}"
    
#     # Solve the optimization problem
#     Transport.solve()
    
#     # Extract the results
#     generation = pd.DataFrame({
#         'id': gen['id'],
#         'node': gen['bus'],
#         'gen': [pulp.value(GEN[g]) for g in G]
#     })
    
#     flows = {(i, j): pulp.value(FLOW[i, j]) for i in N for j in N}

#     # Return the solution and objective as a dictionary (similar to a named tuple)
#     return {
#         'generation': generation,
#         'flows': flows,
#         'cost': pulp.value(Transport.objective),
#         'status': pulp.LpStatus[Transport.status]
#     }

# Optimal Power Flow Problem
def dcopf(gen_t, branch, bus):
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
        
        # Demand at bus i
        demand = bus.loc[bus['bus_i'] == i, 'pd'].values[0]
        
        # Net flow out of bus i
        flow_out = pulp.lpSum(FLOW[i, j] for (i_, j) in FLOW if i_ == i)
        flow_in = pulp.lpSum(FLOW[j, i] for (j, i_) in FLOW if i_ == i)
        
        # Power balance constraint
        DCOPF += gen_sum - demand + flow_in - flow_out == 0, f"Power_Balance_at_Bus_{i}"
    
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

result = dcopf(gen_t, branch, bus)

# Check optimization status
if result['status'] == 'Optimal':
    # Add time to the generation DataFrame
    result['generation']['time'] = selected_time
    # Display the generation results
    print("Generation at time", selected_time)
    print(result['generation'])
else:
    print(f"Optimization failed at time {selected_time}: {result['status']}")

# Display the generation results
print("Generation:")
print(result['generation'])

print("Total Generation Cost:", result['cost'])

# Display the flow results
# print("\nFlows:")
# for (i, j), flow_value in result['flows'].items():
#     print(f"Flow from {i} to {j}: {flow_value}")

# %%
# 5. Visualisation
# ===========================

# Extract generation data
generation = result['generation']

# Plot generation dispatch
plt.figure(figsize=(8, 6))
plt.bar(generation['id'], generation['gen'], color='skyblue')
plt.xlabel('Generator ID')
plt.ylabel('Generation Output (MW)')
plt.title(f'Generation Dispatch at {selected_time}')
plt.xticks(generation['id'])
plt.show()

# Prepare Data for Network Visualization
G = nx.DiGraph()

# Add nodes (buses)
for bus_id in bus['bus_i']:
    G.add_node(bus_id)

# Add edges (branches) with flow as edge attribute
flows = result['flows']
edge_labels = {}
for (i, j), flow in flows.items():
    G.add_edge(i, j, weight=abs(flow))
    edge_labels[(i, j)] = f"{flow:.2f} MW"

# Position nodes using a layout
pos = nx.spring_layout(G, seed=42)

# Node sizes based on generation
bus_generation = generation.groupby('node')['gen'].sum().to_dict()
bus_load = bus.set_index('bus_i')['pd'].to_dict()
node_sizes = []
for node in G.nodes():
    gen = bus_generation.get(node, 0)
    load = bus_load.get(node, 0)
    size = 300 + (gen - load) * 5  # Adjust the multiplier for visualization
    node_sizes.append(size)

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgreen')
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', arrows=True)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title(f'Network Visualization at {selected_time}')
plt.axis('off')
plt.show()

# %%
