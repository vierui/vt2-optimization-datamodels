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

import numpy as np
import pandas as pd

# Prepare Data for Network Visualization
G = nx.DiGraph()

# Add nodes (buses)
G.add_nodes_from(bus['bus_i'])

# Add edges (branches)
flows = result['flows']
edge_labels = {(i, j): f"{flow:.2f} MW" for (i, j), flow in flows.items()}
G.add_edges_from(flows.keys())

# Position nodes using a layout
pos = nx.circular_layout(G)  # You can choose different layouts

# Node sizes and colors
bus_generation = generation.groupby('node')['gen'].sum().to_dict()
bus_load = bus.set_index('bus_i')['pd'].to_dict()
max_node_size = 1000
min_node_size = 300

# Create Series with all nodes as index
nodes = list(G.nodes())
bus_generation_series = pd.Series(bus_generation, index=nodes).fillna(0)
bus_load_series = pd.Series(bus_load, index=nodes).fillna(0)

# Compute net generation
net_gen_series = bus_generation_series - bus_load_series
net_gen_abs = net_gen_series.abs()
max_net_gen = net_gen_abs.max() if net_gen_abs.max() != 0 else 1  # Avoid division by zero

# Initialize lists for node sizes and colors
node_sizes = []
node_colors = []
for node in nodes:
    net_gen = net_gen_series[node]
    # Scale node size
    size = min_node_size + (max_node_size - min_node_size) * abs(net_gen) / max_net_gen
    node_sizes.append(size)
    # Assign node color
    if net_gen > 0:
        node_colors.append('green')  # Net generator
    elif net_gen < 0:
        node_colors.append('red')    # Net load
    else:
        node_colors.append('grey')   # Neutral

# Edge widths and colors
max_edge_width = 5
min_edge_width = 1
flow_values = [abs(flow) for flow in flows.values()]
max_flow = max(flow_values) if flow_values else 1  # Avoid division by zero
edge_widths = []
edge_colors = []
for u, v in G.edges():
    flow = abs(flows.get((u, v), 0))
    width = min_edge_width + (max_edge_width - min_edge_width) * flow / max_flow
    edge_widths.append(width)
    edge_colors.append('blue' if flows.get((u, v), 0) >= 0 else 'red')

# Draw nodes and edges
plt.figure(figsize=(12, 8), dpi=100)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrowsize=20, arrowstyle='-|>')

# Labels
nx.draw_networkx_labels(G, pos, font_size=14, font_color='white', font_weight='bold')
threshold = 10  # Only label edges with significant flow
significant_edges = {k: v for k, v in edge_labels.items() if abs(flows[k]) >= threshold}
nx.draw_networkx_edge_labels(G, pos, edge_labels=significant_edges, font_size=12)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Net Generator', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Net Load', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='grey', markersize=10),
    Line2D([0], [0], color='blue', lw=2, label='Flow Direction (Positive)'),
    Line2D([0], [0], color='red', lw=2, label='Flow Direction (Negative)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.title(f'Network Visualization at {selected_time}', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()


# %%
