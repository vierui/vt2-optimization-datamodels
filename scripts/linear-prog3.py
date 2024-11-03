# %%
# 1. Initialization
# ===========================

import pandas as pd
import numpy as np
import pulp

# Working directory
datadir = "/Users/ruivieira/Documents/Ecole/6_ZHAW/24-Autumn/power-systems-optimization/Notebooks/opf_data/"

# %%
# 2. Data
# ===========================

# Load CSV files into DataFrames
gen = pd.read_csv(datadir + "gen.csv")
gencost = pd.read_csv(datadir + "gencost.csv")
branch = pd.read_csv(datadir + "branch.csv")
bus = pd.read_csv(datadir + "bus.csv")

# Rename all columns to lowercase
for df in [gen, gencost, branch, bus]:
    df.columns = df.columns.str.lower()

# Create generator and line IDs
gen['id'] = np.arange(1, len(gen) + 1)
gencost['id'] = np.arange(1, len(gencost) + 1)
branch['id'] = np.arange(1, len(branch) + 1)

# Add reverse direction rows in branch DataFrame
branch2 = branch.copy()
branch2['f'] = branch2['fbus']
branch2['fbus'] = branch['tbus']
branch2['tbus'] = branch2['f']
branch2 = branch2[branch.columns]  # Ensure branch2 has the same column order as branch
branch = pd.concat([branch, branch2], ignore_index=True)

# Calculate the susceptance of each line based on reactance
# Assuming reactance >> resistance, susceptance ≈ 1 / reactance
branch['sus'] = 1 / branch['x']

# Display the bus DataFrame as an example
print(bus)
print(gen)
print(gencost)
print(branch)

# %%
# 3. Solver function
# ===========================

def transport(gen, branch, gencost, bus):
    # Create the optimization model
    Transport = pulp.LpProblem("TransportFlowProblem", pulp.LpMinimize)
    
    # Define sets
    G = gen['id'].values  # Set of all generators
    N = bus['bus_i'].values  # Set of all nodes

    # Decision variables
    GEN = {g: pulp.LpVariable(f"GEN_{g}", lowBound=0) for g in G}  # Generation variable (GEN >= 0)
    
    FLOW = {(i, j): pulp.LpVariable(f"FLOW_{i}_{j}", lowBound=None) for i in N for j in N}  # Flow variable (can be positive or negative)
    
    # Objective function: Minimize generation costs
    Transport += pulp.lpSum(gencost.loc[g, 'x1'] * GEN[g] for g in G), "Total Generation Cost"
    
    # Supply/demand balance constraints
    for i in N:
        Transport += (
            pulp.lpSum(GEN[g] for g in gen[gen['bus'] == i]['id']) 
            - bus.loc[bus['bus_i'] == i, 'pd'].values[0]
            == pulp.lpSum(FLOW[i, j] for j in branch[branch['tbus'] == i]['fbus'].values)
        ), f"Balance_at_Node_{i}"
    
    # Max generation constraints
    for g in G:
        Transport += GEN[g] <= gen.loc[gen['id'] == g, 'pmax'].values[0], f"MaxGen_{g}"
    
    # Flow constraints on each branch
    for l in range(len(branch)):
        fbus = branch.iloc[l]['fbus']
        tbus = branch.iloc[l]['tbus']
        rate_a = branch.iloc[l]['ratea']
        
        Transport += FLOW[fbus, tbus] <= rate_a, f"FlowLimit_{fbus}_{tbus}"
    
    # Anti-symmetric flow constraints
    for i in N:
        for j in N:
            Transport += FLOW[i, j] == -FLOW[j, i], f"AntiSymmetricFlow_{i}_{j}"
    
    # Solve the optimization problem
    Transport.solve()
    
    # Extract the results
    generation = pd.DataFrame({
        'id': gen['id'],
        'node': gen['bus'],
        'gen': [pulp.value(GEN[g]) for g in G]
    })
    
    flows = {(i, j): pulp.value(FLOW[i, j]) for i in N for j in N}

    # Return the solution and objective as a dictionary (similar to a named tuple)
    return {
        'generation': generation,
        'flows': flows,
        'cost': pulp.value(Transport.objective),
        'status': pulp.LpStatus[Transport.status]
    }


# %%
# 4. Solve
# ===========================
solution = transport(gen, branch, gencost, bus)
# Display the generation results
print("Generation:")
print(result['generation'])

# Display the flow results
print("\nFlows:")
for (i, j), flow_value in result['flows'].items():
    print(f"Flow from {i} to {j}: {flow_value}")

# Display the total cost of the solution
print("\nTotal Cost:", result['cost'])

# Display the solver status
print("Solver Status:", result['status'])

# %%
