# %%
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

# 1. PARAMETERS

# Load the data
load_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/raw/data-load-becc.csv',
                        sep=';', decimal=',')
load_data['time'] = pd.to_datetime(load_data['time'], format='%d.%m.%y %H:%M')
load_data['load'] = pd.to_numeric(load_data['load'].str.replace(',', '.'))

# Select a specific day
selected_date = '2023-02-01'  # Replace with the date you want to select
mask = load_data['time'].dt.strftime('%Y-%m-%d') == selected_date
selected_day_data = load_data.loc[mask]

if len(selected_day_data) != 24:
    raise ValueError(f"Selected date {selected_date} does not have 24 hours of data.")

# Extract the load values
yearly_load_elec_housing = selected_day_data['load'].values

# Time horizon (hours)
N = len(yearly_load_elec_housing)  # Should be 24

# Network structure: Network with M=4 nodes and 5 lines
M = 4  # M is fixed as a parameter

# Susceptances of the lines (p.u.)
Y12 = 0.3  # Line between nodes 1 and 2
Y13 = 0.7  # Line between nodes 1 and 3
Y14 = 0.5  # Line between nodes 1 and 4
Y24 = 0.1  # Line between nodes 2 and 4
Y34 = 0.4  # Line between nodes 3 and 4

# Admittance matrix Y
Y = np.array([
    [Y12 + Y14 + Y13, -Y12, -Y13, -Y14],
    [-Y12, Y24 + Y12, 0, -Y24],
    [-Y13, 0, Y13 + Y34, -Y34],
    [-Y14, -Y24, -Y34, Y14 + Y24 + Y34]
])

# Generator 1 at node 1 (Gas turbine)
f1 = 10  # Cost per unit of energy
G1_max = 1  # Capacity of G1 (per unit)

# Generator 2 at node 2 (Coal power plant)
f2 = 3  # Cost per unit of energy
G2_max = 0.6  # Capacity of G2 (per unit)

# Define line power flow limits (in p.u.)
P_max_12 = 0.2  # Limit for line 1-2
P_max_13 = 0.5  # Limit for line 1-3
P_max_14 = 0.5  # Limit for line 1-4
P_max_24 = 0.5  # Limit for line 2-4
P_max_34 = 0.5  # Limit for line 3-4


# 2. VARIABLES
dim_x = M * 2 * N  # Total number of variables

# 3. OBJECTIVE FUNCTION
f = np.concatenate([
    f1 * np.ones(N),    # Cost for P1
    f2 * np.ones(N),    # Cost for P2
    np.zeros(6 * N)     # No cost associated with P3, P4, and all Vs
])

# 4. CONSTRAINTS

def index_P(node, k):
    """Returns the index of P for a given node and time k."""
    return node * N + k

def index_V(node, k):
    """Returns the index of V for a given node and time k."""
    return M * N + node * N + k

# 4. EQUALITY CONSTRAINTS
A_eq = lil_matrix((N * (4 + 3), dim_x))  # We'll have 7 constraints per time step
b_eq = np.zeros(N * (4 + 3))

row = 0
for k in range(N):
    # Admittance Matrix Equations: P(k) - Y * V(k) = 0
    for i in range(M):
        # P_i(k) variables
        A_eq[row, index_P(i, k)] = 1
        # -Y[i, :] * V(k) variables
        for j in range(M):
            A_eq[row, index_V(j, k)] -= Y[i, j]
        b_eq[row] = 0
        row += 1
    
    # Load at Node 4: P4(k) = -Load(k)
    A_eq[row, index_P(3, k)] = 1  # Node 4 is index 3 (0-based)
    b_eq[row] = -yearly_load_elec_housing[k]
    row += 1

    # Transit Node at Node 3: P3(k) = 0
    A_eq[row, index_P(2, k)] = 1  # Node 3 is index 2 (0-based)
    b_eq[row] = 0
    row += 1

    # Voltage Angle at Node 1: V1(k) = 0
    A_eq[row, index_V(0, k)] = 1  # Node 1 is index 0 (0-based)
    b_eq[row] = 0
    row += 1

# Convert A_eq to CSR format for efficiency
A_eq = A_eq.tocsr()

# 4. INEQUALITY CONSTRAINTS (Updated)
# Initialize inequality constraints for line power flows and bounds
row = 0
num_constraints = 2 * num_lines * N + 4 * N  # Line flow + bounds for P1, P2, P4
A_ub = lil_matrix((num_constraints, dim_x))
b_ub = np.zeros(num_constraints)

# Line flow constraints
for k in range(N):  # For each time step
    for (i, j, Y_ij, P_max_ij) in [
        (0, 1, Y12, P_max_12),
        (0, 2, Y13, P_max_13),
        (0, 3, Y14, P_max_14),
        (1, 3, Y24, P_max_24),
        (2, 3, Y34, P_max_34),
    ]:
        # Line flow: Y_ij * (V_i - V_j) <= P_max
        A_ub[row, index_V(i, k)] = Y_ij
        A_ub[row, index_V(j, k)] = -Y_ij
        b_ub[row] = P_max_ij
        row += 1

        # Line flow: -Y_ij * (V_i - V_j) <= P_max
        A_ub[row, index_V(i, k)] = -Y_ij
        A_ub[row, index_V(j, k)] = Y_ij
        b_ub[row] = P_max_ij
        row += 1

# Bounds as inequalities for P1, P2, and P4
for k in range(N):
    # P1 (Generator 1)
    A_ub[row, index_P(0, k)] = 1  # P1 <= G1_max
    b_ub[row] = G1_max
    row += 1

    # P2 (Generator 2)
    A_ub[row, index_P(1, k)] = 1  # P2 <= G2_max
    b_ub[row] = G2_max
    row += 1


# Convert A_ub to CSR format for efficiency
A_ub = A_ub.tocsr()

# 5. UPDATED BOUNDS
bounds = []

# For P variables (Generators and Loads)
for node in range(M):
    if node == 0:  # P1 (Generator 1)
        bounds.extend([(0, None) for k in range(N)])  # No upper bound, limited by inequality
    elif node == 1:  # P2 (Generator 2)
        bounds.extend([(0, None) for k in range(N)])  # No upper bound, limited by inequality
    elif node == 2:  # P3 (Node 3, now free)
        bounds.extend([(-np.inf, np.inf) for k in range(N)])  # Free variable
    elif node == 3:  # P4 (Load)
        bounds.extend([(None, None) for k in range(N)])  # Limited by inequality

# For V variables (Voltage Angles)
for node in range(M):
    if node == 0:  # V1 (Reference node)
        bounds.extend([(0, 0) for k in range(N)])  # V1 fixed at 0
    else:  # V2, V3, V4
        bounds.extend([(None, None) for k in range(N)])  # Limited by practical bounds


# %%
# 5. SOLVE
result = linprog(
    c=f,
    A_eq=A_eq,
    b_eq=b_eq,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method='highs',
    options={'disp': True}
)

if result.success:
    print("Optimization was successful.")
else:
    print("Optimization failed.")
    print(result.message)

# %%
# 6. EXTRACT
x_opt = result.x

# Initialize arrays
P1 = np.zeros(N)
P2 = np.zeros(N)
P3 = np.zeros(N)
P4 = np.zeros(N)
V1 = np.zeros(N)
V2 = np.zeros(N)
V3 = np.zeros(N)
V4 = np.zeros(N)

for k in range(N):
    P1[k] = x_opt[index_P(0, k)]
    P2[k] = x_opt[index_P(1, k)]
    P3[k] = x_opt[index_P(2, k)]
    P4[k] = x_opt[index_P(3, k)]
    
    V1[k] = x_opt[index_V(0, k)]
    V2[k] = x_opt[index_V(1, k)]
    V3[k] = x_opt[index_V(2, k)]
    V4[k] = x_opt[index_V(3, k)]

#%%
# 7. PLOT
plt.figure()
plt.plot(P1 + P2, label='Total Generation')
plt.plot(-P4, label='Load')  # P4 is negative load
plt.grid(True)
plt.xlabel('Hours')
plt.ylabel('Power')
plt.legend()
plt.title('Total Generation vs. Load')
plt.show()

plt.figure()
plt.plot(P1, 'b', label='G1 (Gas Turbine)')
plt.plot([G1_max] * N, 'b--', label='G1 Max')
plt.plot(P2, 'r', label='G2 (Coal Power Plant)')
plt.plot([G2_max] * N, 'r--', label='G2 Max')
plt.grid(True)
plt.xlabel('Hours')
plt.ylabel('Generation')
plt.legend()
plt.title('Generation Units Output')
plt.show()

# %%
# Plot all power flows together
plt.figure(figsize=(10, 6))

# Maximum power flow limit (same for all lines)
P_max_all = P_max_12  # Example max value, adjust as needed

# Add each line's flow to the plot
for line, flows in line_flows.items():
    plt.plot(flows, label=f"Line {line} Flow")

# Add max/min limits as dashed lines
plt.axhline(P_max_all, color='r', linestyle='--', label='Max Flow')
plt.axhline(-P_max_all, color='r', linestyle='--', label='Min Flow')

# Add labels and legend
plt.xlabel('Hours')
plt.ylabel('Power Flow (p.u.)')
plt.title('Power Flows on All Lines')
plt.grid(True)
plt.legend()
plt.show()

# %%
# Calculate actual power flows
line_flows = {  # Dictionary to store power flow for each line
    "12": np.zeros(N),
    "13": np.zeros(N),
    "14": np.zeros(N),
    "24": np.zeros(N),
    "34": np.zeros(N),
}

for k in range(N):
    line_flows["12"][k] = Y12 * (V1[k] - V2[k])
    line_flows["13"][k] = Y13 * (V1[k] - V3[k])
    line_flows["14"][k] = Y14 * (V1[k] - V4[k])
    line_flows["24"][k] = Y24 * (V2[k] - V4[k])
    line_flows["34"][k] = Y34 * (V3[k] - V4[k])

# Plot line flows
for line, flows in line_flows.items():
    P_max = {
        "12": P_max_12,
        "13": P_max_13,
        "14": P_max_14,
        "24": P_max_24,
        "34": P_max_34
    }[line]
    plt.figure()
    plt.plot(flows, label=f"Line {line} Flow")
    plt.axhline(P_max, color='r', linestyle='--', label='Max Flow')
    plt.axhline(-P_max, color='r', linestyle='--')
    plt.xlabel('Hours')
    plt.ylabel('Power Flow')
    plt.title(f'Line {line} Power Flow')
    plt.grid(True)
    plt.legend()
    plt.show()