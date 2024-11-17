# %%
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

# %%
# 0. DATA

# Load the demand data
load_data = pd.read_csv('/Users/ruivieira/Documents/Projects/vt_energy_opti/data/raw/data-load-becc.csv',
                        sep=';', decimal=',')
load_data['time'] = pd.to_datetime(load_data['time'], format='%d.%m.%y %H:%M')
load_data['load'] = pd.to_numeric(load_data['load'].str.replace(',', '.'))

g1_max_data = pd.read_csv('/Users/ruivieira/Documents/Projects/vt_energy_opti/data/processed/gen-pv.csv', 
                          sep=';', decimal='.')
g1_max_data['time'] = pd.to_datetime(g1_max_data['time'], format='%d.%m.%y %H:%M')
g1_max_data['electricity'] = pd.to_numeric(g1_max_data['electricity'])

# Data selection
selected_date = '2023-01-01'

#Filter data for selected date
mask_load = load_data['time'].dt.strftime('%Y-%m-%d') == selected_date
selected_day_data = load_data.loc[mask_load]
mask_g1 = g1_max_data['time'].dt.strftime('%Y-%m-%d') == selected_date
selected_g1_data = g1_max_data.loc[mask_g1]

# Check if we have 24 hours
N=24
if len(selected_day_data) != N:
    raise ValueError(f"Selected date {selected_date} does not have {N} hours.")
if len(selected_g1_data) != N:
    raise ValueError(f"G1 max generation data for {selected_date} does not have {N} hours.")

# Extract Load and PV-Gen values
yearly_load_elec_housing = selected_day_data['load'].values * 100
G1_max = selected_g1_data['electricity'].values / 5

# %%
# 1. PARAMETERS

# Time horizon (hours)
N = len(yearly_load_elec_housing) # N = 24

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

# Gene, node 1 (PV)
f1 = 0  # Cost per unit of energy
# G1_max

# Gen, node 2 (Coal power plant)
f2 = 3  # Cost per unit of energy
G2_max = 100  # Capacity of G2 (per unit)

# %%
# 2. VARIABLES
dim_x = M * 2 * N  # Total number of variables

# %%
# 3. OBJECTIVE FUNCTION
f = np.concatenate([
    f1 * np.ones(N),    # Cost for P1
    f2 * np.ones(N),    # Cost for P2
    np.zeros(6 * N)     # No cost associated with P3, P4, and all Vs
])

# %%
# 4.1 CONSTRAINTS

def index_P(node, k):
    """Returns the index of P for a given node and time k."""
    return node * N + k

def index_V(node, k):
    """Returns the index of V for a given node and time k."""
    return M * N + node * N + k

# %%
# 4.2 EQUALITY CONSTRAINTS
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

# %%
# 4.3 INEQUALITY CONSTRAINTS

# Initialize A_ub and b_ub
num_lines = 5  # Number of lines
num_ineq = N * num_lines * 2  # Two constraints per line per time step
A_ub = lil_matrix((num_ineq, dim_x))
b_ub = np.zeros(num_ineq)

rate_limit = 100
line_limits = {
    (0, 1): {'B': Y12, 'P_max': rate_limit},  # Line 0-1
    (0, 2): {'B': Y13, 'P_max': rate_limit},  # Line 0-2
    (0, 3): {'B': Y14, 'P_max': rate_limit},  # Line 0-3
    (1, 3): {'B': Y24, 'P_max': rate_limit},  # Line 1-3
    (2, 3): {'B': Y34, 'P_max': rate_limit},  # Line 2-3
}

row = 0
for k in range(N):
    for (i, j), params in line_limits.items():
        B_ij = params['B']
        P_ij_max = params['P_max']
        
        # Upper limit: B_ij (V_i - V_j) <= P_ij_max
        A_ub[row, index_V(i, k)] = B_ij
        A_ub[row, index_V(j, k)] = -B_ij
        b_ub[row] = P_ij_max
        row += 1
        
        # Lower limit: -B_ij (V_i - V_j) <= P_ij_max
        A_ub[row, index_V(i, k)] = -B_ij
        A_ub[row, index_V(j, k)] = B_ij
        b_ub[row] = P_ij_max
        row += 1

# Convert A_ub to CSR format for efficiency
A_ub = A_ub.tocsr()

# %%
# 4.4 BOUNDS
# Define bounds for variables
bounds = []

# For P variables (Generators and Loads)
for node in range(M):
    if node == 0:  # P1 (PV Generator at Node 1)
        bounds.extend([(0, G1_max[k]) for k in range(N)])
    elif node == 1:  # P2 (Coal)
        bounds.extend([(0, G2_max) for k in range(N)])
    elif node == 2:  # P3 (Transit node)
        bounds.extend([(0, 0) for k in range(N)])
    elif node == 3:  # P4 (Load)
        bounds.extend([(-yearly_load_elec_housing[k], -yearly_load_elec_housing[k]) for k in range(N)])

# For V variables (Voltage Angles)
for node in range(M):
    if node == 0:  # V1
        bounds.extend([(0, 0) for k in range(N)])  # V1 fixed at 0
    else:  # V2, V3, V4
        bounds.extend([(None, None) for k in range(N)])

# Convert A_eq to CSR format for efficiency
A_eq = A_eq.tocsr()

# %%
# 5. SOLVE
result = linprog(
    c=f,
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
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

# Compute power flows
P12 = Y12 * (V1 - V2)
P13 = Y13 * (V1 - V3)
P14 = Y14 * (V1 - V4)
P24 = Y24 * (V2 - V4)
P34 = Y34 * (V3 - V4)

# Verify that power flows are within limits
line_flows = {
    'P12': P12,
    'P13': P13,
    'P14': P14,
    'P24': P24,
    'P34': P34
}

for line, flows in line_flows.items():
    max_flow = 100  # Adjust if different for each line
    if np.any(np.abs(flows) > max_flow):
        print(f"Warning: {line} exceeds the flow limit.")
    else:
        print(f"{line} is within limits.")

# %%
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
plt.plot(P1, 'b', label='G1 (PV Generation)')
plt.plot(G1_max, 'b--', label='G1 Max')
plt.plot(P2, 'r', label='G2 (Coal Power Plant)')
plt.plot([G2_max] * N, 'r--', label='G2 Max')
plt.grid(True)
# plt.plot(-P4, label='Load')
plt.xlabel('Hours')
plt.ylabel('Generation')
plt.legend()
plt.title('Generation Units Output')
plt.show()

# %%
# Demand and Generation Stack as Bars

# Extract the demand (negative P4) and generation (P1 and P2)
demand = -P4  # Since P4 is negative load
gen_PV = P1
gen_coal = P2

# Check if total generation matches the demand
total_generation = gen_PV + gen_coal
if not np.allclose(total_generation, demand, atol=1e-3):
    print("Warning: Total generation does not exactly match the demand at all time steps.")

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
hours = np.arange(N)

# Plot PV generation
plt.bar(hours, gen_PV, label='PV Generation', color='goldenrod')

# Plot Coal generation on top of PV generation
plt.bar(hours, gen_coal, bottom=gen_PV, label='Coal Generation', color='grey')

# Plot the demand as a black line
plt.plot(hours, demand, label='Demand', linewidth=2)

plt.xlabel('Hour of the Day')
plt.ylabel('Power [MW]')
plt.xticks(hours)
plt.legend(loc='upper left')
plt.title('Demand and Generation Mix per Hour')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# %%
