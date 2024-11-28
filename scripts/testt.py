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

# Storage Energy
E_max = 50  # Define maximum storage capacity
E_initial = 0.5  # Define initial storage energy
eta = 0.99  # Storage efficiency factor

# Line flow limits (Define maximum power flow for each line)
# Assuming you have these values; replace with actual data if different
P_max_12 = 100
P_max_13 = 100
P_max_14 = 100
P_max_24 = 100
P_max_34 = 100

# 2. VARIABLES
dim_x = M * 2 * N + (N + 1)  # 4 * 2 * 24 + 25 = 217

def index_P(node, k):
    """Returns the index of P for a given node and time k."""
    return node * N + k

def index_V(node, k):
    """Returns the index of V for a given node and time k."""
    return M * N + node * N + k

def index_E(k):
    """Returns the index of E at time k."""
    return M * 2 * N + k  # E variables start after P and V variables

# 3. OBJECTIVE FUNCTION
# Cost for P1 and P2; assume no cost for P3 and P4
f = np.concatenate([
    f1 * np.ones(N),    # Cost for P1
    f2 * np.ones(N),    # Cost for P2
    np.zeros(N),        # Cost for P3 (Storage)
    np.zeros(N),        # Cost for P4 (Load)
    np.zeros(4 * N),    # No cost for V1, V2, V3, V4
    np.zeros(N + 1)     # No cost for E(k)
])

# 4. EQUALITY CONSTRAINTS
# 7 constraints per hour + 2 storage constraints = 7*24 +2 = 170
num_constraints = 7 * N + 2  # Increased by 2 to accommodate E(0) = E_initial and E(N) = E(0)
A_eq = lil_matrix((num_constraints, dim_x))
b_eq = np.zeros(num_constraints)

row = 0
for k in range(N):
    for i in range(M):
        # Power Flow Constraints: Y[i,i]*V_i - sum(Y[i,j]*V_j) - P_i = 0
        A_eq[row, index_V(i, k)] += Y[i, i]  # Self-admittance term
        for j in range(M):
            if j != i:
                A_eq[row, index_V(j, k)] -= Y[i, j]  # Susceptance to other nodes
        A_eq[row, index_P(i, k)] -= 1  # Negative sign for P_i
        b_eq[row] = 0
        row += 1

    # Load at Node 4: P4(k) = -Load(k)
    A_eq[row, index_P(3, k)] = 1  # Node 4 is index 3 (0-based)
    b_eq[row] = -yearly_load_elec_housing[k]
    row += 1

    # Voltage Angle at Node 1: V1(k) = 0
    A_eq[row, index_V(0, k)] = 1  # Node 1 is index 0 (0-based)
    b_eq[row] = 0
    row += 1

    # Storage Dynamics: E(k+1) = eta * E(k) + P3(k)
    A_eq[row, index_E(k + 1)] = 1
    A_eq[row, index_E(k)] = -eta
    A_eq[row, index_P(2, k)] = -1  # Negative sign for P3(k)
    b_eq[row] = 0
    row += 1

# Set E(0) = E_initial
A_eq[row, index_E(0)] = 1
b_eq[row] = E_initial
row += 1

# Set E(N) = E(0)
A_eq[row, index_E(N)] = 1
A_eq[row, index_E(0)] = -1
b_eq[row] = 0
row += 1

# Verify that all constraints have been added correctly
assert row == num_constraints, f"Expected {num_constraints} constraints, but got {row}."

# 5. INEQUALITY CONSTRAINTS (A_ub and b_ub)

# Define line flow constraints and generator bounds
num_lines = 5  # Number of lines
num_ineq_constraints = 2 * num_lines * N + 2 * N  # Line flows and generator bounds

A_ub = lil_matrix((num_ineq_constraints, dim_x))
b_ub = np.zeros(num_ineq_constraints)

row_ub = 0

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
        A_ub[row_ub, index_V(i, k)] = Y_ij
        A_ub[row_ub, index_V(j, k)] = -Y_ij
        b_ub[row_ub] = P_max_ij
        row_ub += 1

        # Line flow: -Y_ij * (V_i - V_j) <= P_max
        A_ub[row_ub, index_V(i, k)] = -Y_ij
        A_ub[row_ub, index_V(j, k)] = Y_ij
        b_ub[row_ub] = P_max_ij
        row_ub += 1

# Generator bounds constraints: P1 <= G1_max and P2 <= G2_max
for k in range(N):
    # P1 (Generator 1)
    A_ub[row_ub, index_P(0, k)] = 1  # P1 <= G1_max
    b_ub[row_ub] = G1_max
    row_ub += 1

    # P2 (Generator 2)
    A_ub[row_ub, index_P(1, k)] = 1  # P2 <= G2_max
    b_ub[row_ub] = G2_max
    row_ub += 1

# Verify that all inequality constraints have been added correctly
assert row_ub == num_ineq_constraints, f"Expected {num_ineq_constraints} inequality constraints, but got {row_ub}."

# Convert A_ub to CSR format for efficiency
A_ub = A_ub.tocsr()

# 6. BOUNDS

# Reset bounds to (-inf, inf) since bounds are now handled via A_ub
bounds = [(None, None) for _ in range(dim_x)]

# Alternatively, you can set specific bounds where appropriate
# For example, P1 and P2 should be non-negative
for node in range(M):
    if node == 0:  # P1 (Generator 1)
        for k in range(N):
            bounds[index_P(node, k)] = (0, None)  # P1 >= 0
    elif node == 1:  # P2 (Generator 2)
        for k in range(N):
            bounds[index_P(node, k)] = (0, None)  # P2 >= 0
    elif node == 2:  # P3 (Storage)
        for k in range(N):
            bounds[index_P(node, k)] = (-np.inf, np.inf)  # P3 can charge or discharge
    elif node == 3:  # P4 (Load)
        for k in range(N):
            bounds[index_P(node, k)] = (-np.inf, np.inf)  # P4 is fixed via equality constraints

# Voltage angles: V1 is fixed at 0, others are free
for node in range(M):
    if node == 0:  # V1 (Reference node)
        for k in range(N):
            bounds[index_V(node, k)] = (0, 0)  # V1 fixed at 0
    else:
        for k in range(N):
            bounds[index_V(node, k)] = (None, None)  # V2, V3, V4 are free

# Storage Energy: 0 <= E(k) <= E_max
for k in range(N + 1):
    bounds[index_E(k)] = (0, E_max)

# 7. SOLVE THE OPTIMIZATION

# Combine all constraints and bounds in linprog
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

# 8. EXTRACT RESULTS

# Initialize arrays
P1 = np.zeros(N)
P2 = np.zeros(N)
P3 = np.zeros(N)
P4 = np.zeros(N)
V1 = np.zeros(N)
V2 = np.zeros(N)
V3 = np.zeros(N)
V4 = np.zeros(N)
E = np.zeros(N + 1)  # Storage energy from E(0) to E(N)

for k in range(N):
    P1[k] = result.x[index_P(0, k)]
    P2[k] = result.x[index_P(1, k)]
    P3[k] = result.x[index_P(2, k)]
    P4[k] = result.x[index_P(3, k)]
    
    V1[k] = result.x[index_V(0, k)]
    V2[k] = result.x[index_V(1, k)]
    V3[k] = result.x[index_V(2, k)]
    V4[k] = result.x[index_V(3, k)]
    
    E[k] = result.x[index_E(k)]
E[N] = result.x[index_E(N)]

# 9. VERIFY LOAD IS MET

# Calculate total generation for each hour
total_generation = P1 + P2 + P3  # Sum of Generators and Storage

# Calculate residuals (generation - load)
residual = total_generation - yearly_load_elec_housing

# Print residuals for each hour
print("\nHour | P1       | P2       | P3        | P4      | Total Generation | Load   | Residual")
print("-------------------------------------------------------------------------------------------")
for k in range(N):
    print(f"{k:>4} | {P1[k]:>8.4f} | {P2[k]:>8.4f} | {P3[k]:>9.4f} | {P4[k]:>7.3f} | {total_generation[k]:>17.4f} | {yearly_load_elec_housing[k]:>6.2f} | {residual[k]:>8.4f}")

# Calculate Maximum and Average Residuals
max_residual = np.max(np.abs(residual))
avg_residual = np.mean(np.abs(residual))

print(f"\nMaximum Residual: {max_residual:.6f}")
print(f"Average Residual: {avg_residual:.6f}")

# Plot Total Generation and Load Over Time
plt.figure(figsize=(12, 6))
hours = range(N)

plt.plot(hours, total_generation, label='Total Generation (P1 + P2 + P3)', marker='o')
plt.plot(hours, yearly_load_elec_housing, label='Load', marker='x')

plt.xlabel('Hour')
plt.ylabel('Power (Units)')
plt.title('Total Generation vs. Load Over 24 Hours')
plt.legend()
plt.grid(True)
plt.xticks(hours)  # Show all hours on the x-axis for clarity
plt.tight_layout()
plt.show()

# %%
