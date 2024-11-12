# %%
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

# %%
# 1. PARAMETERS 

# Time horizon (hours)
N = 10  # This parameter can be chosen arbitrarily

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
G2_max = 0.7  # Capacity of G2 (per unit)

# Load curve (Assumed as an example)
yearly_load_elec_housing = 0.5 * np.ones(N)  # Example load curve

# %%
# 2. VARIABLES
dim_x = M * 2 * N  # Total number of variables

# %%
# 3. OBJECTIVE FUNCTION

# Cost vector
f = np.concatenate([
    f1 * np.ones(N),    # Cost for P1
    f2 * np.ones(N),    # Cost for P2
    np.zeros(6 * N)     # No cost associated with P3, P4, and all Vs
])

# %%
# 4. CONSTRAINTS 

def index_P(node, k):
    """Returns the index of P for a given node and time k."""
    return node * N + k

def index_V(node, k):
    """Returns the index of V for a given node and time k."""
    return (M + node) * N + k

# %%
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

# %%
# 4. INEQUALITY CONSTRAINTS 
A_ub = lil_matrix((4 * N, dim_x))
b_ub = np.zeros(4 * N)

row = 0
for k in range(N):
    # P1(k) ≥ 0  -->  -P1(k) ≤ 0
    A_ub[row, index_P(0, k)] = -1
    b_ub[row] = 0
    row += 1

    # P1(k) ≤ G1_max
    A_ub[row, index_P(0, k)] = 1
    b_ub[row] = G1_max
    row += 1

    # P2(k) ≥ 0  -->  -P2(k) ≤ 0
    A_ub[row, index_P(1, k)] = -1
    b_ub[row] = 0
    row += 1

    # P2(k) ≤ G2_max
    A_ub[row, index_P(1, k)] = 1
    b_ub[row] = G2_max
    row += 1

# %%
# 4. BOUNDS

# Define bounds for variables
bounds = []

for k in range(N):
    # For P variables
    for node in range(M):
        if node == 0:  # P1
            bounds.append((0, G1_max))
        elif node == 1:  # P2
            bounds.append((0, G2_max))
        elif node == 2:  # P3
            bounds.append((0, 0))  # P3 fixed at 0
        elif node == 3:  # P4
            bounds.append((-yearly_load_elec_housing[k], -yearly_load_elec_housing[k]))  # P4 fixed

    # For V variables
    for node in range(M):
        if node == 0:  # V1
            bounds.append((0, 0))  # V1 fixed at 0
        else:  # V2, V3, V4
            bounds.append((None, None))

# %%
# 5. SOLVE

# Convert A_eq and A_ub to CSR format for efficiency
A_eq = A_eq.tocsr()
A_ub = A_ub.tocsr()

# Solve the linear program
result = linprog(
    c=f,
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

# %%
# 7. PLOT

#Generation and load
plt.figure()
plt.plot(P1 + P2, label='Total Generation')
plt.plot(-P4, label='Load')  # P4 is negative load
plt.grid(True)
plt.xlabel('Hours')
plt.ylabel('Power')
plt.legend()
plt.title('Total Generation vs. Load')
plt.show()

#Individual generation and their limits
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
