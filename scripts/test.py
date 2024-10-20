import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. System Set-up
# ===========================

# Reactances between buses (p.u.)
reactances = np.array([[0, 0.02, 0.05, 0.1],  # Line reactances between buses
                       [0.02, 0, 0.03, 0.04],
                       [0.05, 0.03, 0, 0.01],
                       [0.1, 0.04, 0.01, 0]]) 

# B' matrix (admittance matrix)
def create_admittance_matrix(reactances):
    num_buses = reactances.shape[0]
    B_prime = np.zeros((num_buses, num_buses))
    
    for i in range(num_buses):
        for j in range(num_buses):
            if i != j:
                B_prime[i, j] = -1 / reactances[i, j]
        B_prime[i, i] = -np.sum(B_prime[i, :])
    
    return B_prime

B_prime = create_admittance_matrix(reactances)

# Demand at each bus
bus_demand = np.array([2.0, 1.5, 1.0, 2.5])  # Active power demand in p.u.

# Reference bus (slack bus), assume bus 0
reference_bus = 0

# ================================
# 2. Solve DC Power Flow Equations
# ================================

def solve_dc_power_flow(B_prime, bus_demand, reference_bus):
    # Remove the reference bus row and column for solving
    B_prime_reduced = np.delete(np.delete(B_prime, reference_bus, axis=0), reference_bus, axis=1)
    demand_reduced = np.delete(bus_demand, reference_bus)
    
    # Solve for angles (θ) of non-reference buses
    angles = np.linalg.solve(B_prime_reduced, demand_reduced)
    
    # Insert reference bus angle (0)
    full_angles = np.insert(angles, reference_bus, 0)
    
    return full_angles

# Solve for bus angles
theta = solve_dc_power_flow(B_prime, bus_demand, reference_bus)

# ================================
# 3. Calculate Line Flows
# ================================

def calculate_line_flows(theta, reactances):
    num_buses = len(theta)
    line_flows = np.zeros((num_buses, num_buses))
    
    for i in range(num_buses):
        for j in range(num_buses):
            if i != j:
                line_flows[i, j] = (theta[i] - theta[j]) / reactances[i, j]
    
    return line_flows

line_flows = calculate_line_flows(theta, reactances)

# ================================
# 4. Plot Results
# ================================

print("Bus angles (radians):", theta)
print("Line flows (p.u.):")
print(line_flows)

# Plot bus angles
plt.figure(figsize=(8, 4))
plt.bar(range(len(theta)), theta, tick_label=[f"Bus {i}" for i in range(len(theta))])
plt.xlabel("Bus")
plt.ylabel("Angle (radians)")
plt.title("Bus Voltage Angles (DC Power Flow)")
plt.show()

# Plot line flows
plt.figure(figsize=(8, 4))
plt.imshow(line_flows, cmap='Blues', interpolation='nearest')
plt.colorbar(label="Power Flow (p.u.)")
plt.xlabel("Bus")
plt.ylabel("Bus")
plt.title("Power Flow Between Buses")
plt.show()
