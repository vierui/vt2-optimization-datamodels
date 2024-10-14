import numpy as np
from scipy.optimize import minimize

# Define network reactances (in p.u.) between buses
# Bus indices are 0-based, so x[i][j] is reactance between bus i and bus j
x = np.array([[0, 1/3, 1/2, 0],  # Reactances between buses 1, 2, 3, 4
              [1/3, 0, 1/2, 1/4],
              [1/2, 1/2, 0, 1/3],
              [0, 1/4, 1/3, 0]])

# Define power injections (positive values are generation, negative values are load)
P_injections = np.array([1.5, -0.5, -1.0, 0.0])  # P1 = 1.5, P2 = -0.5, P3 = -1.0, P4 = 0

# Define initial guesses for voltage angles (theta) at buses 2, 3, and 4 (since theta_1 is reference)
initial_theta_guess = [0.0, 0.0, 0.0]  # Initial guesses for theta_2, theta_3, theta_4

# Function to calculate total transmission line losses (objective function to minimize)
def objective(theta):
    theta_full = np.concatenate([[0], theta])  # Full set of angles (theta_1 = 0 as slack)
    
    # Calculate power flows on each line using DC power flow equations
    total_losses = 0
    for i in range(len(P_injections)):
        for j in range(i + 1, len(P_injections)):
            if x[i][j] != 0:
                P_ij = (theta_full[i] - theta_full[j]) / x[i][j]  # Power flow on line i-j
                total_losses += P_ij ** 2  # Squared power flow to simulate losses
    return total_losses

# Power balance constraints (sum of flows at each bus = power injection at that bus)
def power_balance_constraint(theta):
    theta_full = np.concatenate([[0], theta])  # Full set of angles (theta_1 = 0 as slack)
    power_flows = np.zeros(len(P_injections))
    
    for i in range(len(P_injections)):
        for j in range(len(P_injections)):
            if i != j and x[i][j] != 0:
                power_flows[i] += (theta_full[i] - theta_full[j]) / x[i][j]
    
    return power_flows - P_injections  # Power flows should match injections

# Define constraints for optimization
constraints = {'type': 'eq', 'fun': power_balance_constraint}

# Set bounds on voltage angles to ensure stability (angles between -30 and 30 degrees in radians)
bounds = [(-np.radians(30), np.radians(30)) for _ in range(len(P_injections) - 1)]

# Perform optimization using minimize function from scipy
result = minimize(objective, initial_theta_guess, bounds=bounds, constraints=constraints)

# Print the optimized voltage angles
optimized_theta = np.concatenate([[0], result.x])  # Include theta_1 = 0
print(f"Optimized voltage angles (in radians): {optimized_theta}")

# Calculate and print the optimized power flows
for i in range(len(P_injections)):
    for j in range(i + 1, len(P_injections)):
        if x[i][j] != 0:
            P_ij = (optimized_theta[i] - optimized_theta[j]) / x[i][j]
            print(f"Power flow on line {i+1}-{j+1}: {P_ij:.3f} p.u.")