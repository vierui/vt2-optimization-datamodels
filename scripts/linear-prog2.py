# %%
from scipy.optimize import linprog

# Final cost vector with θ3 included
c = [5, 9, 0, 0]  # Cost for [P_wind, P_solar, θ2, θ3]

# Reactances for all connections
x_12 = 0.01  # Between Bus 1 and Bus 2
x_23 = 0.01  # Between Bus 2 and Bus 3
x_13 = 0.005  # Between Bus 1 and Bus 3 (direct line)
demand_value = 98.4  # Full demand at Bus 3

# Power balance equations with θ2 and θ3, and direct line from Bus 1 to Bus 3
A_eq = [
    [1, 0, -1 / x_12, -1 / x_13],         # Bus 1: P_wind - (1/x_12) * θ2 - (1/x_13) * θ3 = 0
    [0, 1, 1 / x_12, -1 / x_23],          # Bus 2: P_solar + (1/x_12) * θ2 - (1/x_23) * θ3 = 0
    [0, 0, -1 / x_13, 1 / x_23]           # Bus 3: Flow from θ3 to meet demand
]
b_eq = [0, 0, demand_value]  # Set demand at Bus 3

# Inequality constraints for generation limits and non-negativity
A_ineq = [
    [1, 0, 0, 0],    # Upper bound for P_wind
    [0, 1, 0, 0],    # Upper bound for P_solar
    [-1, 0, 0, 0],   # Non-negativity for P_wind
    [0, -1, 0, 0]    # Non-negativity for P_solar
]
b_ineq = [45.058, 250, 0, 0]  # Generation limits

# Solve the final model
result = linprog(
    c,
    A_eq=A_eq,
    b_eq=b_eq,
    A_ub=A_ineq,
    b_ub=b_ineq,
    options={'disp': True}
)

# Output results
if result.success:
    print("Feasibility test with full model (Bus 1-2-3) successful!")
    print("Solution:", result.x)
else:
    print("Feasibility test failed:", result.message)

# %%
