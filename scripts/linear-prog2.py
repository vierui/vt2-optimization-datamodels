# %%
from scipy.optimize import linprog

# Cost vector with θ3 included
c_test = [5, 9, 0, 0]  # Cost for [P_wind, P_solar, θ2, θ3]

# Reactances
x_12 = 0.01  # Between Bus 1 and Bus 2
x_23 = 0.01  # Between Bus 2 and Bus 3
demand_value = 98.4  # Full demand at Bus 3

# Power balance equations with θ2 and θ3
A_eq_test = [
    [1, 0, -1 / x_12, 0],           # Bus 1: P_wind - (1/x_12) * θ2 = 0
    [0, 1, 1 / x_12, -1 / x_23],    # Bus 2: P_solar + (1/x_12) * θ2 - (1/x_23) * θ3 = 0
    [0, 0, 0, 1 / x_23]             # Bus 3: Flow from Bus 2 to Bus 3 meets Demand
]
b_eq_test = [0, 0, demand_value]

# Inequality constraints for generation limits and non-negativity
A_ineq_test = [
    [1, 0, 0, 0],    # Upper bound for P_wind
    [0, 1, 0, 0],    # Upper bound for P_solar
    [-1, 0, 0, 0],   # Non-negativity for P_wind
    [0, -1, 0, 0]    # Non-negativity for P_solar
]
b_ineq_test = [45.058, 250, 0, 0]

# %%
# Solve the model
result_test = linprog(
    c_test,
    A_eq=A_eq_test,
    b_eq=b_eq_test,
    A_ub=A_ineq_test,
    b_ub=b_ineq_test,
    options={'disp': True}
)

# Output results
if result_test.success:
    print("Feasibility test with θ2 and θ3 (full demand) successful!")
    print("Solution:", result_test.x)
else:
    print("Feasibility test failed:", result_test.message)

# %%
