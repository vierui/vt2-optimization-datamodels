# %%
from scipy.optimize import linprog

# Cost vector
c_test = [5, 9, 0, 0]  # Cost for [P_wind, P_solar, θ2, θ3]

# Reactances (with adjusted x_13 for flexibility)
x_12 = 0.01
x_23 = 0.01
x_13 = 0.005  # Keep x_13 smaller to ease flow between Bus 1 and Bus 3

demand_value = 98.4  # Full demand at Bus 3

# Adjusted power balance equations, omitting Bus 3 temporarily
A_eq_test = [
    [1, 0, -1 / x_12, -1 / x_13],  # Bus 1: P_wind - (1/x_12) * θ2 - (1/x_13) * θ3 = 0
    [0, 1, 1 / x_12, -1 / x_23]    # Bus 2: P_solar + (1/x_12) * θ2 - (1/x_23) * θ3 = 0
]
b_eq_test = [0, 0]

# Inequality constraints for generation limits and non-negativity
A_ineq_test = [
    [1, 0, 0, 0],    # Upper bound for P_wind
    [0, 1, 0, 0],    # Upper bound for P_solar
    [-1, 0, 0, 0],   # Non-negativity for P_wind
    [0, -1, 0, 0]    # Non-negativity for P_solar
]
b_ineq_test = [45.058, 80, 0, 0]  # Generation limits

# Solve the updated model
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
    print("Feasibility test without Bus 3 balance constraint successful!")
    print("Solution:", result_test.x)
else:
    print("Feasibility test failed:", result_test.message)

# %%
