# %%
from scipy.optimize import linprog

# Cost vector with penalty for unmet demand
penalty = 1000  # Large penalty to enforce demand fulfillment
c_test = [5, 9, 0, 0, penalty]  # Cost for [P_wind, P_solar, θ2, θ3, demand_shortfall]

# Reactances
x_12 = 0.01
x_23 = 0.01
x_13 = 0.005  # Adjusted for flexibility

demand_value = 98.4  # Full demand at Bus 3

# Power balance equations with θ2 and θ3, and demand shortfall
A_eq_test = [
    [1, 0, -1 / x_12, -1 / x_13, 0],           # Bus 1: P_wind - (1/x_12) * θ2 - (1/x_13) * θ3 = 0
    [0, 1, 1 / x_12, -1 / x_23, 0],            # Bus 2: P_solar + (1/x_12) * θ2 - (1/x_23) * θ3 = 0
    [0, 0, 0, 1 / x_23, -1]                    # Bus 3: Flow from θ3 meets demand with shortfall option
]
b_eq_test = [0, 0, demand_value]  # Full demand at Bus 3

# Inequality constraints for generation limits, non-negativity, and demand shortfall
A_ineq_test = [
    [1, 0, 0, 0, 0],    # Upper bound for P_wind
    [0, 1, 0, 0, 0],    # Upper bound for P_solar
    [-1, 0, 0, 0, 0],   # Non-negativity for P_wind
    [0, -1, 0, 0, 0],   # Non-negativity for P_solar
    [0, 0, 0, 0, -1]    # Non-negativity for demand_shortfall
]
b_ineq_test = [45.058, 250, 0, 0, 0]

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
    print("Feasibility test with soft demand constraint successful!")
    print("Solution:", result_test.x)
else:
    print("Feasibility test failed:", result_test.message)


# %%
