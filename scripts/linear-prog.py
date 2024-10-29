# %%
from scipy.optimize import linprog
import pandas as pd
import numpy as np
# %%
# ===========================
# 1. Data Loading
# ===========================

# Load data files (assuming CSV format similar to initial code)
wind_data = pd.read_csv('data/raw/wind-sion-2023.csv', skiprows=3, parse_dates=['time'], delimiter=',')
solar_data = pd.read_csv('data/raw/pv-sion-2023.csv', skiprows=3, parse_dates=['time'], delimiter=',')
demand_data = pd.read_csv('data/raw/data-load-becc.csv', header=None, names=['time', 'load'], parse_dates=['time'], delimiter=';')

# Select a specific day
selected_date = '2023-08-01'
start_date = f"{selected_date} 00:00"
end_date = f"{selected_date} 23:59"
# selected_date = '2023-02-01'
# start_date = selected_date + ' 00:00'
# end_date = selected_date + ' 23:59'

# Filter data for the selected day
wind_gen_data = wind_data[(wind_data['time'] >= start_date) & (wind_data['time'] <= end_date)]['electricity'].values
solar_gen_data = solar_data[(solar_data['time'] >= start_date) & (solar_data['time'] <= end_date)]['electricity'].values
demand_filtered = demand_data[(demand_data['time'] >= start_date) & (demand_data['time'] <= end_date)]['load'].values

# %%
# ===========================
# 2. Equality and Inequality Constraint Matrices
# ===========================

# Define cost vector (cost coefficients for wind and solar for all hours)
c = []
for t in range(24):
    # Wind (15), Solar (5), Load (0), Free Bus (0), Angles (0, 0, 0, 0)
    c += [15, 5, 0, 0, 0, 0, 0, 0]

# Define reactances
x_12 = 0.1
x_13 = 0.15
x_14 = 0.2
x_23 = 0.2
x_34 = 0.1 
    
# Initialize empty lists for equality constraint matrix and vector
A_eq = []
b_eq = []

for t in range(24):
    # For each hour, calculate the start index for that hour's 8 variables in the decision vector
    start_idx = t * 8
    
    # Example constraint for power balance at Bus 1 (Wind Bus)
    row_eq_bus1 = [0] * 192
    row_eq_bus1[start_idx] = 1  # P_wind(t)
    row_eq_bus1[start_idx + 4] = -1 / x_12  # Theta_1 reference angle (if needed in the constraint)
    row_eq_bus1[start_idx + 5] = -1 / x_13  # Theta_2 angle component
    row_eq_bus1[start_idx + 6] = -1 / x_14  # Theta_3 angle component
    A_eq.append(row_eq_bus1)
    b_eq.append(0)  # The target power injection for the wind bus (if zero, adjust as needed)

    # Example constraint for power balance at Bus 3 (Load Bus)
    row_eq_bus3 = [0] * 192
    row_eq_bus3[start_idx + 2] = 1  # P_load(t)
    row_eq_bus3[start_idx + 5] = -1 / x_23  # Theta_2 angle component
    row_eq_bus3[start_idx + 6] = -1 / x_34  # Theta_3 angle component
    A_eq.append(row_eq_bus3)
    b_eq.append(-demand_filtered[t] * 1000)  # Scale the demand if needed

    # Reference angle constraint for Theta_1 (setting it to 0)
    row_eq_theta1 = [0] * 192
    row_eq_theta1[start_idx + 4] = 1  # Theta_1 position (reference angle set to 0)
    A_eq.append(row_eq_theta1)
    b_eq.append(0)
    
A_eq = np.array(A_eq)
b_eq = np.array(b_eq)

# ===========================

# Initialize inequality constraint matrix and vector
A_ineq = []
b_ineq = []

for t in range(24):
    # For each hour, position the 8-variable constraints in the right location
    # Start index for the current hour's variables in the full decision vector
    start_idx = t * 8

    # Upper limit for wind generation (places 1 at the wind position for hour t)
    row_ineq_wind_upper = [0] * 192
    row_ineq_wind_upper[start_idx] = 1  # Wind variable position for hour t
    A_ineq.append(row_ineq_wind_upper)
    b_ineq.append(wind_gen_data[t] *1000)  # Apply capacity in kW if needed

    # Upper limit for solar generation
    row_ineq_solar_upper = [0] * 192
    row_ineq_solar_upper[start_idx + 1] = 1  # Solar variable position for hour t
    A_ineq.append(row_ineq_solar_upper)
    b_ineq.append(solar_gen_data[t] *1000)

    # Lower limit for wind generation (non-negativity constraint)
    row_ineq_wind_lower = [0] * 192
    row_ineq_wind_lower[start_idx] = -1  # Wind variable position for hour t
    A_ineq.append(row_ineq_wind_lower)
    b_ineq.append(0)

    # Lower limit for solar generation (non-negativity constraint)
    row_ineq_solar_lower = [0] * 192
    row_ineq_solar_lower[start_idx + 1] = -1  # Solar variable position for hour t
    A_ineq.append(row_ineq_solar_lower)
    b_ineq.append(0)

# Convert A_ineq and b_ineq to numpy arrays
A_ineq = np.array(A_ineq)
b_ineq = np.array(b_ineq)

# ===========================
# 3. Solve
# ===========================

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq, options={'disp': True})

# Check and extract the solution
if result.success:
    print("Optimization successful!")
    solution = result.x  # Optimal values of power injections and angles
else:
    print("Optimization failed:", result.message)

# print("Optimal solution (result.x):")
# print(result.x)

# ===========================
# 5. Debug
# ===========================

# # Solve without equality constraints
# result_no_eq = linprog(c, A_ub=A_ineq, b_ub=b_ineq, options={'disp': True})

# # Check result
# if result_no_eq.success:
#     print("Optimization feasible without equality constraints.")
# else:
#     print("Still infeasible without equality constraints.")
#     print(result_no_eq.message)

# ===========================

# # Solve without inequality constraints
# result_no_ineq = linprog(c, A_eq=A_eq, b_eq=b_eq, options={'disp': True})
# # Solve without inequality constraints
# result_no_ineq = linprog(c, A_eq=A_eq, b_eq=b_eq, options={'disp': True})

# # Check result
# if result_no_ineq.success:
#     print("Optimization feasible without inequality constraints.")
# else:
#     print("Still infeasible without inequality constraints.")
#     print(result_no_ineq.message)

# ===========================

# b_eq_relaxed = b_eq * 0.95  # Decrease demand by 5%
# b_ineq_relaxed = b_ineq * 1.05  # Increase generation bounds by 5%

# result_relaxed = linprog(c, A_eq=A_eq, b_eq=b_eq_relaxed, A_ub=A_ineq, b_ub=b_ineq_relaxed, options={'disp': True})

# if result_relaxed.success:
#     print("Optimization feasible with relaxed constraints.")
# else:
#     print("Still infeasible with relaxed constraints.")
#     print(result_relaxed.message)

# ===========================

# total_generation_capacity = sum(wind_gen_data * 1000) + sum(solar_gen_data * 1000)
# total_demand = sum(demand_filtered * 1000)
# print("Total generation capacity:", total_generation_capacity)
# print("Total demand:", total_demand)
# %%