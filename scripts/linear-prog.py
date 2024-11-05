# %%
from scipy.optimize import linprog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# ===========================
# 1. Data Loading
# ===========================

# Load data files (assuming CSV format similar to initial code)
wind_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/raw/wind-sion-2023.csv', skiprows=3, parse_dates=['time'], delimiter=',')
solar_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/raw/pv-sion-2023.csv', skiprows=3, parse_dates=['time'], delimiter=',')
demand_data = pd.read_csv('/Users/ruivieira/Documents/Ecole/6_ZHAW/VT1/data/raw/data-load-becc.csv', header=None, names=['time', 'load'], parse_dates=['time'], delimiter=';',
                          date_format="%d.%m.%y %H:%M")

# Select a specific day
selected_date = '2023-08-01'
start_date = f"{selected_date} 00:00"
end_date = f"{selected_date} 23:59"

# Filter data for the selected day
wind_gen_data = wind_data[(wind_data['time'] >= start_date) & (wind_data['time'] <= end_date)]['electricity'].values
# solar_gen_data = solar_data[(solar_data['time'] >= start_date) & (solar_data['time'] <= end_date)]['electricity'].values
solar_gen_data = [250] * 24
demand_filtered = demand_data[(demand_data['time'] >= start_date) & (demand_data['time'] <= end_date)]['load'].values * 600

# %%
# ===========================
# 2. Equality and Inequality Constraint Matrices
# ===========================

# Define cost vector (cost coefficients for wind and solar for all hours)
c = []
for t in range(24):
    # P_wind (5), P_solar (9), Theta_1 (0), Theta_2 (0), Theta_3 (0), Theta_4 (0)
    c += [5, 9, 0, 0, 0, 0]

# Define reactances
x_12 = 0.1
x_13 = 0.15
x_14 = 0.2
x_23 = 0.2
x_34 = 0.1 
    
# Initialize empty lists for equality constraint matrix and vector
A_eq = []
b_eq = []

total_variables = 24 * 6  # 144 variables

for t in range(24):
    start_idx = t * 6

    # Power balance at Bus 1 (Wind Bus)
    row_eq_bus1 = [0] * total_variables
    row_eq_bus1[start_idx + 0] = 1  # P_wind(t)
    row_eq_bus1[start_idx + 2] = (1 / x_12) + (1 / x_13) + (1 / x_14)
    row_eq_bus1[start_idx + 3] = -1 / x_12
    row_eq_bus1[start_idx + 4] = -1 / x_13
    row_eq_bus1[start_idx + 5] = -1 / x_14
    A_eq.append(row_eq_bus1)
    b_eq.append(0)

    # Power balance at Bus 2 (Solar Bus)
    row_eq_bus2 = [0] * total_variables
    row_eq_bus2[start_idx + 1] = 1  # P_solar(t)
    row_eq_bus2[start_idx + 2] = -1 / x_12
    row_eq_bus2[start_idx + 3] = (1 / x_12) + (1 / x_23)
    row_eq_bus2[start_idx + 4] = -1 / x_23
    A_eq.append(row_eq_bus2)
    b_eq.append(0)

    # Power balance at Bus 3 (Load Bus)
    row_eq_bus3 = [0] * total_variables
    row_eq_bus3[start_idx + 2] = -1 / x_13
    row_eq_bus3[start_idx + 3] = -1 / x_23
    row_eq_bus3[start_idx + 4] = (1 / x_13) + (1 / x_23) + (1 / x_34)
    row_eq_bus3[start_idx + 5] = -1 / x_34
    A_eq.append(row_eq_bus3)
    b_eq.append(-demand_filtered[t])

    # Power balance at Bus 4 (Transit Node)
    row_eq_bus4 = [0] * total_variables
    row_eq_bus4[start_idx + 2] = -1 / x_14
    row_eq_bus4[start_idx + 4] = -1 / x_34
    row_eq_bus4[start_idx + 5] = (1 / x_14) + (1 / x_34)
    A_eq.append(row_eq_bus4)
    b_eq.append(0)

    # Reference angle constraint for Theta_1
    row_eq_theta1 = [0] * total_variables
    row_eq_theta1[start_idx + 2] = 1
    A_eq.append(row_eq_theta1)
    b_eq.append(0)

# Convert A_eq and b_eq to numpy arrays
A_eq = np.array(A_eq)
b_eq = np.array(b_eq)

# %%
# ===========================

# Initialize inequality constraint matrix and vector
A_ineq = []
b_ineq = []

for t in range(24):
    start_idx = t * 6

    # Upper limit for wind generation
    row_ineq_wind_upper = [0] * total_variables
    row_ineq_wind_upper[start_idx] = 1
    A_ineq.append(row_ineq_wind_upper)
    b_ineq.append(wind_gen_data[t])

    # Lower limit for wind generation
    row_ineq_wind_lower = [0] * total_variables
    row_ineq_wind_lower[start_idx] = -1
    A_ineq.append(row_ineq_wind_lower)
    b_ineq.append(0)

    # Upper limit for solar generation
    row_ineq_solar_upper = [0] * total_variables
    row_ineq_solar_upper[start_idx + 1] = 1
    A_ineq.append(row_ineq_solar_upper)
    b_ineq.append(solar_gen_data[t])

    # Lower limit for solar generation
    row_ineq_solar_lower = [0] * total_variables
    row_ineq_solar_lower[start_idx + 1] = -1
    A_ineq.append(row_ineq_solar_lower)
    b_ineq.append(0)

# Convert A_ineq and b_ineq to numpy arrays
A_ineq = np.array(A_ineq)
b_ineq = np.array(b_ineq)

bounds = []
for t in range(24):
    bounds.append((0, wind_gen_data[t]))  # P_wind(t)
    bounds.append((0, solar_gen_data[t]))  # P_solar(t)
    bounds.extend([(None, None)] * 4)     # Theta_1 to Theta_4

# %%
# ===========================
# 3. Solve
# ===========================

# Solve the linear programming problem
result = linprog(
    c,
    A_eq=A_eq,
    b_eq=b_eq,
    A_ub=A_ineq,
    b_ub=b_ineq,
    bounds=bounds,
    method='highs',
    options={'disp': True}
)

# result = linprog(
#     c,
#     A_eq=A_eq,
#     b_eq=b_eq,
#     bounds=bounds,
#     method='highs',
#     options={'disp': True}
# )

# Check and extract the solution
if result.success:
    print("Optimization successful!")
    solution = result.x  # Optimal values of power injections and angles
else:
    print("Optimization failed:", result.message)

# print("Optimal solution (result.x):")
# print(result.x)

# %%
# ===========================
# 4. Visualization
# ===========================

# Reshape the solution to a 24x8 matrix (24 hours, 8 variables per hour)
solution_reshaped = solution.reshape(24, 8)

# Convert to a DataFrame for easier plotting and labeling
columns = ["P_wind", "P_solar", "P_load", "Free_Bus", "Theta_1", "Theta_2", "Theta_3", "Theta_4"]
solution_df = pd.DataFrame(solution_reshaped, columns=columns)

# Plot Power Generation and Load with Available Generation and Demand
plt.figure(figsize=(12, 6))
# plt.plot(solution_df["P_wind"], label="Optimized Wind Generation (P_wind)", marker='o')
# plt.plot(solution_df["P_solar"], label="Optimized Solar Generation (P_solar)", marker='o')
# plt.plot(solution_df["P_load"], label="Optimized Load (P_load)", marker='o', linestyle='--')
plt.plot(demand_filtered, label="Hourly Demand", color='black', linestyle='--', marker='x')
plt.plot(wind_gen_data, label="Wind Generation Availability", color='blue', linestyle=':', marker='s')
plt.plot(solar_gen_data, label="Solar Generation Availability", color='orange', linestyle=':', marker='s')
plt.xlabel("Hour of the Day")
plt.ylabel("Power (kW)")
plt.title("Hourly Generation Availability and Demand")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Plot Voltage Angles for Each Bus
plt.figure(figsize=(12, 6))
plt.plot(solution_df["Theta_1"], label="Theta_1 (Reference Bus)", marker='o')
plt.plot(solution_df["Theta_2"], label="Theta_2", marker='o')
plt.plot(solution_df["Theta_3"], label="Theta_3", marker='o')
plt.plot(solution_df["Theta_4"], label="Theta_4", marker='o')
plt.xlabel("Hour of the Day")
plt.ylabel("Voltage Angle (rad)")
plt.title("Voltage Angles Over 24 Hours")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Plot Power Generation and Load with Available Generation and Demand
plt.figure(figsize=(12, 6))
plt.plot(solution_df["P_wind"], label="Optimized Wind Generation (P_wind)", marker='o')
plt.plot(solution_df["P_solar"], label="Optimized Solar Generation (P_solar)", marker='o')
# plt.plot(solution_df["P_load"], label="Optimized Load (P_load)", marker='o', linestyle='--')
plt.plot(demand_filtered, label="Hourly Demand", color='black', linestyle='--', marker='x')

plt.xlabel("Hour of the Day")
plt.ylabel("Power (kW)")
plt.title("Optimized Generations vs Demand")
plt.legend()
plt.grid(True)
plt.show()
# %%
# Calculate total generation for each hour vs
total_generation = np.array(wind_gen_data) + np.array(solar_gen_data)
hours = range(24)
plt.figure(figsize=(10, 6))
plt.plot(hours, demand_filtered, label="Demand", color="red", marker="o")
plt.plot(hours, total_generation, label="Total Generation (Wind + Solar)", color="blue", marker="x")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.title("Hourly Demand vs. Generation")
plt.legend()
plt.grid(True)
plt.show()

# %%
