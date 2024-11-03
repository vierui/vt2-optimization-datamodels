
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

# %%
c_one_hour = [5, 9, 0, 0, 0]  # [P_wind, P_solar, θ2, θ3, θ4]

# Define reactances
x_12 = 0.005
x_13 = 0.0075
x_14 = 0.01
x_23 = 0.01
x_34 = 0.005

# Initialize A_eq and b_eq for one-hour problem
A_eq = []
b_eq = []

# Power balance at Bus 1 (Wind Generation Bus)
row_eq_bus1 = [0] * 5  # [P_wind, P_solar, θ2, θ3, θ4]
row_eq_bus1[0] = 1     # P_wind coefficient
row_eq_bus1[2] = -1 / x_12   # θ2 (Bus 2)
row_eq_bus1[3] = -1 / x_13   # θ3 (Bus 3)
row_eq_bus1[4] = -1 / x_14   # θ4 (Bus 4)
A_eq.append(row_eq_bus1)
b_eq.append(0)  # Power balance at Bus 1, θ1 contribution moved to b_eq as 0

# Power balance at Bus 2 (Solar Generation Bus)
row_eq_bus2 = [0] * 5
row_eq_bus2[1] = 1       # P_solar coefficient
row_eq_bus2[2] = (1 / x_12) + (1 / x_23)  # θ2 (Bus 2)
row_eq_bus2[3] = -1 / x_23   # θ3 (Bus 3)
A_eq.append(row_eq_bus2)
b_eq.append(0)  # Power balance at Bus 2, θ1 contribution moved to b_eq as 0

# Power balance at Bus 3 (Load Bus) with fixed demand
row_eq_bus3 = [0] * 5
row_eq_bus3[2] = -1 / x_23   # θ2 (Bus 2)
row_eq_bus3[3] = (1 / x_13) + (1 / x_23) + (1 / x_34)  # θ3 (Bus 3)
row_eq_bus3[4] = -1 / x_34   # θ4 (Bus 4)
A_eq.append(row_eq_bus3)
b_eq.append(-96)  # Demand at Bus 3

# Power balance at Bus 4 (Free Bus)
row_eq_bus4 = [0] * 5
row_eq_bus4[3] = -1 / x_34   # θ3 (Bus 3)
row_eq_bus4[4] = (1 / x_14) + (1 / x_34)  # θ4 (Bus 4)
A_eq.append(row_eq_bus4)
b_eq.append(0)  # Power balance at Bus 4

# Convert A_eq and b_eq to numpy arrays
A_eq_one_hour = np.array(A_eq)
b_eq_one_hour = np.array(b_eq)

# Initialize inequality constraint matrix and vector
A_ineq = []
b_ineq = []

# Upper limit for wind generation (P_wind <= wind_gen_data)
row_ineq_wind_upper = [0] * 5
row_ineq_wind_upper[0] = 1  # P_wind
A_ineq.append(row_ineq_wind_upper)
b_ineq.append(45.058)  # Maximum wind generation

# Upper limit for solar generation (P_solar <= solar_gen_data)
row_ineq_solar_upper = [0] * 5
row_ineq_solar_upper[1] = 1  # P_solar
A_ineq.append(row_ineq_solar_upper)
b_ineq.append(2500)  # Maximum solar generation

# Lower bound constraints (non-negativity for P_wind and P_solar)
row_ineq_wind_lower = [0] * 5
row_ineq_wind_lower[0] = -1  # P_wind >= 0
A_ineq.append(row_ineq_wind_lower)
b_ineq.append(0)

row_ineq_solar_lower = [0] * 5
row_ineq_solar_lower[1] = -1  # P_solar >= 0
A_ineq.append(row_ineq_solar_lower)
b_ineq.append(0)

# Convert A_ineq and b_ineq to numpy arrays
A_ineq_one_hour = np.array(A_ineq)
b_ineq_one_hour = np.array(b_ineq)

# %%
# Solve the one-hour problem with the reduced variables
result_one_hour = linprog(
    c_one_hour,
    A_eq=A_eq_one_hour,
    b_eq=b_eq_one_hour,
    A_ub=A_ineq_one_hour,
    b_ub=b_ineq_one_hour,
    options={'disp': True}
)

# Display the results
if result_one_hour.success:
    print("One-hour optimization successful!")
    solution_one_hour = result_one_hour.x
    print("Solution for one hour:", solution_one_hour)
else:
    print("One-hour optimization failed:", result_one_hour.message)

# %%
