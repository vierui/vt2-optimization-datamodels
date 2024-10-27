from scipy.optimize import linprog
import pandas as pd
import numpy as np

# ===========================
# 1. Data Loading
# ===========================

# Load data files (assuming CSV format similar to initial code)
wind_data = pd.read_csv('data/raw/wind-sion-2023.csv', skiprows=3, parse_dates=['time'])
solar_data = pd.read_csv('data/raw/pv-sion-2023.csv', skiprows=3, parse_dates=['time'])
demand_data = pd.read_csv('data/raw/demand-sion-2023.csv', skiprows=3, parse_dates=['time'])

# Select a specific day
selected_date = '2023-02-01'
start_date = selected_date + ' 00:00'
end_date = selected_date + ' 23:59'

# Filter data for the selected day
wind_gen_data = wind_data[(wind_data['time'] >= start_date) & (wind_data['time'] <= end_date)]['electricity'].values
solar_gen_data = solar_data[(solar_data['time'] >= start_date) & (solar_data['time'] <= end_date)]['electricity'].values
total_demand_data = demand_data[(demand_data['time'] >= start_date) & (demand_data['time'] <= end_date)]['total_demand'].values * 100  # Scaling demand

# ===========================
# 2. Equality and Inequality Constraint Matrices
# ===========================

# Define cost vector (cost coefficients for wind and solar for all hours)
c = []
for t in range(24):
    # Wind (15), Solar (5), Load (0), Free Bus (0), Angles (0, 0, 0, 0)
    c += [15, 5, 0, 0, 0, 0, 0, 0]
    
# Initialize empty lists for equality constraint matrix and vector
A_eq = []
b_eq = []

for t in range(24):
    # Power balance constraint at each bus
    # Wind, Solar, Load, Free, Reference angle, Angle2, Angle3, Angle4
    row_eq_wind = [1, 0, -1, 0, 0, 0, 0, 0]   # Power balance for Wind
    row_eq_solar = [0, 1, -1, 0, 0, 0, 0, 0]  # Power balance for Solar
    row_eq_load = [0, 0, 1, 0, 0, 0, 0, 0]    # Load power injection equal to demand
    row_eq_free = [0, 0, 0, 1, 0, 0, 0, 0]    # Free bus power injection is zero
    row_eq_theta1 = [0, 0, 0, 0, 1, 0, 0, 0]  # Reference angle constraint (theta_1 = 0)

    # Add these rows for the current hour t to A_eq
    A_eq.extend([row_eq_wind, row_eq_solar, row_eq_load, row_eq_free, row_eq_theta1])

    # Corresponding values for b_eq
    b_eq.extend([0, 0, -total_demand_data[t], 0, 0])  # Demand as negative load injection

# Initialize inequality constraint matrix and vector
A_ineq = []
b_ineq = []

for t in range(24):
    # Upper limit for wind generation
    row_ineq_wind_upper = [1, 0, 0, 0, 0, 0, 0, 0]
    A_ineq.append(row_ineq_wind_upper)
    b_ineq.append(wind_gen_data[t])

    # Upper limit for solar generation
    row_ineq_solar_upper = [0, 1, 0, 0, 0, 0, 0, 0]
    A_ineq.append(row_ineq_solar_upper)
    b_ineq.append(solar_gen_data[t])

    # Lower limit for wind generation (non-negativity)
    row_ineq_wind_lower = [-1, 0, 0, 0, 0, 0, 0, 0]
    A_ineq.append(row_ineq_wind_lower)
    b_ineq.append(0)

    # Lower limit for solar generation (non-negativity)
    row_ineq_solar_lower = [0, -1, 0, 0, 0, 0, 0, 0]
    A_ineq.append(row_ineq_solar_lower)
    b_ineq.append(0)

# ===========================
# 4. Solve
# ===========================

# Convert lists to matrices
A_eq = np.array(A_eq)
b_eq = np.array(b_eq)
A_ineq = np.array(A_ineq)
b_ineq = np.array(b_ineq)

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq)

# Check and extract the solution
if result.success:
    print("Optimization successful!")
    solution = result.x  # Optimal values of power injections and angles
else:
    print("Optimization failed:", result.message)
