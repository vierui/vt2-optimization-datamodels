import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pickle
import networkx as nx
import pandas as pd

# ===========================
# 1. Data Loading
# ===========================

def load_filtered_data(wind_file, solar_file, demand_file, selected_date):
    # Load the data
    wind_data = pd.read_csv('data/raw/wind-sion-2023.csv', skiprows=3, parse_dates=['time'])
    solar_data = pd.read_csv('data/raw/pv-sion-2023.csv', skiprows=3, parse_dates=['time'])
    demand_data = pd.read_csv('data/raw/demand-sion-2023.csv', skiprows=3, parse_dates=['time'])

    # Define the time range for the selected date
    start_date = selected_date + ' 00:00'
    end_date = selected_date + ' 23:59'
    
    # Filter data for the selected date
    wind_filtered = wind_data[(wind_data['time'] >= start_date) & (wind_data['time'] <= end_date)]
    solar_filtered = solar_data[(solar_data['time'] >= start_date) & (solar_data['time'] <= end_date)]
    demand_filtered = demand_data[(demand_data['time'] >= start_date) & (demand_data['time'] <= end_date)]
    
    # Extract electricity and demand columns
    wind_gen = wind_filtered['electricity'].values
    solar_gen = solar_filtered['electricity'].values
    total_demand = demand_filtered['total_demand'].values * 100
    
    return wind_gen, solar_gen, total_demand

# ===========================
# 2. Variables and Set-up
# ===========================

# Cost coefficients per p.u. for:
cost_coefficients = np.array([50,   # nuclear
                              0,    # wind
                              0,    # solar 
                              70])  # gas

# Hourly demand for 24 hours
# hourly_demand = np.array([3, 3, 3, 2.5, 2.5, 3, 4, 5, 5.5, 6, 6.5, 7, 7.5, 7, 6.5, 6, 6, 6.5, 6, 5, 4, 3.5, 3, 3])
# Split total demand between Load 1 and Load 2 (e.g., 60% to Load 1, 40% to Load 2)
# load1_demand = hourly_demand * 0.6  # 60% of total demand to Load 1
# load2_demand = hourly_demand * 0.4  # 40% of total demand to Load 2

# Estimation of Generation availability (in p.u.)
nuclear_availability = np.ones(24) * 200.0
gas_availability = np.ones(24) * 400.0

# Estimation of Reactances between buses (in p.u.)
x_nuclear_wind = 0.1
x_nuclear_solar = 0.15
x_wind_gas = 0.2
x_solar_gas = 0.1
x_gas_load = 0.2
x_nuclear_load = 0.2
x_solar_load = 0.25
x_wind_load = 0.3

# Admittance matrix B' (inverse of reactances)
B_prime = np.array([
    [1/x_nuclear_wind + 1/x_nuclear_solar + 1/x_nuclear_load, -1/x_nuclear_wind, -1/x_nuclear_solar, 0, -1/x_nuclear_load],
    [-1/x_nuclear_wind, 1/x_nuclear_wind + 1/x_wind_gas + 1/x_wind_load, 0, -1/x_wind_gas, -1/x_wind_load],
    [-1/x_nuclear_solar, 0, 1/x_nuclear_solar + 1/x_solar_gas + 1/x_solar_load, -1/x_solar_gas, -1/x_solar_load],
    [0, -1/x_wind_gas, -1/x_solar_gas, 1/x_wind_gas + 1/x_solar_gas, 0],
    [-1/x_nuclear_load, -1/x_wind_load, -1/x_solar_load, 0, 1/x_nuclear_load + 1/x_wind_load + 1/x_solar_load + 1/x_gas_load]
])

# ========================
# 3. Functions
# ========================

def objective(P):
    """Objective function: total cost of generation"""
    return np.dot(cost_coefficients, P)

def dc_power_flow(P_injections):
    # Remove the last row/column for the reference bus (Gas bus here)
    B_prime_reduced = B_prime[:-1, :-1]
    P_injections_reduced = P_injections[:-1]
    
    # Solve for bus angles (excluding reference bus)
    theta = np.linalg.solve(B_prime_reduced, P_injections_reduced)
    theta = np.append(theta, 0)  # Reference bus angle set to 0 (Gas bus)

    # Calculate power flows between buses
    P_nuclear_wind = (theta[0] - theta[1]) / x_nuclear_wind
    P_nuclear_solar = (theta[0] - theta[2]) / x_nuclear_solar
    P_wind_gas = (theta[1] - theta[3]) / x_wind_gas
    P_solar_gas = (theta[2] - theta[3]) / x_solar_gas
    P_nuclear_load = (theta[0] - theta[4]) / x_nuclear_load
    P_solar_load = (theta[2] - theta[4]) / x_solar_load
    P_wind_load = (theta[1] - theta[4]) / x_wind_load

    return theta, P_nuclear_wind, P_nuclear_solar, P_wind_gas, P_solar_gas, P_nuclear_load, P_solar_load, P_wind_load

# ================================
# 4. Main
# ================================

# Optimized results storage
nuclear_gen = []
wind_gen = []
solar_gen = []
gas_gen = []
total_costs = []

# DATE SELECTION
selected_date = '2023-02-01'
wind_gen_data, solar_gen_data, total_demand = load_filtered_data('data/raw/wind.csv', 'data/raw/pv.csv', 'data/raw/demand.csv', selected_date)

# Loop through the 24 hours
for hour in range(24):
    demand = total_demand[hour]
    
    # Use wind and solar first, capped by their availability
    wind_gen_hour = min(wind_gen_data[hour], demand)
    remaining_demand = demand - wind_gen_hour
    
    solar_gen_hour = min(solar_gen_data[hour], remaining_demand)
    remaining_demand -= solar_gen_hour
    
    # Use nuclear generation next
    nuclear_gen_hour = min(nuclear_availability[hour], remaining_demand)
    remaining_demand -= nuclear_gen_hour
    
    # Gas fills any remaining demand
    gas_gen_hour = max(0, remaining_demand)
    
    # Power injections (no load splitting)
    P_injections = [nuclear_gen_hour, wind_gen_hour, solar_gen_hour, gas_gen_hour, -demand]
    
    # Run power flow with the updated injections
    theta, P_nuclear_wind, P_nuclear_solar, P_wind_gas, P_solar_gas, P_nuclear_load, P_solar_load, P_wind_load = dc_power_flow(P_injections)
    
    # Store generation outputs
    nuclear_gen.append(nuclear_gen_hour)
    wind_gen.append(wind_gen_hour)
    solar_gen.append(solar_gen_hour)
    gas_gen.append(gas_gen_hour)
    
    # Calculate and store total cost (only for generation)
    total_cost = objective([nuclear_gen_hour, wind_gen_hour, solar_gen_hour, gas_gen_hour])
    total_costs.append(total_cost)
    
    print(f"Hour {hour}: Nuclear: {nuclear_gen_hour:.2f} p.u., Wind: {wind_gen_hour:.2f} p.u., Solar: {solar_gen_hour:.2f} p.u., Gas: {gas_gen_hour:.2f} p.u.")
    print(f"Total Generation Cost: ${total_cost:.2f}")

# ================================
# 5. Save Data 
# ================================

# Save the variables into a pickle file after optimization completes
data_to_save = {
    "total_demand": total_demand,                   # Total demand without splitting
    "nuclear_availability": nuclear_availability,   # Nuclear availability (estimated)
    "wind_availability": wind_gen_data,             # Actual wind data (loaded from file)
    "solar_availability": solar_gen_data,           # Actual solar data (loaded from file)
    "gas_availability": gas_availability,           # Gas availability (estimated)
    "nuclear_gen": nuclear_gen,                     # Nuclear generation results
    "wind_gen": wind_gen,                           # Wind generation results
    "solar_gen": solar_gen,                         # Solar generation results
    "gas_gen": gas_gen,                             # Gas generation results
    "total_costs": total_costs                      # Total cost per hour
}

# Save the data in a pickle file for future use
with open('data/results/optimization_results.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
print("Optimization results saved to 'data/results/optimization_results.pkl'")

# ================================
# 6. Results Plotting 
# ================================

# Ensure the logs/figures directory exists
os.makedirs('logs/figures', exist_ok=True)

# Plot the results
hours = np.arange(24)

# Plot results for the selected date
plt.figure(figsize=(10, 6))
plt.plot(hours, nuclear_gen, label="Nuclear Generation", marker='o')
plt.plot(hours, wind_gen, label="Wind Generation", marker='o')
plt.plot(hours, solar_gen, label="Solar Generation", marker='o')
plt.plot(hours, gas_gen, label="Gas Generation", marker='o')
plt.xlabel("Hour of the Day")
plt.ylabel("Generation (p.u.)")
plt.title(f"Generation per Source on {selected_date}")
plt.legend()
plt.grid(True)
plt.show()

# Plot total generation costs over time
plt.figure(figsize=(10, 6))
plt.plot(hours, total_costs, label="Total Generation Cost", marker='o', color='r')
plt.xlabel("Hour of the Day")
plt.ylabel("Total Generation Cost ($)")
plt.title("Total Generation Costs Over 24 Hours")
plt.grid(True)
plt.show()

# ================================

# Grid overview
G = nx.DiGraph()  # Create a directed graph
G.add_node("Nuclear", pos=(0, 2))
G.add_node("Wind", pos=(2, 3))
G.add_node("Solar", pos=(2, 1))
G.add_node("Gas", pos=(4, 2))
G.add_node("Load", pos=(5, 2))  # Single load bus

# Add edges representing transmission lines (reactances)
G.add_edge("Nuclear", "Wind", weight=x_nuclear_wind)
G.add_edge("Nuclear", "Solar", weight=x_nuclear_solar)
G.add_edge("Wind", "Gas", weight=x_wind_gas)
G.add_edge("Solar", "Gas", weight=x_solar_gas)
G.add_edge("Nuclear", "Load", weight=x_nuclear_load)  # Single load
G.add_edge("Solar", "Load", weight=x_solar_load)
G.add_edge("Gas", "Load", weight=x_gas_load)
G.add_edge("Wind", "Load", weight=x_wind_load)

# Get positions for all nodes
pos = nx.get_node_attributes(G, 'pos')

# Draw the network
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', edge_color='gray', width=2)

# Draw edge labels (reactances)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]} p.u.' for u, v, d in G.edges(data=True)}, font_size=10)

plt.title("Grid Overview with Reactances (p.u.)")
plt.grid(False)

# Save the grid plot
plt.savefig('logs/figures/grid_overview.png')

# Show the plot
plt.show()
