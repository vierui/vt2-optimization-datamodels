import numpy as np
from scipy.optimize import minimize
import csv
import matplotlib.pyplot as plt
import os
import pickle

# ===========================
# 1. Variables and Set-up
# ===========================

# Cost coefficients per p.u. for:
cost_coefficients = np.array([50,   # nuclear
                              0,    # wind
                              0,    # solar 
                              70])  # gas

# Hourly demand for 24 hours
hourly_demand = np.array([3, 3, 3, 2.5, 2.5, 3, 4, 5, 5.5, 6, 6.5, 7, 7.5, 7, 6.5, 6, 6, 6.5, 6, 5, 4, 3.5, 3, 3])

# Generation availability for each source (in p.u.)
nuclear_availability = np.ones(24) * 2.0
wind_availability = np.array([0.5, 0.4, 0.3, 0.4, 0.5, 1.0, 1.2, 1.5, 1.3, 1.0, 0.8, 0.6, 0.7, 1.0, 1.2, 1.5, 1.3, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])
solar_availability = np.array([0, 0, 0, 0, 0, 0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7, 1.6, 1.5, 1.0, 0.8, 0.5, 0.2, 0, 0, 0, 0, 0])
gas_availability = np.ones(24) * 4.0

# ========================
# 2. Functions
# ========================

# Objective function: Minimize total cost of generation
def objective(generation):
    P1, P2, P3, P4 = generation
    return cost_coefficients[0] * P1 + cost_coefficients[1] * P2 + cost_coefficients[2] * P3 + cost_coefficients[3] * P4

# Power balance constraint: Ensure generation meets demand
def power_balance_constraint(generation, demand):
    P1, P2, P3, P4 = generation
    total_generation = P1 + P2 + P3 + P4
    return total_generation - demand

# ================================
# 3. Main
# ================================

# Optimized results storage
nuclear_gen = []
wind_gen = []
solar_gen = []
gas_gen = []
total_costs = []

# Initial guesses for power generation at buses 1-4
initial_generation = [1.0,  # nuclear
                      0.5,  # wind
                      0.5,  # solar
                      2.0]  # gas

# Opti-Loop to optimize generation over each hour 
for hour in range(24):
    # Available generation limits for this hour
    generation_limits = [(0, nuclear_availability[hour]), # nuclear
                         (0, wind_availability[hour]),    # wind
                         (0, solar_availability[hour]),   # solar
                         (0, gas_availability[hour])]     # gas

    # Define demand for this hour
    demand = hourly_demand[hour]

    # Define constraints for the current hour
    constraints = [{'type': 'eq', 'fun': lambda gen: power_balance_constraint(gen, demand)}]

    # Perform optimization
    result = minimize(objective, initial_generation, bounds=generation_limits, constraints=constraints)

    # Optimized generation values
    P1_opt, P2_opt, P3_opt, P4_opt = result.x

    # Append results to arrays (for plot)
    nuclear_gen.append(P1_opt)
    wind_gen.append(P2_opt)
    solar_gen.append(P3_opt)
    gas_gen.append(P4_opt)
    total_costs.append(objective(result.x))

    # Print the results for this hour
    print(f"Hour {hour}:")
    print(f"  Nuclear: {P1_opt:.2f} p.u., Wind: {P2_opt:.2f} p.u., Solar: {P3_opt:.2f} p.u., Gas: {P4_opt:.2f} p.u.")
    print(f"  Total generation cost: ${objective(result.x):.2f}")
    print("-" * 40)

# ================================
# 4. Save Data 
# ================================

# Save the variables into a pickle file after optimization completes
data_to_save = {
    "hourly_demand": hourly_demand,
    "nuclear_availability": nuclear_availability,
    "wind_availability": wind_availability,
    "solar_availability": solar_availability,
    "gas_availability": gas_availability,
    "nuclear_gen": nuclear_gen,
    "wind_gen": wind_gen,
    "solar_gen": solar_gen,
    "gas_gen": gas_gen,
    "total_costs": total_costs
}

with open('data/optimization_results.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
print("Optimization results saved to 'data/optimization_results.pkl'")

# Save the results in a CSV file
with open("generation_results.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header with units
    writer.writerow(["Hour", "Nuclear (p.u.)", "Wind (p.u.)", "Solar (p.u.)", "Gas (p.u.)", "Total Cost ($)"])
    
    # Write the data for each hour with formatted numbers
    for hour in range(24):
        writer.writerow([
            f"{hour:02}",  # Ensure hour is always 2 digits
            f"{nuclear_gen[hour]:.2f}", 
            f"{wind_gen[hour]:.2f}", 
            f"{solar_gen[hour]:.2f}", 
            f"{gas_gen[hour]:.2f}", 
            f"{total_costs[hour]:.2f}"
        ])

# ================================
# 5. Results Plotting 
# ================================

# Ensure the logs/figures directory exists
os.makedirs('logs/figures', exist_ok=True)

# Plot the results
hours = np.arange(24)

# Plot the generation per source
plt.figure(figsize=(10, 6))
plt.plot(hours, nuclear_gen, label="Nuclear Generation", marker='o')
plt.plot(hours, wind_gen, label="Wind Generation", marker='o')
plt.plot(hours, solar_gen, label="Solar Generation", marker='o')
plt.plot(hours, gas_gen, label="Gas Generation", marker='o')
plt.xlabel("Hour of the Day")
plt.ylabel("Generation (p.u.)")
plt.title("Generation per Source Over 24 Hours")
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
