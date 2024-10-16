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

def objective(P):
    """Objective function: total cost of generation"""
    return np.dot(cost_coefficients, P)

def power_balance_constraint(P, demand):
    """Power balance constraint: ensures total generation meets demand"""
    return np.sum(P) - demand

def jacobian():
    """Jacobian matrix for power balance"""
    # Since power balance constraint is linear in P, the Jacobian is just ones.
    return np.array([1, 1, 1, 1])

def newton_raphson(demand, initial_guess, tol=1e-6, max_iter=100):
    """Newton-Raphson method for solving the power generation optimization problem"""
    P = initial_guess
    for i in range(max_iter):
        # Evaluate the power balance constraint (F)
        F = power_balance_constraint(P, demand)
        
        # Check if solution is within tolerance
        if abs(F) < tol:
            print(f"Converged after {i+1} iterations")
            return P
        
        # Compute the Jacobian (J)
        J = jacobian()
        
        # Update generation values
        delta_P = -F / np.sum(J)
        P = P + delta_P
        
        # Ensure generation is within limits (generation limits can be set here)
        P = np.clip(P, [0, 0, 0, 0], [nuclear_availability[0], wind_availability[0], solar_availability[0], gas_availability[0]])
    
    print("Did not converge within the maximum number of iterations")
    return P

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

for hour in range(24):
    demand = hourly_demand[hour]
    # Run Newton-Raphson to solve for generation values
    P_opt = newton_raphson(demand, initial_generation)
    
    # Append the optimized generation values
    nuclear_gen.append(P_opt[0])
    wind_gen.append(P_opt[1])
    solar_gen.append(P_opt[2])
    gas_gen.append(P_opt[3])
    
    # Calculate and store total cost
    total_cost = objective(P_opt)
    total_costs.append(total_cost)

    print(f"Hour {hour}: Nuclear: {P_opt[0]:.2f} p.u., Wind: {P_opt[1]:.2f} p.u., Solar: {P_opt[2]:.2f} p.u., Gas: {P_opt[3]:.2f} p.u.")
    print(f"Total Generation Cost: ${total_cost:.2f}")

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
