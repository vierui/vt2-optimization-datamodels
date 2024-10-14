import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the optimization results from the pickle file
with open('data/optimization_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract data
hourly_demand = data["hourly_demand"]
nuclear_availability = data["nuclear_availability"]
wind_availability = data["wind_availability"]
solar_availability = data["solar_availability"]
gas_availability = data["gas_availability"]

nuclear_gen = data["nuclear_gen"]
wind_gen = data["wind_gen"]
solar_gen = data["solar_gen"]
gas_gen = data["gas_gen"]
total_costs = data["total_costs"]

# Plot demand and generation availability
hours = np.arange(24)

plt.figure(figsize=(10, 6))

# Plot demand
plt.plot(hours, hourly_demand, label='Demand', color='black', linestyle='-', marker='o')

# Plot generation availability
plt.plot(hours, nuclear_availability, label='Nuclear Availability', color='blue', linestyle='--', marker='o')
plt.plot(hours, wind_availability, label='Wind Availability', color='green', linestyle='--', marker='o')
plt.plot(hours, solar_availability, label='Solar Availability', color='orange', linestyle='--', marker='o')
plt.plot(hours, gas_availability, label='Gas Availability', color='red', linestyle='--', marker='o')

plt.xlabel("Hour of the Day")
plt.ylabel("Power (p.u.)")
plt.title("Hourly Demand and Generation Availability")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('logs/figures/demand_vs_availability.png')

# Show the plot
plt.show()

# Plot optimized generation vs demand
plt.figure(figsize=(10, 6))

# Plot demand
plt.plot(hours, hourly_demand, label='Demand', color='black', linestyle='-', marker='o')

# Plot optimized generation
plt.plot(hours, nuclear_gen, label='Nuclear Generation', color='blue', linestyle='-', marker='o')
plt.plot(hours, wind_gen, label='Wind Generation', color='green', linestyle='-', marker='o')
plt.plot(hours, solar_gen, label='Solar Generation', color='orange', linestyle='-', marker='o')
plt.plot(hours, gas_gen, label='Gas Generation', color='red', linestyle='-', marker='o')

plt.xlabel("Hour of the Day")
plt.ylabel("Power (p.u.)")
plt.title("Optimized Generation vs Demand")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('logs/figures/optimized_generation_vs_demand.png')

# Show the plot
plt.show()
