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

# Calculate total generation
total_generation = np.array(nuclear_gen) + np.array(wind_gen) + np.array(solar_gen) + np.array(gas_gen)

# Create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

hours = np.arange(24)

# 1. Plot demand and generation availability
axs[0, 0].plot(hours, hourly_demand, label='Demand', color='black', linestyle='-', marker='o')
axs[0, 0].plot(hours, nuclear_availability, label='Nuclear Availability', color='blue', linestyle='--', marker='o')
axs[0, 0].plot(hours, wind_availability, label='Wind Availability', color='green', linestyle='--', marker='o')
axs[0, 0].plot(hours, solar_availability, label='Solar Availability', color='orange', linestyle='--', marker='o')
axs[0, 0].plot(hours, gas_availability, label='Gas Availability', color='red', linestyle='--', marker='o')
axs[0, 0].set_xlabel("Hour of the Day")
axs[0, 0].set_ylabel("Power (p.u.)")
axs[0, 0].set_title("Hourly Demand and Generation Availability")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Plot optimized generation vs demand
axs[0, 1].plot(hours, hourly_demand, label='Demand', color='black', linestyle='-', marker='o')
axs[0, 1].plot(hours, nuclear_gen, label='Nuclear Generation', color='blue', linestyle='-', marker='o')
axs[0, 1].plot(hours, wind_gen, label='Wind Generation', color='green', linestyle='-', marker='o')
axs[0, 1].plot(hours, solar_gen, label='Solar Generation', color='orange', linestyle='-', marker='o')
axs[0, 1].plot(hours, gas_gen, label='Gas Generation', color='red', linestyle='-', marker='o')
axs[0, 1].set_xlabel("Hour of the Day")
axs[0, 1].set_ylabel("Power (p.u.)")
axs[0, 1].set_title("Optimized Generation vs Demand")
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Plot total generation vs demand
axs[1, 0].plot(hours, hourly_demand, label='Demand', color='black', linestyle='-', marker='o')
axs[1, 0].plot(hours, total_generation, label='Total Generation', color='purple', linestyle='--', marker='x')
axs[1, 0].set_xlabel("Hour of the Day")
axs[1, 0].set_ylabel("Power (p.u.)")
axs[1, 0].set_title("Total Generation vs Demand")
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Stacked bar plot to show contributions of each energy source
axs[1, 1].bar(hours, nuclear_gen, label="Nuclear Generation", color="blue", edgecolor='black')
axs[1, 1].bar(hours, wind_gen, bottom=nuclear_gen, label="Wind Generation", color="green", edgecolor='black')
axs[1, 1].bar(hours, solar_gen, bottom=np.array(nuclear_gen) + np.array(wind_gen), label="Solar Generation", color="orange", edgecolor='black')
axs[1, 1].bar(hours, gas_gen, bottom=np.array(nuclear_gen) + np.array(wind_gen) + np.array(solar_gen), label="Gas Generation", color="red", edgecolor='black')
axs[1, 1].plot(hours, hourly_demand, label="Demand", color="black", linestyle='-', marker='o', linewidth=2)  # Plot demand on top
axs[1, 1].set_xlabel("Hour of the Day")
axs[1, 1].set_ylabel("Power (p.u.)")
axs[1, 1].set_title("Generation Contributions vs Demand (Stacked)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('logs/figures/four_plots_combined.png')
plt.show()
