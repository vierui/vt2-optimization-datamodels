# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.dates import DateFormatter, DayLocator

# Directory where your .csv files are located
data_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/"

# Define file paths for each season
files = {
    "autumn_spring": [
        os.path.join(data_dir, "load-autumn_spring.csv"),
        os.path.join(data_dir, "solar-autumn_spring.csv"),
        os.path.join(data_dir, "wind-autumn_spring.csv"),
    ],
    "summer": [
        os.path.join(data_dir, "load-summer.csv"),
        os.path.join(data_dir, "solar-summer.csv"),
        os.path.join(data_dir, "wind-summer.csv"),
    ],
    "winter": [
        os.path.join(data_dir, "load-winter.csv"),
        os.path.join(data_dir, "solar-winter.csv"),
        os.path.join(data_dir, "wind-winter.csv"),
    ],
}

# Function to plot data for a season
def plot_season_data(season, file_list):
    # Read data from CSV files
    load_data = pd.read_csv(file_list[0], sep=";")
    solar_data = pd.read_csv(file_list[1], sep=";")
    wind_data = pd.read_csv(file_list[2], sep=";")
    
    # Extract time and values
    time = load_data['time']
    load = load_data['value']*100
    solar = solar_data['value'] /10
    wind = wind_data['value'] /10
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, load, label="Load", marker="o")
    plt.plot(time, solar, label="Solar", marker="^")
    plt.plot(time, wind, label="Wind", marker="s")
    
    # Remove x-axis legend and labels
    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_visible(False)  # Hide the entire x-axis
    
    # Add titles and labels
    plt.title(f"{season.capitalize()} Season - Solar, Wind, and Load")
    plt.xlabel("Time (Month-Day)")
    plt.ylabel("Energy Generation (MW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save and show the plot
    output_path = os.path.join(data_dir, f"{season}_season_plot.png")
    plt.savefig(output_path)
    plt.show()
    print(f"Saved plot for {season} season at: {output_path}")

# Loop through seasons and create plots
for season, file_list in files.items():
    plot_season_data(season, file_list)
# %%
