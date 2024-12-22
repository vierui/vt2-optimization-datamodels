# %%
# 2. Data
# ===========================

import pandas as pd
import numpy as np
datadir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/"

# %%
# Load demand data
demand_data = pd.read_csv(datadir + 'processed/load-autumn_spring.csv', delimiter=';', parse_dates=['time'], header=0)
demand_data['value'] = pd.to_numeric(demand_data['value']) * 100
demand_data2 = pd.read_csv(datadir + 'processed/load2-autumn_spring.csv',delimiter=';',parse_dates=['time'],header=0)
demand_data2['value'] = pd.to_numeric(demand_data2['value']) * 100

# Create a DataFrame for time-varying demand at bus 5
load_1 = pd.DataFrame({
    'time': demand_data['time'],
    'bus': 5,
    'pd': demand_data['value'],
})
load_1['time'] = pd.to_datetime(load_1['time'])

load_2 = pd.DataFrame({
    'time': demand_data['time'],
    'bus': 6,
    'pd': demand_data2['value'],
})
load_2['time'] = pd.to_datetime(load_2['time'])

demand_time_series = pd.concat([load_1,load_2 ], ignore_index=True)

# Renewable generation data
#solar_data = pd.read_csv(datadir + 'processed/solar-summer.csv', parse_dates=['time'], delimiter=';',header=0)
#wind_data = pd.read_csv(datadir + 'processed/wind-summer.csv', parse_dates=['time'], delimiter=';',header=0)

#solar_data = pd.read_csv(datadir + 'processed/solar-autumn_spring.csv', parse_dates=['time'], delimiter=';',header=0)
#wind_data = pd.read_csv(datadir + 'processed/wind-autumn_spring.csv', parse_dates=['time'], delimiter=';',header=0)

solar_data = pd.read_csv(datadir + 'processed/solar-winter.csv', parse_dates=['time'], delimiter=';',header=0)
wind_data = pd.read_csv(datadir + 'processed/wind-winter.csv', parse_dates=['time'], delimiter=';',header=0)

# Create generator time series DataFrames
# Assign IDs and buses as per the new configuration:
# Nuclear: id=1, bus=1
# Gas: id=2, bus=2
# Wind: id=3, bus=3 (using wind_data)
# Solar: id=4, bus=4 (using solar_data)
# Wind Battery: id=5, bus=6
# Solar Battery: id=6, bus=7


# %%
# Nuclear
nuclear_gen = pd.DataFrame({
    'time': solar_data['time'],
    'id': 1,
    'bus': 1,
    'pmax': 300,
    'pmin': 0,
    'gencost': 3
})

# Gas
gas_gen = pd.DataFrame({
    'time': solar_data['time'],
    'id': 2,
    'bus': 2,
    'pmax': 1000,
    'pmin': 0,
    'gencost': 7
})

# Wind
wind_gen = pd.DataFrame({
    'time': wind_data['time'],
    'id': 3,
    'bus': 3,
    'pmax': wind_data['value']/10,  
    'pmin': 0,
    'gencost': 0
})

# Solar
solar_gen = pd.DataFrame({
    'time': wind_data['time'],
    'id': 4,
    'bus': 4,
    'pmax': solar_data['value']/10,
    'pmin': 0,
    'gencost': 0
})

# Wind Battery at bus 6
wind_storage = pd.DataFrame({
    'time': wind_data['time'],
    'id': 7,
    'bus': 7,
    'pmax': 40,
    'pmin': -40,
    'gencost': 0,
    
    'emax':100,
    'einitial':0,
    'eta':0.99
})

solar_storage = pd.DataFrame({
    'time': solar_data['time'],
    'id': 8,
    'bus': 8,
    'pmax': 40,
    'pmin': -40,
    'gencost': 0,
    
    'emax':100,
    'einitial':0,
    'eta':0.99
})

# Combine all generators and storage, fill NaNs 
gen_time_series = pd.concat([nuclear_gen, gas_gen, wind_gen, solar_gen, wind_storage, solar_storage], ignore_index=True)
gen_time_series = gen_time_series.fillna(0)

# Load branch and bus data (already processed and updated as requested)
branch = pd.read_csv(datadir + "processed/branch2.csv")
bus = pd.read_csv(datadir + "processed/bus2.csv")

# Rename columns to lowercase
for df in [branch, bus]:
    df.columns = df.columns.str.lower()

# Create IDs for branches
branch['id'] = np.arange(1, len(branch) + 1)

# Calculate susceptance
branch['sus'] = 1 / branch['x']

# Print to verify
# print(bus)
# print(branch)
#print(gen_time_series)
print(demand_time_series)
# print(gas_gen)
# print(nuclear_gen)
# print(gas_gen)
# print(wind_storage)

# print(gen_time_series.head())
# print(gen_time_series['id'].unique())  # Check generator IDs
# print(gen_time_series['time'].min(), gen_time_series['time'].max())  # Check time range

# print(demand_time_series['time'].min(), demand_time_series['time'].max())  # Check time range
# print(demand_time_series['time'].unique())  # Check unique time steps

# print(branch[['fbus', 'tbus']])

# print(bus['bus_i'].dtype)
# print(branch['fbus'].dtype, branch['tbus'].dtype)
# print(demand_time_series['bus'].dtype)
# %%
# Dataframes to .csv files
# demand_time_series.to_csv(datadir + 'scenarios/autumn_spring/demand.csv', index=False)
gen_time_series.to_csv(datadir + 'scenarios/winter/gen.csv', index=False)
# %%
import pandas as pd

# Path to the input file
input_file = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/wind-2023.csv"
# Path to the output file
output_file = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/wind-2023.csv"

# Read the CSV file
data = pd.read_csv(input_file)

# Scale the 'value' column by 100
data['value'] = data['value'] * 10

# Save the updated data to a new CSV file
data.to_csv(output_file, index=False)

print(f"Scaled data has been saved to {output_file}")

# %%
