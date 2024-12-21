# %%
import os
import pandas as pd
from datetime import datetime

# Define your input directory (raw data) and output directory (processed data)
raw_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/raw"
processed_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed"

# Filenames
load_raw_file = os.path.join(raw_dir, "data-load-becc.csv")
pv_raw_file   = os.path.join(raw_dir, "pv-sion-2023.csv")
wind_raw_file = os.path.join(raw_dir, "wind-sion-2023.csv")

# Output processed filenames
load_out_file = os.path.join(processed_dir, "load-2023.csv")
pv_out_file   = os.path.join(processed_dir, "solar-2023.csv")
wind_out_file = os.path.join(processed_dir, "wind-2023.csv")

def process_load_data(load_csv, output_csv):
    """
    Process the load raw data from data-load-becc.csv, which has format:
      time;load
      01.01.23 00:00;3.68E-01
      ...
    and produce a CSV with columns: time,value
    """

    # Read using ';' as delimiter
    df = pd.read_csv(load_csv, delimiter=';', names=['raw_time', 'raw_load'], header=0)
    # Example of raw_time = "01.01.23 00:00"
    # Convert raw_time to a proper datetime; note the format might be dd.mm.yy HH:MM
    # Make sure year is 2023 or deduce from the file
    def parse_time(x):
        # Some data might include 2-digit year. We'll assume it is 20YY
        # e.g. 01.01.23 => 2023
        # Format is: dd.mm.yy HH:MM
        return datetime.strptime(x, "%d.%m.%y %H:%M")

    df['time'] = df['raw_time'].apply(parse_time)

    # Convert raw_load to numeric
    df['value'] = pd.to_numeric(df['raw_load'], errors='coerce')
    
    # Keep only needed columns
    df = df[['time','value']]

    # Sort by time, just in case
    df = df.sort_values('time').reset_index(drop=True)

    # Write processed file
    df.to_csv(output_csv, index=False)
    print(f"Processed load data -> {output_csv}")

def process_pv_data(pv_csv, output_csv):
    """
    Process the PV data from Renewables.ninja CSV, which has format:
      time,local_time,electricity
      2023-01-01 00:00,2023-01-01 01:00,0
      ...
    Output: time,value (in UTC), sorted.
    """
    df = pd.read_csv(pv_csv, comment='#')  # skip lines starting with '#'
    # By default columns are: time, local_time, electricity
    # We keep 'time' as UTC, rename 'electricity' -> 'value'
    df.rename(columns={'electricity': 'value'}, inplace=True)

    # Convert 'time' column to datetime
    # The format from Renewables.ninja is typically ISO8601, e.g. "2023-01-01 00:00"
    df['time'] = pd.to_datetime(df['time'])

    # Keep only time, value
    df = df[['time','value']]

    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"Processed PV data -> {output_csv}")

def process_wind_data(wind_csv, output_csv):
    """
    Similar to PV, Renewables.ninja wind CSV:
      time,local_time,electricity
      2023-01-01 00:00,2023-01-01 01:00,110.064
      ...
    Output: time,value, sorted.
    """
    df = pd.read_csv(wind_csv, comment='#')
    df.rename(columns={'electricity': 'value'}, inplace=True)
    
    df['time'] = pd.to_datetime(df['time'])
    
    df = df[['time','value']]

    df = df.sort_values('time').reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed wind data -> {output_csv}")

def main():
    # Ensure the processed_dir exists
    os.makedirs(processed_dir, exist_ok=True)

    # Process each dataset
    process_load_data(load_raw_file, load_out_file)
    process_pv_data(pv_raw_file, pv_out_file)
    process_wind_data(wind_raw_file, wind_out_file)

if __name__ == "__main__":
    main()
# %%
