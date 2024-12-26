#!/usr/bin/env python3

"""
create_master_gen.py

Reads wind-2023.csv and solar-2023.csv (each containing [time, value] for the full year),
filters three defined seasonal windows, and concatenates them into a single CSV:
master_gen.csv with columns:
    time, id, type, pmax, pmin, gencost, emax, einitial, eta, season
"""
# %%
import os
import pandas as pd

# Adjust paths as needed
processed_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data"
wind_file     = os.path.join(processed_dir, "processed/wind-2023.csv")
solar_file    = os.path.join(processed_dir, "processed/solar-2023.csv")
output_file   = os.path.join(processed_dir, "working/master_gen.csv")

# Define seasonal date ranges
season_info = {
    "winter": {
        "start": "2023-01-08 00:00:00",
        "end":   "2023-01-14 23:00:00"
    },
    "summer": {
        "start": "2023-08-06 00:00:00",
        "end":   "2023-08-12 23:00:00"
    },
    "autumn_spring": {
        "start": "2023-10-22 00:00:00",
        "end":   "2023-10-28 23:00:00"
    },
}

# Map each gen/storage type to a numeric ID:
type_to_id = {
    "nuclear": 1,
    "gas": 2,
    "wind": 3,
    "solar": 4,
    "battery1": 101,
    "battery2": 102
}

def load_and_filter(file_path, gen_type, season_name, start_str, end_str):
    """
    Reads the entire year from file_path ([time,value]),
    filters by [start_str, end_str],
    assigns columns including numeric 'id' from type_to_id,
    and returns a DataFrame with the standard columns.
    """
    df = pd.read_csv(file_path, parse_dates=["time"])

    # Filter by the specified date range
    mask = (
        (df["time"] >= pd.to_datetime(start_str)) &
        (df["time"] <= pd.to_datetime(end_str))
    )
    df_season = df.loc[mask].copy()

    # Rename 'value' -> 'pmax'
    df_season.rename(columns={"value": "pmax"}, inplace=True)

    # Add columns
    df_season["type"] = gen_type
    df_season["id"] = type_to_id[gen_type]  # numeric ID
    df_season["pmin"] = 0.0
    df_season["gencost"] = 0.0
    df_season["emax"] = 0.0
    df_season["einitial"] = 0.0
    df_season["eta"] = 1.0
    df_season["season"] = season_name

    # Reorder columns for clarity
    df_season = df_season[
        ["time", "id", "type", "pmax", "pmin", "gencost", "emax", "einitial", "eta", "season"]
    ]
    df_season.sort_values("time", inplace=True)

    return df_season

def create_constant_gen(gen_type, season_name, start_str, end_str,
                        pmax, pmin, gencost, emax, einitial, eta):
    """
    Creates a DataFrame for a time-invariant generator (e.g. nuclear, gas, battery)
    for each hour in [start_str, end_str].
    Now also assigns 'id' from type_to_id dict.
    Columns: time, id, type, pmax, pmin, gencost, emax, einitial, eta, season.
    """
    date_range = pd.date_range(start=start_str, end=end_str, freq='H')
    df = pd.DataFrame({"time": date_range})
    
    df["type"]     = gen_type
    df["id"]       = type_to_id[gen_type]  # numeric ID
    df["pmax"]     = pmax
    df["pmin"]     = pmin
    df["gencost"]  = gencost
    df["emax"]     = emax
    df["einitial"] = einitial
    df["eta"]      = eta
    df["season"]   = season_name

    df = df[["time","id","type","pmax","pmin","gencost","emax","einitial","eta","season"]]
    return df

def main():
    # Initialize an empty list to store partial DataFrames
    master_list = []

    for season_name, rng in season_info.items():
        start_time = rng["start"]
        end_time   = rng["end"]

        # 1) Load wind
        wind_df = load_and_filter(
            file_path=wind_file,
            gen_type="wind",
            season_name=season_name,
            start_str=start_time,
            end_str=end_time
        )
        master_list.append(wind_df)

        # 2) Load solar
        solar_df = load_and_filter(
            file_path=solar_file,
            gen_type="solar",
            season_name=season_name,
            start_str=start_time,
            end_str=end_time
        )
        master_list.append(solar_df)
        
        # 3) Add nuclear
        nuclear_df = create_constant_gen(
            gen_type="nuclear",
            season_name=season_name,
            start_str=start_time,
            end_str=end_time,
            pmax=800.0,   # e.g. 800 MW
            pmin=0.0,
            gencost=3.0,
            emax=0.0,
            einitial=0.0,
            eta=1.0
        )
        master_list.append(nuclear_df)

        # 4) Add gas
        gas_df = create_constant_gen(
            gen_type="gas",
            season_name=season_name,
            start_str=start_time,
            end_str=end_time,
            pmax=250.0,
            pmin=0.0,
            gencost=8.0,
            emax=0.0,
            einitial=0.0,
            eta=1.0
        )
        master_list.append(gas_df)
        
        # 5) Add battery1
        battery1_df = create_constant_gen(
            gen_type="battery1",
            season_name=season_name,
            start_str=start_time,
            end_str=end_time,
            pmax=40.0,
            pmin=-40.0,
            gencost=0.0,
            emax=100.0,       # MWh capacity
            einitial=0.0,     # start empty
            eta=0.99
        )
        master_list.append(battery1_df)

        # 6) Add battery2
        battery2_df = create_constant_gen(
            gen_type="battery2",
            season_name=season_name,
            start_str=start_time,
            end_str=end_time,
            pmax=50.0,
            pmin=-50.0,
            gencost=0.0,
            emax=200.0,
            einitial=0.0,
            eta=0.95
        )
        master_list.append(battery2_df)
        
    # Concatenate everything
    master_gen = pd.concat(master_list, ignore_index=True)
    # Sort for consistency
    master_gen.sort_values(["season","time","id"], inplace=True)
    master_gen.reset_index(drop=True, inplace=True)

    # Write final output
    master_gen.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(master_gen)} rows.")

if __name__ == "__main__":
    main()
# %%
