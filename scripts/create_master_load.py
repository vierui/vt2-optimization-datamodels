# %%
#!/usr/bin/env python3

"""
create_master_load.py

Creates master_load.csv with columns:
    time, bus, pd, season

For each season (winter, summer, autumn_spring):
  - Bus 5 uses load data from the same week as the season range.
  - Bus 6 uses load data from the *following* week, then time-shifts it 
    so it aligns with the season’s date range.

Example:
  If 'winter' is 2023-01-08 -> 2023-01-14,
    - bus 5 loads = 2023-01-08 -> 2023-01-14 (from load-2023.csv)
    - bus 6 loads = 2023-01-15 -> 2023-01-21 (from load-2023.csv), 
      but time-shifted 7 days backward so final times still show 2023-01-08 -> 2023-01-14
"""

import os
import pandas as pd

# Path setup
processed_dir = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed"
load_file     = os.path.join(processed_dir, "load-2023.csv")
output_file   = os.path.join(processed_dir, "master_load.csv")

# Define seasonal date ranges (same as in master_gen)
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

def load_and_filter(load_csv, start_str, end_str):
    """
    Loads the full-year load data [time, value] from load_csv,
    filters it to [start_str, end_str].
    Returns a DataFrame with columns [time, pd], sorted by time.
    """
    df = pd.read_csv(load_csv, parse_dates=["time"])
    
    mask = (
        (df["time"] >= pd.to_datetime(start_str)) &
        (df["time"] <= pd.to_datetime(end_str))
    )
    df_week = df.loc[mask].copy()
    
    # Rename 'value' -> 'pd' for clarity
    df_week.rename(columns={"value": "pd"}, inplace=True)

    # Sort by time
    df_week.sort_values("time", inplace=True)
    df_week.reset_index(drop=True, inplace=True)
    
    return df_week

def shift_load_data(df, shift_days):
    """
    Shifts the 'time' column in df by 'shift_days' (can be positive or negative).
    E.g., if shift_days = -7, every row's time moves 7 days earlier.
    """
    if shift_days != 0:
        df["time"] = df["time"] + pd.Timedelta(days=shift_days)
    return df

def main():
    master_list = []

    for season_name, rng in season_info.items():
        start_time = pd.to_datetime(rng["start"])
        end_time   = pd.to_datetime(rng["end"])

        # 1) Bus 5 load => same time range as the season
        bus5_df = load_and_filter(load_file, start_time, end_time)
        # Add columns bus, season
        bus5_df["bus"]    = 5
        bus5_df["season"] = season_name

        # 2) Bus 6 load => next week's data, then shift -7 days
        # Next week means [end_time+1 hour, end_time+7 days], or simply
        # [start_time+7 days, end_time+7 days], to keep it aligned hour-by-hour.
        # Let's do [start_time + 7days, end_time + 7days].
        next_week_start = start_time + pd.Timedelta(days=7)
        next_week_end   = end_time   + pd.Timedelta(days=7)
        bus6_df = load_and_filter(load_file, next_week_start, next_week_end)

        # Shift back by 7 days, so its final time range matches [start_time, end_time].
        bus6_df = shift_load_data(bus6_df, shift_days=-7)
        bus6_df["bus"]    = 6
        bus6_df["season"] = season_name

        # Concatenate bus5, bus6 for this season
        season_load_df = pd.concat([bus5_df, bus6_df], ignore_index=True)
        # Reorder columns for clarity
        season_load_df = season_load_df[["time", "bus", "pd", "season"]]

        master_list.append(season_load_df)
    
    # Combine all seasons
    master_load = pd.concat(master_list, ignore_index=True)
    # Sort for consistency
    master_load.sort_values(["season", "time", "bus"], inplace=True)
    master_load.reset_index(drop=True, inplace=True)

    # Write final output
    master_load.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(master_load)} rows.")

if __name__ == "__main__":
    main()
# %%
