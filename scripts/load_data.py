#!/usr/bin/env python3

"""
load_data.py

Module for loading real data from CSV files and extracting representative weeks.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

def load_csv_data():
    """
    Load the CSV data files for load, solar, and wind.
    Raises an error if any file is missing or has invalid format.
    
    Returns:
        Tuple of (load_df, solar_df, wind_df)
    """
    # Define paths to CSV files
    load_csv = os.path.join(project_root, 'data', 'processed', 'load-2023.csv')
    solar_csv = os.path.join(project_root, 'data', 'processed', 'solar-2023.csv')
    wind_csv = os.path.join(project_root, 'data', 'processed', 'wind-2023.csv')
    
    # Check if all files exist
    if not os.path.exists(load_csv):
        raise FileNotFoundError(f"Load data file not found: {load_csv}")
    if not os.path.exists(solar_csv):
        raise FileNotFoundError(f"Solar data file not found: {solar_csv}")
    if not os.path.exists(wind_csv):
        raise FileNotFoundError(f"Wind data file not found: {wind_csv}")
    
    # Load the CSV files
    print(f"Loading data from CSV files...")
    try:
        load_df = pd.read_csv(load_csv, parse_dates=['time'], index_col='time')
        solar_df = pd.read_csv(solar_csv, parse_dates=['time'], index_col='time')
        wind_df = pd.read_csv(wind_csv, parse_dates=['time'], index_col='time')
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")
    
    # Validate the data
    if load_df.empty or solar_df.empty or wind_df.empty:
        raise ValueError("One or more of the loaded dataframes is empty")
    
    # Make sure all dataframes have the same index
    if not load_df.index.equals(solar_df.index) or not load_df.index.equals(wind_df.index):
        raise ValueError("The time indices of the load, solar, and wind data do not match")
    
    print(f"Successfully loaded {len(load_df)} data points from each CSV file")
    
    return load_df, solar_df, wind_df

def extract_representative_weeks(load_df, solar_df, wind_df):
    """
    Extract the representative weeks from the full-year data.
    - Winter week (Week 2): January 9-15
    - Summer week (Week 31): July 31-August 6
    - Spring/Autumn week (Week 43): October 23-29
    
    Args:
        load_df: DataFrame with load data
        solar_df: DataFrame with solar data
        wind_df: DataFrame with wind data
        
    Returns:
        Dictionary with representative week data
    """
    # Define the date ranges for each representative week
    winter_start = datetime(2023, 1, 9)
    winter_end = winter_start + timedelta(days=7) - timedelta(seconds=1)
    
    summer_start = datetime(2023, 7, 31)
    summer_end = summer_start + timedelta(days=7) - timedelta(seconds=1)
    
    spring_autumn_start = datetime(2023, 10, 23)
    spring_autumn_end = spring_autumn_start + timedelta(days=7) - timedelta(seconds=1)
    
    # Extract the data for each week
    try:
        winter_load = load_df.loc[winter_start:winter_end].copy()
        winter_solar = solar_df.loc[winter_start:winter_end].copy()
        winter_wind = wind_df.loc[winter_start:winter_end].copy()
        
        summer_load = load_df.loc[summer_start:summer_end].copy()
        summer_solar = solar_df.loc[summer_start:summer_end].copy()
        summer_wind = wind_df.loc[summer_start:summer_end].copy()
        
        spring_autumn_load = load_df.loc[spring_autumn_start:spring_autumn_end].copy()
        spring_autumn_solar = solar_df.loc[spring_autumn_start:spring_autumn_end].copy()
        spring_autumn_wind = wind_df.loc[spring_autumn_start:spring_autumn_end].copy()
    except KeyError as e:
        raise ValueError(f"Failed to extract representative weeks: {str(e)}")
    
    # Verify that we have exactly 168 hours (7 days) for each week
    weeks = {
        'winter': (winter_load, winter_solar, winter_wind),
        'summer': (summer_load, summer_solar, summer_wind),
        'spring_autumn': (spring_autumn_load, spring_autumn_solar, spring_autumn_wind)
    }
    
    for season, (ld, sl, wn) in weeks.items():
        if len(ld) != 168:
            raise ValueError(f"{season} load data has {len(ld)} rows instead of 168")
        if len(sl) != 168:
            raise ValueError(f"{season} solar data has {len(sl)} rows instead of 168")
        if len(wn) != 168:
            raise ValueError(f"{season} wind data has {len(wn)} rows instead of 168")
    
    print(f"Successfully extracted representative weeks:")
    print(f"- Winter week (Week 2): {len(winter_load)} hours")
    print(f"- Summer week (Week 31): {len(summer_load)} hours")
    print(f"- Spring/Autumn week (Week 43): {len(spring_autumn_load)} hours")
    
    # Return the data in a structured format
    return {
        'winter': {
            'load': winter_load,
            'solar': winter_solar,
            'wind': winter_wind,
            'times': list(winter_load.index)
        },
        'summer': {
            'load': summer_load,
            'solar': summer_solar,
            'wind': summer_wind,
            'times': list(summer_load.index)
        },
        'spring_autumn': {
            'load': spring_autumn_load,
            'solar': spring_autumn_solar,
            'wind': spring_autumn_wind,
            'times': list(spring_autumn_load.index)
        }
    }

if __name__ == "__main__":
    # Test the functions
    try:
        load_df, solar_df, wind_df = load_csv_data()
        weeks_data = extract_representative_weeks(load_df, solar_df, wind_df)
        
        # Print some statistics
        for season, data in weeks_data.items():
            print(f"\n{season.capitalize()} Week Statistics:")
            print(f"  Load: min={data['load'].min().iloc[0]:.2f}, max={data['load'].max().iloc[0]:.2f}, mean={data['load'].mean().iloc[0]:.2f}")
            print(f"  Solar: min={data['solar'].min().iloc[0]:.2f}, max={data['solar'].max().iloc[0]:.2f}, mean={data['solar'].mean().iloc[0]:.2f}")
            print(f"  Wind: min={data['wind'].min().iloc[0]:.2f}, max={data['wind'].max().iloc[0]:.2f}, mean={data['wind'].mean().iloc[0]:.2f}")
            
    except Exception as e:
        print(f"Error: {str(e)}") 