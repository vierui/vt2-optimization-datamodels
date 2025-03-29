"""
Data loader module for PyPSA-like grid data
"""
import os
import pandas as pd
import numpy as np
import datetime

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, index_col='id')

def load_grid_data(data_dir="data/grid"):
    """
    Load all grid component data from CSV files
    
    Args:
        data_dir: Directory containing grid component CSV files
        
    Returns:
        Dictionary with DataFrames for each component type
    """
    # Ensure paths are relative to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one more level to the project root
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, data_dir)
    
    print(f"Loading data from: {data_path}")
    
    # Load each component type
    buses = load_csv(os.path.join(data_path, "buses.csv"))
    generators = load_csv(os.path.join(data_path, "generators.csv"))
    loads = load_csv(os.path.join(data_path, "loads.csv"))
    
    # Handle possible naming difference for storage_units
    try:
        storage_units = load_csv(os.path.join(data_path, "storage_units.csv"))
    except FileNotFoundError:
        try:
            storage_units = load_csv(os.path.join(data_path, "storages.csv"))
        except FileNotFoundError:
            print("No storage file found, using empty DataFrame")
            storage_units = pd.DataFrame(columns=['name', 'bus', 'p_mw', 'energy_mwh', 
                                                'charge_efficiency', 'discharge_efficiency'])
    
    lines = load_csv(os.path.join(data_path, "lines.csv"))
    
    # Create time series data for loads
    T = 24  # Default to 24 hours
    time_index = pd.RangeIndex(T)
    loads_t = pd.DataFrame(index=time_index)
    
    # Create constant load profiles based on p_mw column
    for load_id, load_data in loads.iterrows():
        loads_t[load_id] = [load_data['p_mw']] * T
    
    return {
        'buses': buses,
        'generators': generators,
        'loads': loads,
        'storage_units': storage_units,
        'lines': lines,
        'loads_t': loads_t,
        'T': T
    }

def create_time_series(T=24):
    """
    Create time series indices
    
    Args:
        T: Number of time steps
    
    Returns:
        Time index
    """
    return pd.RangeIndex(T)

def load_day_profiles(day=10, data_dir="data/processed"):
    """
    Load time series data for a specific day of the year
    
    Args:
        day: Day of the year (1-365)
        data_dir: Directory containing processed time series data
        
    Returns:
        Dictionary with DataFrames for load, wind, and solar profiles
    """
    # Ensure paths are relative to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, data_dir)
    
    print(f"Loading time series data from: {data_path}")
    
    # Calculate the start date for the given day of the year
    start_date = datetime.datetime(2023, 1, 1) + datetime.timedelta(days=day-1)
    end_date = start_date + datetime.timedelta(days=1)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Extracting data for day {day} of the year: {start_str}")
    
    # Load time series data
    load_df = pd.read_csv(os.path.join(data_path, "load-2023.csv"))
    wind_df = pd.read_csv(os.path.join(data_path, "wind-2023.csv"))
    solar_df = pd.read_csv(os.path.join(data_path, "solar-2023.csv"))
    
    # Convert time column to datetime
    load_df['time'] = pd.to_datetime(load_df['time'])
    wind_df['time'] = pd.to_datetime(wind_df['time'])
    solar_df['time'] = pd.to_datetime(solar_df['time'])
    
    # Filter data for the specified day
    load_day = load_df[(load_df['time'] >= start_str) & (load_df['time'] < end_str)]
    wind_day = wind_df[(wind_df['time'] >= start_str) & (wind_df['time'] < end_str)]
    solar_day = solar_df[(solar_df['time'] >= start_str) & (solar_df['time'] < end_str)]
    
    # Reset indices for the filtered data
    load_day = load_day.reset_index(drop=True)
    wind_day = wind_day.reset_index(drop=True)
    solar_day = solar_day.reset_index(drop=True)
    
    return {
        'load': load_day['value'].values,
        'wind': wind_day['value'].values,
        'solar': solar_day['value'].values,
        'T': len(load_day)
    } 