"""
Data loader module for PyPSA-like grid data
"""
import os
import pandas as pd
import numpy as np

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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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