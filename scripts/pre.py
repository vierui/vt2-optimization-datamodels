#!/usr/bin/env python3
"""
Pre-processing module for data preparation

This script handles all data preprocessing tasks in one place:
1. Loads raw data directly from processed directory
2. Extracts data for the three representative weeks (winter, summer, spring/autumn)
3. Returns normalized data structures ready for optimization
"""

import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import json

# Constants for representative seasons - centralized here
SEASON_WEEKS = {
    'winter': 2,      # Week 2 (January)
    'summer': 32,     # Week 32 (August)
    'spri_autu': 43   # Week 43 (October/November)
}

# Constants for annual calculation
SEASON_WEIGHTS = {
    'winter': 13,     # Winter represents 13 weeks
    'summer': 13,     # Summer represents 13 weeks
    'spri_autu': 26   # Spring/Autumn represents 26 weeks
}

def get_week_dates(year, week_number):
    """
    Get the start and end dates for a specific week of the year
    
    Args:
        year: The year
        week_number: Week of the year (1-52)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    # Create a datetime for January 1st of the given year
    jan1 = datetime(year, 1, 1)
    
    # Calculate days since the first day of the year
    days_offset = (week_number - 1) * 7
    
    # Calculate the target date (Monday of the target week)
    target_date = jan1 + timedelta(days=days_offset)
    
    # Adjust to get to Monday (weekday 0)
    target_monday = target_date - timedelta(days=target_date.weekday())
    
    # Calculate the end date (Sunday)
    target_sunday = target_monday + timedelta(days=6)
    
    return target_monday, target_sunday

def load_grid_data(base_dir='data/grid'):
    """
    Load the grid configuration data (generators, loads, etc.)
    
    Args:
        base_dir: Base directory for grid data
        
    Returns:
        Dictionary containing dataframes for each component type
    """
    components = {}
    
    # Load generators
    generators_path = os.path.join(base_dir, 'generators.csv')
    if os.path.exists(generators_path):
        components['generators'] = pd.read_csv(generators_path)
    
    # Load loads
    loads_path = os.path.join(base_dir, 'loads.csv')
    if os.path.exists(loads_path):
        components['loads'] = pd.read_csv(loads_path)
    
    # Load buses
    buses_path = os.path.join(base_dir, 'buses.csv')
    if os.path.exists(buses_path):
        components['buses'] = pd.read_csv(buses_path)

    # Load lines
    lines_path = os.path.join(base_dir, 'lines.csv')
    if os.path.exists(lines_path):
        components['lines'] = pd.read_csv(lines_path)
    
    # Load storage units
    try:
        storage_path = os.path.join(base_dir, 'storage_units.csv')
        components['storage_units'] = pd.read_csv(storage_path)
    except:
        try:
            storage_path = os.path.join(base_dir, 'storages.csv')
            components['storage_units'] = pd.read_csv(storage_path)
        except:
            print("No storage data found")
    
    # Load analysis data (planning horizon, etc.)
    analysis_path = os.path.join(base_dir, 'analysis.json')
    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, 'r') as f:
                analysis_data = json.load(f)
            
            # Convert absolute years to relative years
            if 'planning_horizon' in analysis_data and 'years' in analysis_data['planning_horizon']:
                absolute_years = analysis_data['planning_horizon']['years']
                
                # Use the first year as the base year
                base_year = absolute_years[0] if absolute_years else 2023
                
                # Create relative years (1, 2, 3...) while preserving the original years
                relative_years = list(range(1, len(absolute_years) + 1))
                year_mapping = dict(zip(absolute_years, relative_years))
                inverse_mapping = dict(zip(relative_years, absolute_years))
                
                # Replace absolute years with relative years
                analysis_data['planning_horizon']['absolute_years'] = absolute_years.copy()
                analysis_data['planning_horizon']['years'] = relative_years
                analysis_data['planning_horizon']['base_year'] = base_year
                analysis_data['planning_horizon']['year_mapping'] = year_mapping
                analysis_data['planning_horizon']['inverse_mapping'] = inverse_mapping
                
                # Convert load growth factors to use relative years
                if 'load_growth' in analysis_data:
                    relative_load_growth = {}
                    for year, factor in analysis_data['load_growth'].items():
                        if year.isdigit() and int(year) in year_mapping:
                            relative_load_growth[str(year_mapping[int(year)])] = factor
                        elif year != "description":  # Skip description field
                            # Handle case where the year is already in relative form
                            relative_load_growth[year] = factor
                    
                    # Preserve original absolute year mapping
                    analysis_data['load_growth_absolute'] = analysis_data['load_growth'].copy()
                    # Update with relative year mapping
                    analysis_data['load_growth'] = relative_load_growth
                
                # Remove unused features
                if 'renewable_availability_factors' in analysis_data:
                    print("Note: 'renewable_availability_factors' in analysis.json is not being used")
                
                if 'asset_lifetime_extensions' in analysis_data:
                    print("Note: 'asset_lifetime_extensions' in analysis.json is not being used")
                    
                if 'cost_learning_curves' in analysis_data:
                    print("Note: 'cost_learning_curves' in analysis.json is not being used")
            
            components['analysis'] = analysis_data
            print(f"Loaded and processed analysis data from {analysis_path}")
            print(f"Using relative years instead of absolute years")
        except Exception as e:
            print(f"Error loading or processing analysis data: {e}")
            import traceback
            traceback.print_exc()
    
    return components

def load_processed_data(data_dir='data/processed'):
    """
    Load pre-processed time series data for solar, wind and load
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Dictionary with processed dataframes
    """
    processed = {}
    
    # Load data files
    file_mapping = {
        'solar': os.path.join(data_dir, 'solar-2023.csv'),
        'wind': os.path.join(data_dir, 'wind-2023.csv'),
        'load': os.path.join(data_dir, 'load-2023.csv')
    }
    
    for data_type, filepath in file_mapping.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=['time'])
            processed[data_type] = df
        else:
            print(f"Warning: {filepath} not found")
    
    return processed

def extract_season_data(processed_data, season, year=2023):
    """
    Extract data for a specific season
    
    Args:
        processed_data: Dictionary with processed data
        season: Season name (winter, summer, spri_autu)
        year: Year for the data
        
    Returns:
        Dictionary with data for the specified season
    """
    if season not in SEASON_WEEKS:
        raise ValueError(f"Invalid season: {season}. Must be one of: {', '.join(SEASON_WEEKS.keys())}")
    
    week_number = SEASON_WEEKS[season]
    start_date, end_date = get_week_dates(year, week_number)
    
    print(f"Extracting data for {season} season: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Extract data for the week
    season_data = {}
    
    for data_type, df in processed_data.items():
        # Filter data for the week
        mask = (df['time'] >= start_date) & (df['time'] <= end_date)
        season_data[data_type] = df[mask].copy()
    
    return season_data

def prepare_season_profiles(grid_data, processed_data):
    """
    Prepare normalized profiles for all seasons
    
    Args:
        grid_data: Dictionary with grid component data
        processed_data: Dictionary with processed time series data
        
    Returns:
        Dictionary with normalized profiles for each season
    """
    seasons_profiles = {}
    
    for season in SEASON_WEEKS.keys():
        # Extract data for this season
        season_data = extract_season_data(processed_data, season)
        
        # Prepare generators and loads normalized profiles
        gen_profiles = prepare_generator_profiles(grid_data, season_data, processed_data)
        load_profiles = prepare_load_profiles(grid_data, season_data, processed_data)
        
        # Store profiles for this season
        seasons_profiles[season] = {
            'generators': gen_profiles,
            'loads': load_profiles,
            'time_index': season_data['load']['time'] if 'load' in season_data else None,
            'hours': len(season_data['load']) if 'load' in season_data else 168  # Default to 168 hours (1 week)
        }
    
    return seasons_profiles

def prepare_generator_profiles(grid_data, season_data, processed_data):
    """
    Prepare normalized generator profiles
    
    Args:
        grid_data: Dictionary with grid component data
        season_data: Dictionary with season-specific data
        processed_data: Dictionary with all processed data
        
    Returns:
        DataFrame with normalized generator profiles
    """
    if 'generators' not in grid_data:
        return pd.DataFrame()
    
    # Create a multi-index DataFrame for generators
    index = pd.MultiIndex.from_product(
        [season_data['load']['time'], grid_data['generators']['id']],
        names=['time', 'generator_id']
    )
    gen_profiles = pd.DataFrame(index=index, columns=['p_max_pu'])
    
    # For each generator, create its profile
    for _, gen in grid_data['generators'].iterrows():
        gen_id = gen['id']
        gen_type = gen['type']
        
        # Get the right data source based on generator type
        if gen_type == 'solar' and 'solar' in season_data:
            source_data = season_data['solar']
            max_value = processed_data['solar']['value'].max()
        elif gen_type == 'wind' and 'wind' in season_data:
            source_data = season_data['wind']
            max_value = processed_data['wind']['value'].max()
        else:  # thermal or other dispatchable
            # For thermal, use constant availability of 1.0
            for t in season_data['load']['time']:
                gen_profiles.loc[(t, gen_id), 'p_max_pu'] = 1.0
            continue
        
        # Normalize to per-unit (0-1 scale)
        normalized = source_data['value'] / max_value if max_value > 0 else 1.0
        
        # Assign values to the DataFrame
        for i, t in enumerate(source_data['time']):
            gen_profiles.loc[(t, gen_id), 'p_max_pu'] = normalized.iloc[i] if not pd.isna(normalized.iloc[i]) else 1.0
    
    return gen_profiles

def prepare_load_profiles(grid_data, season_data, processed_data):
    """
    Prepare normalized load profiles
    
    Args:
        grid_data: Dictionary with grid component data
        season_data: Dictionary with season-specific data
        processed_data: Dictionary with all processed data
        
    Returns:
        DataFrame with normalized load profiles
    """
    if 'loads' not in grid_data or 'load' not in season_data:
        return pd.DataFrame()
    
    # Create a multi-index DataFrame for loads
    index = pd.MultiIndex.from_product(
        [season_data['load']['time'], grid_data['loads']['id']],
        names=['time', 'load_id']
    )
    load_profiles = pd.DataFrame(index=index, columns=['p_pu'])
    
    # Normalize load data
    max_load = processed_data['load']['value'].max()
    normalized = season_data['load']['value'] / max_load if max_load > 0 else 1.0
    
    # For each load, create its profile
    for _, load in grid_data['loads'].iterrows():
        load_id = load['id']
        
        # Assign values to the DataFrame
        for i, t in enumerate(season_data['load']['time']):
            load_profiles.loc[(t, load_id), 'p_pu'] = normalized.iloc[i] if not pd.isna(normalized.iloc[i]) else 1.0
    
    return load_profiles

def process_data_for_optimization(grid_dir='data/grid', processed_dir='data/processed', planning_years=10):
    """
    Main function to process data for optimization
    
    Args:
        grid_dir: Directory with grid component data
        processed_dir: Directory with processed time series data
        planning_years: Number of years in the planning horizon (default: 10)
        
    Returns:
        Dictionary with data structures ready for optimization
    """
    print("Starting data preprocessing...")
    
    # Load grid data
    print(f"Loading grid data from {grid_dir}...")
    grid_data = load_grid_data(grid_dir)
    
    # Customize the planning horizon based on the specified number of years
    if 'analysis' in grid_data:
        if 'planning_horizon' not in grid_data['analysis']:
            grid_data['analysis']['planning_horizon'] = {}
        
        # Create or update the years list - ensure we respect the full planning horizon
        relative_years = list(range(1, planning_years + 1))
        grid_data['analysis']['planning_horizon']['years'] = relative_years
        
        # If no absolute years mapping exists, create one
        if 'absolute_years' not in grid_data['analysis']['planning_horizon']:
            base_year = grid_data['analysis'].get('base_year', 2023)
            absolute_years = [base_year + i - 1 for i in relative_years]
            grid_data['analysis']['planning_horizon']['absolute_years'] = absolute_years
            
            # Create mappings
            year_mapping = dict(zip(absolute_years, relative_years))
            inverse_mapping = dict(zip(relative_years, absolute_years))
            grid_data['analysis']['planning_horizon']['year_mapping'] = year_mapping
            grid_data['analysis']['planning_horizon']['inverse_mapping'] = inverse_mapping
        else:
            # Trim or extend the existing absolute years list
            existing_abs_years = grid_data['analysis']['planning_horizon']['absolute_years']
            base_year = existing_abs_years[0] if existing_abs_years else 2023
            
            if len(existing_abs_years) < planning_years:
                # Extend the list
                last_year = existing_abs_years[-1] if existing_abs_years else base_year
                for i in range(len(existing_abs_years) + 1, planning_years + 1):
                    existing_abs_years.append(last_year + i - len(existing_abs_years))
            
            # Trim if needed
            absolute_years = existing_abs_years[:planning_years]
            grid_data['analysis']['planning_horizon']['absolute_years'] = absolute_years
            
            # Update mappings
            year_mapping = dict(zip(absolute_years, relative_years))
            inverse_mapping = dict(zip(relative_years, absolute_years))
            grid_data['analysis']['planning_horizon']['year_mapping'] = year_mapping
            grid_data['analysis']['planning_horizon']['inverse_mapping'] = inverse_mapping
        
        # Ensure load growth factors are specified for all years
        if 'load_growth' not in grid_data['analysis']:
            grid_data['analysis']['load_growth'] = {}
        
        load_growth = grid_data['analysis']['load_growth']
        for i, year in enumerate(relative_years):
            year_str = str(year)
            if year_str not in load_growth and year_str.isdigit():
                # Default to 5% annual growth
                load_growth[year_str] = 1.0 + (i * 0.05)
        
        print(f"Planning horizon set to {planning_years} years")
        print(f"Relative years: {relative_years}")
        print(f"Absolute years: {grid_data['analysis']['planning_horizon']['absolute_years']}")
        print(f"Load growth factors: {load_growth}")
    
    # Load processed data
    print(f"Loading processed time series data from {processed_dir}...")
    processed_data = load_processed_data(processed_dir)
    
    # Prepare season profiles
    print("Preparing season profiles...")
    seasons_profiles = prepare_season_profiles(grid_data, processed_data)
    
    # Return a dictionary with all the data needed for optimization
    print("Data preprocessing completed.")
    return {
        'grid_data': grid_data,
        'seasons_profiles': seasons_profiles
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for power grid optimization')
    parser.add_argument('--grid-dir', type=str, default='data/grid',
                      help='Directory with grid component data')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                      help='Directory with processed time series data')
    
    args = parser.parse_args()
    
    # Process data and print summary
    data = process_data_for_optimization(args.grid_dir, args.processed_dir)
    
    # Print a summary of the data
    print("\nData summary:")
    print(f"Grid components:")
    for component, df in data['grid_data'].items():
        print(f"  - {component}: {len(df)} items")
    
    print("\nSeason profiles:")
    for season, profiles in data['seasons_profiles'].items():
        print(f"  - {season}: {profiles['hours']} hours")
        print(f"    - Generators: {len(profiles['generators'].index.levels[1]) if not profiles['generators'].empty else 0} generators")
        print(f"    - Loads: {len(profiles['loads'].index.levels[1]) if not profiles['loads'].empty else 0} loads") 