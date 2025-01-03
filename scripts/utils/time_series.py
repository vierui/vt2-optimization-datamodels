"""
Functions for building time series data for the DCOPF model.
"""
import pandas as pd

def build_gen_time_series(master_gen: pd.DataFrame, gen_positions: dict, 
                         storage_positions: dict, season: str) -> pd.DataFrame:
    """Build generation time series for a specific season"""
    # Filter master_gen for the season
    season_gen = master_gen[master_gen['time'].dt.strftime('%B').isin(get_season_months(season))]
    
    # Create generation time series
    gen_ts = []
    
    # Add generators
    for bus, gen_id in gen_positions.items():
        gen_data = season_gen[season_gen['id'] == gen_id].copy()
        gen_data['bus'] = bus
        gen_ts.append(gen_data)
    
    # Add storage units
    for bus, storage_id in storage_positions.items():
        storage_data = season_gen[season_gen['id'] == storage_id].copy()
        storage_data['bus'] = bus
        gen_ts.append(storage_data)
    
    if gen_ts:
        return pd.concat(gen_ts, ignore_index=True)
    return pd.DataFrame()

def build_demand_time_series(master_load: pd.DataFrame, load_factor: float, season: str) -> pd.DataFrame:
    """Build demand time series for a specific season"""
    # Filter master_load for the season
    season_load = master_load[master_load['time'].dt.strftime('%B').isin(get_season_months(season))]
    
    # Apply load factor
    season_load = season_load.copy()
    season_load['pd'] = season_load['pd'] * load_factor
    
    return season_load

def get_season_months(season: str) -> list:
    """Get list of months for a given season"""
    season_months = {
        'winter': ['December', 'January', 'February'],
        'summer': ['June', 'July', 'August'],
        'autumn_spring': ['March', 'April', 'May', 'September', 'October', 'November']
    }
    return season_months.get(season, []) 