"""Core time series functionality for DCOPF model"""
import pandas as pd
from typing import Dict

def build_gen_time_series(master_gen: pd.DataFrame, gen_positions: Dict[int, int], 
                         storage_positions: Dict[int, int], season: str) -> pd.DataFrame:
    """Build generation time series for a specific season"""
    # Filter master_gen for the season
    season_gen = master_gen[master_gen['season'] == season]
    
    # Create empty DataFrame
    gen_ts = pd.DataFrame()
    
    # Add conventional generators
    for bus, gen_id in gen_positions.items():
        if gen_id in season_gen['id'].unique():
            bus_gen = season_gen[season_gen['id'] == gen_id].copy()
            bus_gen['bus'] = bus
            gen_ts = pd.concat([gen_ts, bus_gen])
    
    # Add storage units
    for bus, storage_id in storage_positions.items():
        if storage_id in season_gen['id'].unique():
            storage_data = season_gen[season_gen['id'] == storage_id].copy()
            storage_data['bus'] = bus
            gen_ts = pd.concat([gen_ts, storage_data])
    
    return gen_ts

def build_demand_time_series(master_load: pd.DataFrame, load_factor: float, season: str) -> pd.DataFrame:
    """Build demand time series for a specific season"""
    # Filter for season and apply load factor
    season_load = master_load[master_load['season'] == season].copy()
    season_load['pd'] = season_load['pd'] * load_factor
    return season_load 