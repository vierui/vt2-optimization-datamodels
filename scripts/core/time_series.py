"""
Functions for building time series data for scenarios.
"""
import pandas as pd

def build_gen_time_series(master_gen, gen_positions_dict, storage_positions_dict, season_key):
    """Build generation time series for the scenario."""
    scenario_gen = master_gen[master_gen["season"] == season_key].copy()
    merged_positions = {**gen_positions_dict, **storage_positions_dict}
    selected_ids = list(merged_positions.values())
    scenario_gen = scenario_gen[scenario_gen["id"].isin(selected_ids)].copy()
    
    for bus_i, gen_id in merged_positions.items():
        scenario_gen.loc[scenario_gen["id"] == gen_id, "bus"] = bus_i
    
    scenario_gen.drop_duplicates(subset=['time', 'id'], inplace=True)
    scenario_gen.sort_values(["time", "id"], inplace=True)
    scenario_gen.reset_index(drop=True, inplace=True)
    return scenario_gen

def build_demand_time_series(master_load, load_factor, season_key):
    """Build demand time series for the scenario."""
    scenario_load = master_load[master_load["season"] == season_key].copy()
    scenario_load["pd"] *= load_factor
    scenario_load.drop_duplicates(subset=['time', 'bus'], inplace=True)
    scenario_load.sort_values(["time", "bus"], inplace=True)
    scenario_load.reset_index(drop=True, inplace=True)
    return scenario_load 