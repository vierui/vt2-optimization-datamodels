#!/usr/bin/env python3

"""
investment_utils.py

Utility functions for multi-stage investment modeling with the chunk-based approach.
This module provides helper functions to:
1. Calculate chunks based on asset lifetimes
2. Process investment periods 
3. Prepare data for the investment optimization model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset

def calculate_chunks(asset_lifetime, planning_horizon):
    """
    Calculate the chunks for an asset based on its lifetime and the planning horizon.
    
    Args:
        asset_lifetime (int): Lifetime of the asset in years
        planning_horizon (int): Total planning horizon in years
    
    Returns:
        Tuple of:
        - full_chunks (int): Number of full chunks
        - remainder (int): Remainder (partial chunk length, 0 if no remainder)
        - total_chunks (int): Total number of chunks including the potential remainder chunk
    """
    full_chunks = planning_horizon // asset_lifetime
    remainder = planning_horizon % asset_lifetime
    total_chunks = full_chunks + (1 if remainder > 0 else 0)
    
    return full_chunks, remainder, total_chunks

def create_chunk_periods(asset_lifetimes, planning_horizon):
    """
    Create a mapping of asset chunks to time periods.
    
    Args:
        asset_lifetimes (dict): Dictionary mapping asset IDs to their lifetimes in years
        planning_horizon (int): Total planning horizon in years
    
    Returns:
        Dict: For each asset, maps chunk indices to the corresponding time periods
    """
    chunk_periods = {}
    
    for asset_id, lifetime in asset_lifetimes.items():
        full_chunks, remainder, total_chunks = calculate_chunks(lifetime, planning_horizon)
        
        # Initialize for this asset
        chunk_periods[asset_id] = {}
        
        # Map each chunk to its time periods
        for chunk in range(full_chunks):
            start_year = chunk * lifetime
            end_year = (chunk + 1) * lifetime - 1  # inclusive end
            chunk_periods[asset_id][chunk] = (start_year, end_year)
        
        # Add the remainder chunk if it exists
        if remainder > 0:
            start_year = full_chunks * lifetime
            end_year = planning_horizon - 1  # inclusive end
            chunk_periods[asset_id][full_chunks] = (start_year, end_year)
    
    return chunk_periods

def create_representative_periods(chunk_periods, annual_periods):
    """
    Create representative periods for each chunk.
    
    Args:
        chunk_periods (dict): Mapping of asset chunks to time periods 
        annual_periods (list): List of representative time periods for a year
        
    Returns:
        Dict: Maps each chunk to its representative periods
    """
    representative_periods = {}
    
    for asset_id, chunks in chunk_periods.items():
        representative_periods[asset_id] = {}
        
        for chunk_idx, (start_year, end_year) in chunks.items():
            # Create periods for this chunk
            chunk_duration = end_year - start_year + 1  # inclusive range
            
            # For representative periods, we might just use the annual ones scaled
            # by the number of years in the chunk
            representative_periods[asset_id][chunk_idx] = {
                'periods': annual_periods,
                'scaling_factor': chunk_duration
            }
    
    return representative_periods

def create_gen_data_for_investment(base_gen_data, planning_horizon, start_year=2023):
    """
    Prepare generation data for investment modeling by adding investment-related fields.
    
    Args:
        base_gen_data (DataFrame): Base generator data
        planning_horizon (int): Planning horizon in years
        start_year (int): Starting year for the planning horizon
        
    Returns:
        DataFrame: Enhanced generator data with investment fields
    """
    # Make a copy to avoid modifying the original
    gen_data = base_gen_data.copy()
    
    # Add investment-related fields if they don't exist
    if 'lifetime' not in gen_data.columns:
        # Default to 20 years for traditional generators, 10 for renewables, 5 for storage
        gen_data['lifetime'] = 20
        # Shorter lifetime for renewables (if they can be identified)
        gen_data.loc[gen_data['gencost'] < 5, 'lifetime'] = 10  # Assuming low cost = renewables
        # Shorter lifetime for storage
        gen_data.loc[gen_data['emax'] > 0, 'lifetime'] = 5  # Storage
    
    if 'capex' not in gen_data.columns:
        # Default CAPEX in $/MW or $/MWh for storage
        gen_data['capex'] = 1000000  # $1M/MW
        # Higher for storage
        gen_data.loc[gen_data['emax'] > 0, 'capex'] = 300000  # $300k/MWh for storage
    
    if 'build_year' not in gen_data.columns:
        # Default all to first year of planning horizon
        gen_data['build_year'] = start_year
    
    return gen_data

def scale_opex_by_chunk(dcopf_results, representative_periods):
    """
    Scale operational costs by the duration of each chunk.
    
    Args:
        dcopf_results (dict): Results from DCOPF with operational costs
        representative_periods (dict): Dict of representative periods by chunk
        
    Returns:
        dict: Updated DCOPF results with scaled costs
    """
    # For each chunk, scale the operational cost
    scaled_results = {}
    
    for chunk_id, chunk_result in dcopf_results.items():
        # Get the scaling factor for this chunk
        asset_id = chunk_result.get('asset_id')
        scaling_factor = representative_periods[asset_id][chunk_id]['scaling_factor']
        
        # Scale the operational cost
        scaled_cost = chunk_result['cost'] * scaling_factor
        
        # Update the result
        scaled_result = chunk_result.copy()
        scaled_result['scaled_cost'] = scaled_cost
        scaled_results[chunk_id] = scaled_result
    
    return scaled_results

def calculate_investment_costs(investment_decisions, asset_lifetimes, asset_capex):
    """
    Calculate investment costs based on decisions.
    
    Args:
        investment_decisions (dict): Dict mapping (asset_id, chunk_idx) to binary decision
        asset_lifetimes (dict): Dict mapping asset IDs to lifetimes in years
        asset_capex (dict): Dict mapping asset IDs to capital costs
        
    Returns:
        dict: Investment costs by asset and chunk
    """
    investment_costs = {}
    
    for (asset_id, chunk_idx), decision in investment_decisions.items():
        if decision == 1:  # If this investment is chosen
            # Calculate the annual cost
            annual_cost = asset_capex[asset_id] / asset_lifetimes[asset_id]
            
            # Store the cost for this decision
            investment_costs[(asset_id, chunk_idx)] = annual_cost
    
    return investment_costs 