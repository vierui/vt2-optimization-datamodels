#!/usr/bin/env python3

"""
investment_utils.py

Utility functions for multi-stage investment modeling with the lifetime-based approach.
This module provides helper functions to:
1. Calculate lifetime periods based on asset lifetimes
2. Process investment periods 
3. Prepare data for the investment optimization model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset

def safe_float(value):
    """Convert any value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} to float. Using 0.0 instead.")
        return 0.0

# Make sure all numeric values are properly converted to float
def ensure_float_list(values):
    """Ensure all values in a list are properly converted to float."""
    return [safe_float(v) for v in values]

def calculate_lifetime_periods(asset_lifetime, planning_horizon):
    """
    Calculate the lifetime periods for an asset based on its lifetime and planning horizon.
    
    Args:
        asset_lifetime (int): Lifetime of the asset in years
        planning_horizon (int): Total planning horizon in years
    
    Returns:
        Tuple of:
        - full_periods (int): Number of full lifetime periods
        - remainder (int): Remainder (partial lifetime length, 0 if no remainder)
        - total_periods (int): Total number of lifetime periods including the potential remainder
    """
    full_periods = planning_horizon // asset_lifetime
    remainder = planning_horizon % asset_lifetime
    total_periods = full_periods + (1 if remainder > 0 else 0)
    
    return full_periods, remainder, total_periods

def create_lifetime_periods_mapping(asset_lifetimes, planning_horizon):
    """
    Create a mapping of asset lifetime periods to calendar years.
    
    Args:
        asset_lifetimes (dict): Dictionary mapping asset IDs to their lifetimes in years
        planning_horizon (int): Total planning horizon in years
    
    Returns:
        Dict: For each asset, maps lifetime period indices to the corresponding time periods
    """
    lifetime_periods = {}
    
    for asset_id, lifetime in asset_lifetimes.items():
        full_periods, remainder, total_periods = calculate_lifetime_periods(lifetime, planning_horizon)
        
        # Initialize for this asset
        lifetime_periods[asset_id] = {}
        
        # Map each lifetime period to its time periods
        for period in range(full_periods):
            start_year = period * lifetime
            end_year = (period + 1) * lifetime - 1  # inclusive end
            lifetime_periods[asset_id][period] = (start_year, end_year)
        
        # Add the remainder period if it exists
        if remainder > 0:
            start_year = full_periods * lifetime
            end_year = planning_horizon - 1  # inclusive end
            lifetime_periods[asset_id][full_periods] = (start_year, end_year)
    
    return lifetime_periods

def create_representative_periods(lifetime_periods, annual_periods):
    """
    Create representative operational periods for each lifetime period.
    
    Args:
        lifetime_periods (dict): Mapping of asset lifetime periods to time periods 
        annual_periods (list): List of representative time periods for a year
        
    Returns:
        Dict: Maps each lifetime period to its representative periods
    """
    representative_periods = {}
    
    for asset_id, periods in lifetime_periods.items():
        representative_periods[asset_id] = {}
        
        for period_idx, (start_year, end_year) in periods.items():
            # Create periods for this lifetime period
            period_duration = end_year - start_year + 1  # inclusive range
            
            # For representative periods, we use the annual ones scaled by duration
            representative_periods[asset_id][period_idx] = {
                'periods': annual_periods,
                'scaling_factor': period_duration
            }
    
    return representative_periods

def create_typical_periods(start_date, num_periods=4, hours_per_period=24):
    """
    Create typical periods for operational representation.
    
    Args:
        start_date: Starting date for the periods
        num_periods: Number of typical periods (e.g., seasons)
        hours_per_period: Hours in each period
        
    Returns:
        List of datetime objects representing the periods
    """
    periods = []
    current_date = start_date
    
    for i in range(num_periods):
        for h in range(hours_per_period):
            periods.append(current_date + DateOffset(hours=h))
        current_date += DateOffset(days=90)  # Roughly quarterly
    
    return periods

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

def scale_opex_by_lifetime_period(dcopf_results, representative_periods):
    """
    Scale operational costs by the duration of each lifetime period.
    
    Args:
        dcopf_results (dict): Results from DCOPF with operational costs
        representative_periods (dict): Dict of representative periods by lifetime period
        
    Returns:
        dict: Updated DCOPF results with scaled costs
    """
    # For each lifetime period, scale the operational cost
    scaled_results = {}
    
    for period_id, period_result in dcopf_results.items():
        # Get the scaling factor for this lifetime period
        asset_id = period_result.get('asset_id')
        scaling_factor = representative_periods[asset_id][period_id]['scaling_factor']
        
        # Scale the operational cost
        scaled_cost = period_result['cost'] * scaling_factor
        
        # Update the result
        scaled_result = period_result.copy()
        scaled_result['scaled_cost'] = scaled_cost
        scaled_results[period_id] = scaled_result
    
    return scaled_results

def calculate_investment_costs(investment_decisions, asset_lifetimes, asset_capex):
    """
    Calculate investment costs based on decisions.
    
    Args:
        investment_decisions (dict): Dict mapping (asset_id, period_idx) to binary decision
        asset_lifetimes (dict): Dict mapping asset IDs to lifetimes in years
        asset_capex (dict): Dict mapping asset IDs to capital costs
        
    Returns:
        dict: Investment costs by asset and lifetime period
    """
    investment_costs = {}
    
    for (asset_id, period_idx), decision in investment_decisions.items():
        if decision == 1:  # If this investment is chosen
            # Calculate the annual cost
            annual_cost = asset_capex[asset_id] / asset_lifetimes[asset_id]
            
            # Store the cost for this decision
            investment_costs[(asset_id, period_idx)] = annual_cost
    
    return investment_costs 