#!/usr/bin/env python3

"""
representative_weeks.py

Implementation of the representative weeks approach for power system optimization.
This script uses 3 representative weeks to approximate a full year of operations:
- Winter (Week 2): 13x weighting
- Summer (Week 31): 13x weighting
- Spring/Autumn (Week 43): 26x weighting

Each week consists of 168 consecutive hourly data points (7 days * 24 hours).
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import traceback

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Import DCOPF functions from local modules
from scripts.optimization import dcopf, investment_dcopf

# Define block information (representative weeks)
BLOCK_INFO = {
    'winter': {
        'week_num': 2,  # Week 2 (January)
        'weight': 13,  # represents ~13 winter weeks
        'start_hour': 0,  # first hour in the mega timeseries
    },
    'summer': {
        'week_num': 31,  # Week 31 (July)
        'weight': 13,  # represents ~13 summer weeks
        'start_hour': 168,  # starts after the winter block
    },
    'spring_autumn': {
        'week_num': 43,  # Week 43 (October)
        'weight': 26,  # represents ~26 spring/autumn weeks
        'start_hour': 336,  # starts after the summer block
    }
}

def get_week_dates(year, week_num):
    """
    Get start and end dates for a specific ISO week number.
    
    Args:
        year: Year
        week_num: ISO week number (1-53)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    # Find the first day of the year
    first_day = datetime(year, 1, 1)
    
    # Find the first day of week 1 (the first week containing a Thursday)
    if first_day.weekday() <= 3:  # Monday to Thursday
        # Week 1 contains the first day of the year
        start_of_week1 = first_day - timedelta(days=first_day.weekday())
    else:
        # Week 1 starts the following Monday
        start_of_week1 = first_day + timedelta(days=7 - first_day.weekday())
    
    # Calculate the start of the requested week
    start_date = start_of_week1 + timedelta(days=7 * (week_num - 1))
    
    # End date is 7 days later
    end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    return start_date, end_date

def extract_representative_weeks(df, year=2023):
    """
    Extract the representative weeks from a full-year DataFrame.
    
    Args:
        df: DataFrame with datetime index or 'time' column with datetime values
        year: Year for extracting the weeks
        
    Returns:
        Dictionary with keys 'winter', 'summer', 'spring_autumn' containing the extracted DataFrames
    """
    print("Extracting representative weeks...")
    
    # Check if the DataFrame has a 'season' column
    if 'season' in df.columns:
        print("Using season column to extract representative weeks")
        result = {}
        
        for block_name in BLOCK_INFO.keys():
            print(f"Extracting {block_name} week")
            # Extract data for this season
            season_df = df[df['season'] == block_name].copy()
            
            # Verify we have data
            if len(season_df) == 0:
                print(f"Warning: No data found for {block_name}")
                # Create an empty DataFrame with the same columns
                result[block_name] = pd.DataFrame(columns=df.columns)
            else:
                # Verify we have enough hours (should be 168 hours - 7 days * 24 hours)
                if len(season_df) < 168:
                    print(f"Warning: {block_name} data has only {len(season_df)} rows instead of 168")
                
                result[block_name] = season_df
        
        return result
    
    # If no season column, use the date-based approach
    # Ensure we have a datetime index
    if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time')
    
    # Extract the weeks
    result = {}
    for block_name, block_info in BLOCK_INFO.items():
        start_date, end_date = get_week_dates(year, block_info['week_num'])
        print(f"Extracting {block_name} week (Week {block_info['week_num']}): {start_date} to {end_date}")
        
        # Extract the week data
        week_data = df.loc[start_date:end_date].copy()
        
        # Verify we have 168 hours
        if len(week_data) != 168:
            print(f"Warning: {block_name} week has {len(week_data)} hours instead of 168")
            
            # If missing some hours at the end, fill with the last value
            if len(week_data) < 168:
                expected_index = pd.date_range(start=start_date, periods=168, freq='H')
                week_data = week_data.reindex(expected_index, method='ffill')
                print(f"  - Filled missing values. Now has {len(week_data)} hours.")
            
            # If too many hours, truncate
            if len(week_data) > 168:
                week_data = week_data.iloc[:168]
                print(f"  - Truncated to 168 hours.")
        
        result[block_name] = week_data
    
    return result

def create_mega_timeseries(week_data_dict):
    """
    Create a single 'mega timeseries' with all representative weeks.
    
    Args:
        week_data_dict: Dictionary with keys 'winter', 'summer', 'spring_autumn' containing week DataFrames
        
    Returns:
        DataFrame with the combined data and additional columns 'block' and 'weight'
    """
    print("Creating mega timeseries...")
    combined_data = []
    
    for block_name, block_info in BLOCK_INFO.items():
        # Get the week data
        week_data = week_data_dict[block_name]
        if week_data.empty:
            print(f"Warning: No data for {block_name} week, skipping")
            continue
            
        # Create a new index for this block
        block_start = block_info['start_hour']
        new_index = range(block_start, block_start + len(week_data))
        
        # Create a copy with the new index
        block_data = week_data.copy()
        
        # If the data has a DatetimeIndex, reset it
        if isinstance(block_data.index, pd.DatetimeIndex):
            block_data = block_data.reset_index()
        
        # Add the index hour, block name, and weight
        block_data['index_hour'] = new_index
        block_data['block'] = block_name
        block_data['weight'] = block_info['weight']
        
        combined_data.append(block_data)
    
    # Combine all blocks
    if not combined_data:
        print("Error: No data to combine in mega timeseries")
        return pd.DataFrame()
        
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Sort by the new index
    combined_df = combined_df.sort_values('index_hour').reset_index(drop=True)
    
    return combined_df

def prepare_generator_data(gen_data, mega_timeseries):
    """
    Prepare generator data for the DCOPF model.
    
    Args:
        gen_data: Base generator data
        mega_timeseries: The combined timeseries with block and weight information
        
    Returns:
        Generator data for the DCOPF model
    """
    print("Preparing generator data for the mega timeseries...")
    
    # Create a mapping from time to block and weight
    time_to_block = {}
    time_to_weight = {}
    
    # Create a time mapping for each hour in the mega timeseries
    base_date = datetime(2023, 1, 1)
    for block_name, block_info in BLOCK_INFO.items():
        block_start = block_info['start_hour']
        block_end = block_start + 167
        weight = block_info['weight']
        
        for hour in range(block_start, block_end + 1):
            synthetic_time = base_date + timedelta(hours=hour)
            time_to_block[synthetic_time] = block_name
            time_to_weight[synthetic_time] = weight
    
    # Create a new generator dataframe for the mega timeseries
    weighted_gen_data = []
    
    # Check if we have season information in the data
    if 'season' in gen_data.columns:
        print("Using season information from generator data")
        
        # For each synthetic time in the mega timeseries
        for synthetic_time, block_name in time_to_block.items():
            # Get hour of day and day of week
            hour_of_day = synthetic_time.hour
            day_of_week = synthetic_time.weekday()
            
            # Get all generator data for this season
            season_data = gen_data[gen_data['season'] == block_name]
            
            # Find data with the same hour of day and day of week
            matching_data = season_data[
                (season_data['time'].dt.hour == hour_of_day) & 
                (season_data['time'].dt.weekday == day_of_week)
            ]
            
            if not matching_data.empty:
                # Process each generator in the matching data
                for gen_id in matching_data['id'].unique():
                    gen_data_unit = matching_data[matching_data['id'] == gen_id]
                    if not gen_data_unit.empty:
                        new_row = gen_data_unit.iloc[0].copy()
                        new_row['time'] = synthetic_time
                        weighted_gen_data.append(new_row)
            else:
                # If no matching data found, try any data from that season with the same hour
                fallback_data = season_data[season_data['time'].dt.hour == hour_of_day]
                if not fallback_data.empty:
                    for gen_id in fallback_data['id'].unique():
                        gen_data_unit = fallback_data[fallback_data['id'] == gen_id]
                        if not gen_data_unit.empty:
                            new_row = gen_data_unit.iloc[0].copy()
                            new_row['time'] = synthetic_time
                            weighted_gen_data.append(new_row)
    else:
        # No season information, use a simpler approach
        print("No season information found in generator data, using all generator data")
        
        # Get unique generator IDs
        gen_ids = gen_data['id'].unique()
        
        # For each time in the mega timeseries and each generator
        for synthetic_time in time_to_block.keys():
            hour_of_day = synthetic_time.hour
            
            for gen_id in gen_ids:
                # Get all data for this generator
                gen_data_unit = gen_data[gen_data['id'] == gen_id]
                
                # Find data with matching hour
                matching_hour_data = gen_data_unit[gen_data_unit['time'].dt.hour == hour_of_day]
                
                if not matching_hour_data.empty:
                    # Use the first matching row
                    new_row = matching_hour_data.iloc[0].copy()
                    new_row['time'] = synthetic_time
                    weighted_gen_data.append(new_row)
                elif not gen_data_unit.empty:
                    # Use any row for this generator
                    new_row = gen_data_unit.iloc[0].copy()
                    new_row['time'] = synthetic_time
                    weighted_gen_data.append(new_row)
    
    result_df = pd.DataFrame(weighted_gen_data)
    print(f"Prepared generator data with {len(result_df)} rows")
    return result_df

def prepare_demand_data(demand_data, mega_timeseries):
    """
    Prepare demand data for the DCOPF model.
    
    Args:
        demand_data: Base demand data
        mega_timeseries: The combined timeseries with block and weight information
        
    Returns:
        Demand data for the DCOPF model
    """
    print("Preparing demand data for the mega timeseries...")
    
    # Create a mapping from time to block and weight
    time_to_block = {}
    time_to_weight = {}
    
    # Create a time mapping for each hour in the mega timeseries
    base_date = datetime(2023, 1, 1)
    for block_name, block_info in BLOCK_INFO.items():
        block_start = block_info['start_hour']
        block_end = block_start + 167
        weight = block_info['weight']
        
        for hour in range(block_start, block_end + 1):
            synthetic_time = base_date + timedelta(hours=hour)
            time_to_block[synthetic_time] = block_name
            time_to_weight[synthetic_time] = weight
    
    # Create a new demand dataframe for the mega timeseries
    weighted_demand_data = []
    
    # Check if we have season information in the data
    if 'season' in demand_data.columns:
        print("Using season information from demand data")
        
        # For each synthetic time in the mega timeseries
        for synthetic_time, block_name in time_to_block.items():
            # Get hour of day and day of week
            hour_of_day = synthetic_time.hour
            day_of_week = synthetic_time.weekday()
            
            # Get all demand data for this season
            season_data = demand_data[demand_data['season'] == block_name]
            
            # Find data with the same hour of day and day of week
            matching_data = season_data[
                (season_data['time'].dt.hour == hour_of_day) & 
                (season_data['time'].dt.weekday == day_of_week)
            ]
            
            if not matching_data.empty:
                # Process each bus in the matching data
                for bus_id in matching_data['bus'].unique():
                    bus_data = matching_data[matching_data['bus'] == bus_id]
                    if not bus_data.empty:
                        new_row = bus_data.iloc[0].copy()
                        new_row['time'] = synthetic_time
                        weighted_demand_data.append(new_row)
            else:
                # If no matching data found, try any data from that season with the same hour
                fallback_data = season_data[season_data['time'].dt.hour == hour_of_day]
                if not fallback_data.empty:
                    for bus_id in fallback_data['bus'].unique():
                        bus_data = fallback_data[fallback_data['bus'] == bus_id]
                        if not bus_data.empty:
                            new_row = bus_data.iloc[0].copy()
                            new_row['time'] = synthetic_time
                            weighted_demand_data.append(new_row)
    else:
        # No season information, use the old approach based on date matching
        print("No season information found, using date-based approach")
        
        # For each bus
        for bus_id in demand_data['bus'].unique():
            bus_rows = demand_data[demand_data['bus'] == bus_id]
            
            # For each time in the mega timeseries
            for synthetic_time, block_name in time_to_block.items():
                # Find the corresponding week in the original data
                # Extract hour of day (0-23) and day of week (0-6)
                hour_of_day = synthetic_time.hour
                day_of_week = synthetic_time.weekday()
                
                # Find matching rows in the original data for this block
                block_start, block_end = get_week_dates(2023, BLOCK_INFO[block_name]['week_num'])
                matching_rows = bus_rows[
                    (bus_rows['time'].dt.hour == hour_of_day) & 
                    (bus_rows['time'].dt.weekday == day_of_week) &
                    (bus_rows['time'] >= block_start) & 
                    (bus_rows['time'] <= block_end)
                ]
                
                if not matching_rows.empty:
                    # Use the first matching row
                    new_row = matching_rows.iloc[0].copy()
                    new_row['time'] = synthetic_time
                    weighted_demand_data.append(new_row)
                else:
                    # No matching data found, use a default value or skip
                    if not bus_rows.empty:
                        # Use the first row as a fallback
                        new_row = bus_rows.iloc[0].copy()
                        new_row['time'] = synthetic_time
                        new_row['pd'] = 0  # Default to zero load if no match found
                        weighted_demand_data.append(new_row)
    
    result_df = pd.DataFrame(weighted_demand_data)
    print(f"Prepared demand data with {len(result_df)} rows")
    return result_df

def run_representative_weeks_dcopf(gen_data, branch, bus, demand_data, year=2023, 
                              investment_mode=False, investment_candidates=None):
    """
    Run a DCOPF model using representative weeks for the full year.
    
    Args:
        gen_data: Generator data
        branch: Branch data
        bus: Bus data
        demand_data: Demand data
        year: Year to analyze
        investment_mode: Whether to run in investment mode
        investment_candidates: DataFrame containing investment candidates
        
    Returns:
        Results dictionary containing:
        - dispatch_df: Generation dispatch
        - flow_df: Branch flows
        - cost: Total cost
        - annual_cost: Annual cost (already weighted)
        - annual_generation: Annual generation by asset
        - annual_flows: Annual flows by branch
        - If investment_mode=True:
          - investment_decisions: Investment decisions
          - investment_cost: Investment cost
    """
    print("Running DCOPF with representative weeks approach...")
    
    # Step 1: Extract representative weeks from the data
    print("Step 1: Extracting representative weeks...")
    representative_weeks_data = extract_representative_weeks(demand_data, gen_data)
    
    # Step 2: Create the mega timeseries with all blocks
    print("Step 2: Creating mega timeseries...")
    mega_timeseries = create_mega_timeseries(representative_weeks_data)
    
    # Step 3: Prepare the data for the DCOPF model
    print("Step 3: Preparing weighted data for DCOPF...")
    weighted_demand_data = prepare_demand_data(demand_data, mega_timeseries)
    
    # Create a complete generator dataset for the mega timeseries
    # This ensures we have generator data for every time in the mega timeseries
    weighted_gen_data = prepare_generator_data(gen_data, mega_timeseries)
    
    # Print information about the generator data
    print(f"Prepared generator data with {len(weighted_gen_data)} rows")
    print(f"Generator data time range: {weighted_gen_data['time'].min()} to {weighted_gen_data['time'].max()}")
    print(f"Generator IDs in data: {weighted_gen_data['id'].unique()}")
    
    # Ensure we have generator data for the first hour (often a problem source)
    first_hour = datetime(2023, 1, 1, 0, 0, 0)
    first_hour_gens = weighted_gen_data[weighted_gen_data['time'] == first_hour]
    print(f"Generator data for first hour ({first_hour}): {len(first_hour_gens)} rows")
    if len(first_hour_gens) < len(weighted_gen_data['id'].unique()):
        print("WARNING: Missing generator data for first hour. Filling with data from first available time.")
        for gen_id in weighted_gen_data['id'].unique():
            if len(first_hour_gens[first_hour_gens['id'] == gen_id]) == 0:
                # Find the first data point for this generator
                gen_rows = weighted_gen_data[weighted_gen_data['id'] == gen_id]
                if not gen_rows.empty:
                    first_row = gen_rows.iloc[0].copy()
                    first_row['time'] = first_hour
                    weighted_gen_data = pd.concat([weighted_gen_data, pd.DataFrame([first_row])], ignore_index=True)
                    print(f"Added data for generator {gen_id} at first hour")
    
    # Step 4: Run the DCOPF optimization
    print("Step 4: Running DCOPF optimization...")
    
    try:
        if investment_mode and investment_candidates is not None:
            # Run in investment mode
            print("Running in investment mode with candidate assets")
            dcopf_results = run_dcopf(
                gen_data=weighted_gen_data,
                branch=branch,
                bus=bus,
                demand=weighted_demand_data,
                investment_mode=True,
                investment_candidates=investment_candidates
            )
        else:
            # Run in operations mode
            print("Running in operations mode")
            dcopf_results = run_dcopf(
                gen_data=weighted_gen_data,
                branch=branch,
                bus=bus,
                demand=weighted_demand_data,
                investment_mode=False
            )
            
        if dcopf_results is None:
            print("DCOPF optimization failed, returning empty results")
            # Return an empty results dictionary
            return {
                'dispatch_df': pd.DataFrame(),
                'flow_df': pd.DataFrame(),
                'storage_soc_df': pd.DataFrame(),
                'cost': 0,
                'annual_cost': 0,
                'annual_generation': {},
                'annual_flows': {}
            }
            
    except Exception as e:
        print(f"Error running DCOPF: {str(e)}")
        print(traceback.format_exc())
        # Return an empty results dictionary
        return {
            'dispatch_df': pd.DataFrame(),
            'flow_df': pd.DataFrame(),
            'storage_soc_df': pd.DataFrame(),
            'cost': 0,
            'annual_cost': 0,
            'annual_generation': {},
            'annual_flows': {}
        }
    
    # Step 5: Post-process the results
    print("Step 5: Post-processing results...")
    
    # Get the results from the DCOPF run
    dispatch_df = dcopf_results.get('dispatch_df', pd.DataFrame())
    flow_df = dcopf_results.get('flow_df', pd.DataFrame())
    storage_soc_df = dcopf_results.get('storage_soc_df', pd.DataFrame())
    objective_value = dcopf_results.get('objective_value', 0)
    
    # Create output dictionary
    results = {
        'dispatch_df': dispatch_df,
        'flow_df': flow_df,
        'storage_soc_df': storage_soc_df,
        'cost': objective_value,
        'annual_cost': objective_value  # The objective is already annualized
    }
    
    # Calculate annual generation by asset
    if not dispatch_df.empty:
        print("Annual Generation by Asset:")
        annual_generation = {}
        
        for unit in dispatch_df['unit'].unique():
            unit_dispatch = dispatch_df[dispatch_df['unit'] == unit]
            
            # Get the weights for each time
            if 'time' in unit_dispatch.columns:
                unit_dispatch = unit_dispatch.copy()
                
                # Assign weights to each row
                for block_name, block_info in BLOCK_INFO.items():
                    block_start = block_info['start_hour']
                    block_end = block_start + 167
                    weight = block_info['weight']
                    
                    # Create a datetime range for this block
                    block_times = [datetime(2023, 1, 1) + timedelta(hours=h) for h in range(block_start, block_end + 1)]
                    
                    # Assign weights to rows that fall within this block's time range
                    unit_dispatch.loc[unit_dispatch['time'].isin(block_times), 'weight'] = weight
            
            # Calculate the annual generation for this asset
            if 'weight' in unit_dispatch.columns:
                weighted_gen = unit_dispatch['p'] * unit_dispatch['weight']
                annual_mwh = weighted_gen.sum()
            else:
                # If weights are not present, equal weighting
                annual_mwh = unit_dispatch['p'].sum() * (8760 / len(unit_dispatch))
            
            annual_generation[unit] = annual_mwh
            print(f"Asset {unit}: {annual_mwh:.2f} MWh")
        
        results['annual_generation'] = annual_generation
    
    # Calculate annual flows by branch
    if not flow_df.empty:
        print("\nAnnual Flows by Branch:")
        annual_flows = {}
        
        for branch_id in flow_df['branch'].unique():
            branch_flows = flow_df[flow_df['branch'] == branch_id]
            
            # Get the weights for each time
            if 'time' in branch_flows.columns:
                branch_flows = branch_flows.copy()
                
                # Assign weights to each row
                for block_name, block_info in BLOCK_INFO.items():
                    block_start = block_info['start_hour']
                    block_end = block_start + 167
                    weight = block_info['weight']
                    
                    # Create a datetime range for this block
                    block_times = [datetime(2023, 1, 1) + timedelta(hours=h) for h in range(block_start, block_end + 1)]
                    
                    # Assign weights to rows that fall within this block's time range
                    branch_flows.loc[branch_flows['time'].isin(block_times), 'weight'] = weight
            
            # Calculate the annual flow for this branch
            if 'weight' in branch_flows.columns:
                weighted_flow = branch_flows['flow'] * branch_flows['weight']
                annual_mwh = weighted_flow.sum()
            else:
                # If weights are not present, equal weighting
                annual_mwh = branch_flows['flow'].sum() * (8760 / len(branch_flows))
            
            annual_flows[branch_id] = annual_mwh
            print(f"Branch {branch_id}: {annual_mwh:.2f} MWh")
        
        results['annual_flows'] = annual_flows
    
    # Include investment decisions if in investment mode
    if investment_mode and 'investment_decisions' in dcopf_results:
        results['investment_decisions'] = dcopf_results['investment_decisions']
        results['investment_cost'] = dcopf_results.get('investment_cost', 0)
        
        print("\nInvestment Decisions:")
        for unit_id, decision in results['investment_decisions'].items():
            status = "Selected" if decision == 1 else "Not selected"
            print(f"Asset {unit_id}: {status}")
    
    return results

def post_process_results(results, block_weights):
    """
    Post-process DCOPF results to account for block weighting.
    
    Args:
        results: DCOPF results
        block_weights: Dictionary mapping timestamps to block weights
        
    Returns:
        Post-processed results
    """
    # Calculate weighted metrics
    if 'cost' in results:
        # The cost is already weighted in the objective function
        results['annual_cost'] = results['cost']
        print(f"Annual cost (already weighted): ${results['annual_cost']:,.2f}")
    
    if 'generation' in results:
        # Add weights to generation results
        gen_df = results['generation']
        
        # Add weights column based on time
        gen_df['weight'] = gen_df['time'].map(lambda t: block_weights.get(t, 1.0))
        
        # Calculate weighted generation
        gen_df['weighted_gen'] = gen_df['gen'] * gen_df['weight']
        
        # Calculate annual totals by generator
        annual_gen = gen_df.groupby('id')['weighted_gen'].sum().reset_index()
        annual_gen.rename(columns={'weighted_gen': 'annual_mwh'}, inplace=True)
        
        # Add column to indicate if this is annual generation
        annual_gen['is_annual'] = True
        
        # Add to results
        results['annual_generation'] = annual_gen
        
        # Print summary
        print("\nAnnual Generation by Asset:")
        for _, row in annual_gen.iterrows():
            print(f"Asset {row['id']}: {row['annual_mwh']:,.2f} MWh")
    
    if 'flows' in results:
        # Add weights to flow results
        flows_df = results['flows']
        
        # Add weights column based on time
        flows_df['weight'] = flows_df['time'].map(lambda t: block_weights.get(t, 1.0))
        
        # Calculate weighted flows
        flows_df['weighted_flow'] = flows_df['flow'] * flows_df['weight']
        
        # Calculate annual totals by branch
        annual_flows = flows_df.groupby(['from_bus', 'to_bus'])['weighted_flow'].sum().reset_index()
        annual_flows.rename(columns={'weighted_flow': 'annual_mwh'}, inplace=True)
        
        # Add to results
        results['annual_flows'] = annual_flows
        
        # Print summary
        print("\nAnnual Flows by Branch:")
        for _, row in annual_flows.iterrows():
            print(f"Branch {row['from_bus']}-{row['to_bus']}: {row['annual_mwh']:,.2f} MWh")
    
    # Add block information to results for visualization
    # Create a mapping from time to block for easier access in plotting
    time_to_block = {}
    base_date = datetime(2023, 1, 1)
    for block_name, block_info in BLOCK_INFO.items():
        block_start = block_info['start_hour']
        block_end = block_start + 167
        for hour in range(block_start, block_end + 1):
            synthetic_time = base_date + timedelta(hours=hour)
            time_to_block[synthetic_time] = block_name
    
    # Add block information to the results
    if 'generation' in results:
        results['generation']['block'] = results['generation']['time'].map(time_to_block)
    
    if 'flows' in results:
        results['flows']['block'] = results['flows']['time'].map(time_to_block)
    
    if 'marginal_prices' in results:
        results['marginal_prices']['block'] = results['marginal_prices']['time'].map(time_to_block)
    
    if 'storage' in results:
        results['storage']['block'] = results['storage']['time'].map(time_to_block)
    
    # Store the block information for reference
    results['block_info'] = BLOCK_INFO
    results['time_to_block'] = time_to_block
    
    return results

def plot_representative_weeks_results(results, results_dir):
    """
    Plot the results of the representative weeks DCOPF run.
    
    Args:
        results: Results dictionary from run_representative_weeks_dcopf
        results_dir: Directory to save plots
    """
    print("Plotting representative weeks results...")
    
    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the dispatch dataframe
    dispatch_df = results.get('dispatch_df', None)
    
    if dispatch_df is None or dispatch_df.empty:
        print("No dispatch data available for plotting")
        return
    
    # Make sure time is the index
    if 'time' in dispatch_df.columns:
        dispatch_df = dispatch_df.set_index('time')
    
    # Sort dispatch by unit ID
    unit_ids = sorted(dispatch_df['unit'].unique())
    
    # Identify storage units (those that have both positive and negative values)
    storage_units = []
    for unit in unit_ids:
        unit_data = dispatch_df[dispatch_df['unit'] == unit]['p']
        if (unit_data > 0).any() and (unit_data < 0).any():
            storage_units.append(unit)
    
    # Create a pivot table for generation
    dispatch_df_non_storage = dispatch_df[~dispatch_df['unit'].isin(storage_units)]
    if not dispatch_df_non_storage.empty:
        pivot_df = dispatch_df_non_storage.pivot(columns='unit', values='p')
        
        # Plot generation for non-storage units
        plt.figure(figsize=(14, 8))
        pivot_df.plot(kind='area', stacked=True, ax=plt.gca())
        plt.title('Generation Dispatch by Unit (Non-Storage)')
        plt.xlabel('Time')
        plt.ylabel('Power (MW)')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'generation_dispatch.png'), dpi=300, bbox_inches='tight')
    
    # Plot storage units separately
    if storage_units:
        for storage_unit in storage_units:
            storage_data = dispatch_df[dispatch_df['unit'] == storage_unit]
            
            plt.figure(figsize=(14, 8))
            # Separate positive (discharge) and negative (charge) values
            discharge = storage_data[storage_data['p'] > 0]['p']
            charge = storage_data[storage_data['p'] < 0]['p']
            
            # Plot discharge as positive area
            if not discharge.empty:
                plt.fill_between(discharge.index, 0, discharge.values, alpha=0.7, 
                                  label=f'Unit {storage_unit} - Discharge', color='green')
            
            # Plot charge as negative area
            if not charge.empty:
                plt.fill_between(charge.index, 0, charge.values, alpha=0.7, 
                                 label=f'Unit {storage_unit} - Charge', color='red')
            
            plt.title(f'Storage Unit {storage_unit} Charge/Discharge Profile')
            plt.xlabel('Time')
            plt.ylabel('Power (MW)')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(results_dir, f'storage_unit_{storage_unit}_profile.png'), 
                        dpi=300, bbox_inches='tight')
    
    # Plot total system balance
    plt.figure(figsize=(14, 8))
    dispatch_df_agg = dispatch_df.groupby('time')['p'].sum()
    plt.plot(dispatch_df_agg.index, dispatch_df_agg.values)
    plt.title('Total System Generation')
    plt.xlabel('Time')
    plt.ylabel('Power (MW)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'total_system_generation.png'), dpi=300, bbox_inches='tight')
    
    # Plot flows if available
    if 'flow_df' in results and not results['flow_df'].empty:
        flow_df = results['flow_df']
        if 'time' in flow_df.columns:
            flow_df = flow_df.set_index('time')
        
        plt.figure(figsize=(14, 8))
        pivot_flow = flow_df.pivot(columns='branch', values='flow')
        pivot_flow.plot(ax=plt.gca())
        plt.title('Branch Flows')
        plt.xlabel('Time')
        plt.ylabel('Flow (MW)')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'branch_flows.png'), dpi=300, bbox_inches='tight')
        
    # Plot storage state of charge if available
    if 'storage_soc_df' in results and not results['storage_soc_df'].empty:
        soc_df = results['storage_soc_df']
        if 'time' in soc_df.columns:
            soc_df = soc_df.set_index('time')
        
        plt.figure(figsize=(14, 8))
        pivot_soc = soc_df.pivot(columns='unit', values='soc')
        pivot_soc.plot(ax=plt.gca())
        plt.title('Storage State of Charge')
        plt.xlabel('Time')
        plt.ylabel('Energy (MWh)')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'storage_soc.png'), dpi=300, bbox_inches='tight')
    
    print(f"Plots saved to {results_dir}")
    
    plt.close('all')  # Close all figures to free memory

def run_dcopf(gen_data, branch, bus, demand, investment_mode=False, investment_candidates=None):
    """
    Run a DCOPF model using the provided data.
    
    Args:
        gen_data: Generator data
        branch: Branch data
        bus: Bus data
        demand: Demand data
        investment_mode: Whether to run in investment mode
        investment_candidates: DataFrame containing investment candidates
        
    Returns:
        Dictionary with DCOPF results
    """
    try:
        # Print basic information about the input data
        print(f"Running DCOPF with {len(gen_data)} generator records, {len(demand)} demand records")
        print(f"Generator time range: {gen_data['time'].min()} to {gen_data['time'].max()}")
        print(f"Demand time range: {demand['time'].min()} to {demand['time'].max()}")
        
        if investment_mode and investment_candidates is not None:
            print(f"Investment mode with {len(investment_candidates)} candidates")
            
            # Make a copy of the generator data to avoid modifying the original
            gen_data = gen_data.copy()
            
            # Reset investment_required flags to match our investment_candidates
            if 'investment_required' in gen_data.columns:
                # First, set all assets to not require investment
                gen_data['investment_required'] = 0
                
                # Then, set investment_required=1 only for assets in the investment_candidates DataFrame
                candidate_ids = investment_candidates['id'].unique()
                print(f"Setting investment_required=1 for assets: {candidate_ids}")
                gen_data.loc[gen_data['id'].isin(candidate_ids), 'investment_required'] = 1
            else:
                # If the column doesn't exist, create it
                gen_data['investment_required'] = 0
                candidate_ids = investment_candidates['id'].unique()
                gen_data.loc[gen_data['id'].isin(candidate_ids), 'investment_required'] = 1
                
            # Now, prepare asset_lifetimes and asset_capex dictionaries
            # These should include ALL assets that have investment_required=1
            asset_lifetimes = {}
            asset_capex = {}
            
            # First, check the gen_data for built-in lifetime and capex
            for asset_id in candidate_ids:
                # Get data for this asset
                asset_data = gen_data[gen_data['id'] == asset_id].iloc[0]
                
                # Extract lifetime and capex if they exist
                if 'lifetime' in asset_data and asset_data['lifetime'] > 0:
                    asset_lifetimes[asset_id] = asset_data['lifetime']
                else:
                    # Use a default lifetime of 20 years if not specified
                    asset_lifetimes[asset_id] = 20
                    
                if 'capex' in asset_data and asset_data['capex'] > 0:
                    asset_capex[asset_id] = asset_data['capex']
            
            # Then override with any values from investment_candidates
            for _, row in investment_candidates.iterrows():
                asset_id = row['id']
                # Use the annualized_cost from the investment_candidates DataFrame
                if 'annualized_cost' in row:
                    asset_capex[asset_id] = row['annualized_cost']
            
            # Ensure all candidates have a capex value
            for asset_id in candidate_ids:
                if asset_id not in asset_capex:
                    print(f"WARNING: No capex specified for asset {asset_id}. Using default value of 100,000.")
                    asset_capex[asset_id] = 100000  # Default value
            
            print(f"Asset IDs requiring investment: {candidate_ids}")
            print(f"Asset lifetimes: {asset_lifetimes}")
            print(f"Asset capex: {asset_capex}")
            
            # Call the DCOPF with investment
            dcopf_results = dcopf(
                gen_time_series=gen_data,
                branch=branch,
                bus=bus,
                demand_time_series=demand,
                include_investment=True,
                planning_horizon=1,
                asset_lifetimes=asset_lifetimes,
                asset_capex=asset_capex
            )
        else:
            print("Operations mode")
            # Call the operations DCOPF function
            dcopf_results = dcopf(
                gen_time_series=gen_data,
                branch=branch,
                bus=bus,
                demand_time_series=demand,
                include_investment=False
            )
        
        # If DCOPF returned None, return empty results
        if dcopf_results is None:
            return {
                'dispatch_df': pd.DataFrame(),
                'flow_df': pd.DataFrame(),
                'storage_soc_df': pd.DataFrame(),
                'objective_value': 0
            }
            
        # Map the DCOPF result keys to our expected format
        result = {}
        
        # Generation/dispatch
        if 'generation' in dcopf_results:
            # Rename from DCOPF's 'generation' dataframe to our 'dispatch_df'
            gen_df = dcopf_results['generation'].copy()
            
            # Rename columns to match expected format
            if 'gen' in gen_df.columns:
                gen_df = gen_df.rename(columns={'gen': 'p', 'id': 'unit', 'node': 'bus'})
            
            result['dispatch_df'] = gen_df
        else:
            result['dispatch_df'] = pd.DataFrame()
            
        # Flows
        if 'flows' in dcopf_results:
            # Copy flow dataframe
            flow_df = dcopf_results['flows'].copy()
            
            # Create a branch identifier column
            flow_df['branch'] = flow_df['from_bus'].astype(str) + '-' + flow_df['to_bus'].astype(str)
            
            result['flow_df'] = flow_df
        else:
            result['flow_df'] = pd.DataFrame()
            
        # Storage state of charge
        if 'storage' in dcopf_results:
            # Copy storage dataframe
            storage_df = dcopf_results['storage'].copy()
            
            # Rename columns to match expected format
            storage_df = storage_df.rename(columns={'storage_id': 'unit', 'E': 'soc'})
            
            result['storage_soc_df'] = storage_df
        else:
            result['storage_soc_df'] = pd.DataFrame()
            
        # Objective value
        if 'cost' in dcopf_results:
            result['objective_value'] = dcopf_results['cost']
        else:
            result['objective_value'] = 0
            
        # Investment decisions
        if 'investment_decisions' in dcopf_results:
            result['investment_decisions'] = dcopf_results['investment_decisions']
            
        if 'investment_cost' in dcopf_results:
            result['investment_cost'] = dcopf_results['investment_cost']
            
        return result
        
    except Exception as e:
        print(f"Error in run_dcopf: {str(e)}")
        print(traceback.format_exc())
        # Return a minimal results dictionary
        return {
            'dispatch_df': pd.DataFrame(),
            'flow_df': pd.DataFrame(),
            'storage_soc_df': pd.DataFrame(),
            'objective_value': 0
        }

if __name__ == "__main__":
    print("Representative weeks module loaded. Use run_representative_weeks_dcopf() to run the model.") 