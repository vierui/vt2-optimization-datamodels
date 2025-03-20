#!/usr/bin/env python3

"""
dcopf_investment.py

A multi-stage investment model for power systems using a chunk-based approach.
This model integrates with DCOPF to optimize both investment decisions and operational dispatch.

Key features:
- Multi-stage investment decisions based on asset lifetimes
- Chunk-based approach to reduce problem size
- Integration with DCOPF for operational feasibility
- Support for different asset lifetimes and capital costs
- Flexible planning horizon
"""

import sys
import os
import pandas as pd
import numpy as np
import math
from pandas.tseries.offsets import DateOffset
from datetime import datetime

# Add CPLEX Python API to the Python path
cplex_python_path = "/Applications/CPLEX_Studio2211/python"
if os.path.exists(cplex_python_path):
    sys.path.append(cplex_python_path)

# Import CPLEX - if not available, raise an error
try:
    import cplex
    from cplex.exceptions import CplexError
except ImportError:
    raise ImportError("CPLEX Python API not found. Please ensure CPLEX is properly installed "
                     f"and the Python API is available at {cplex_python_path}")

# Import local modules
from scripts.investment_utils import (
    calculate_chunks, create_chunk_periods, create_representative_periods,
    create_gen_data_for_investment, scale_opex_by_chunk, calculate_investment_costs
)
from scripts.dcopf import dcopf

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

def prepare_chunk_data(gen_data, planning_horizon, asset_lifetimes=None):
    """
    Prepare data for chunk-based investment modeling.
    
    Args:
        gen_data: DataFrame with generator data
        planning_horizon: Planning horizon in years
        asset_lifetimes: Optional dict mapping asset IDs to lifetimes
        
    Returns:
        Tuple of dictionaries for chunk periods, chunks by asset
    """
    # If asset_lifetimes not provided, extract from gen_data
    if asset_lifetimes is None:
        asset_lifetimes = {}
        for _, row in gen_data.iterrows():
            asset_id = row['id']
            lifetime = row.get('lifetime', 20)  # Default 20 years
            asset_lifetimes[asset_id] = lifetime
    
    # Create chunks
    chunk_periods = create_chunk_periods(asset_lifetimes, planning_horizon)
    
    # Organize chunks by asset
    chunks_by_asset = {}
    for asset_id, chunks in chunk_periods.items():
        chunks_by_asset[asset_id] = list(chunks.keys())
    
    return chunk_periods, chunks_by_asset

def dcopf_investment(gen_time_series, branch, bus, demand_time_series, 
                    planning_horizon=10, start_year=2023, 
                    asset_lifetimes=None, asset_capex=None,
                    operational_periods_per_year=4, hours_per_period=24,
                    delta_t=1):
    """
    Multi-stage investment model for power systems.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        planning_horizon: Planning horizon in years
        start_year: Starting year for the planning horizon
        asset_lifetimes: Optional dict mapping asset IDs to lifetimes
        asset_capex: Optional dict mapping asset IDs to capital costs
        operational_periods_per_year: Number of representative operational periods per year
        hours_per_period: Hours in each operational period
        delta_t: Time step duration in hours
        
    Returns:
        Dictionary with investment results
    """
    print("[INVEST] Entering dcopf_investment function...")
    
    # Prepare generator data with investment-related fields if needed
    gen_data = create_gen_data_for_investment(gen_time_series, planning_horizon, start_year)
    
    # If asset_lifetimes not provided, extract from gen_data
    if asset_lifetimes is None:
        asset_lifetimes = {}
        for _, row in gen_data.iterrows():
            asset_id = row['id']
            if 'time' in gen_data.columns:  # Handle time series data
                # Just get the first row for this asset
                first_row = gen_data[gen_data['id'] == asset_id].iloc[0]
                lifetime = first_row.get('lifetime', 20)  # Default 20 years
            else:
                lifetime = row.get('lifetime', 20)  # Default 20 years
            asset_lifetimes[asset_id] = lifetime
    
    # If asset_capex not provided, extract from gen_data
    if asset_capex is None:
        asset_capex = {}
        for _, row in gen_data.iterrows():
            asset_id = row['id']
            if 'time' in gen_data.columns:  # Handle time series data
                # Just get the first row for this asset
                first_row = gen_data[gen_data['id'] == asset_id].iloc[0]
                capex = first_row.get('capex', 1000000)  # Default $1M/MW
            else:
                capex = row.get('capex', 1000000)  # Default $1M/MW
            asset_capex[asset_id] = capex
    
    # Function to safely convert any value to float
    def safe_float(value):
        return float(value)
    
    try:
        # Create CPLEX problem
        problem = cplex.Cplex()
        
        # Minimize objective function
        problem.objective.set_sense(problem.objective.sense.minimize)
        
        # Set MIP parameters
        problem.parameters.mip.display.set(4)  # Detailed display
        
        # 1. Prepare chunk data
        chunk_periods, chunks_by_asset = prepare_chunk_data(
            gen_data, planning_horizon, asset_lifetimes
        )
        
        # 2. Create representative operational periods
        start_date = datetime(start_year, 1, 1)
        annual_periods = create_typical_periods(
            start_date, operational_periods_per_year, hours_per_period
        )
        
        # 3. Map chunks to operational periods
        representative_periods = create_representative_periods(chunk_periods, annual_periods)
        
        # 4. Define sets
        # Set of all assets
        assets = sorted(set(gen_data['id']))
        
        # Extract storage vs. non-storage units
        storage_data = gen_data[gen_data['emax'] > 0] if 'emax' in gen_data.columns else pd.DataFrame()
        S = storage_data['id'].unique() if not storage_data.empty else []
        non_storage_data = gen_data[gen_data['emax'] == 0] if 'emax' in gen_data.columns else gen_data
        G = non_storage_data['id'].unique()
        
        # 5. Create binary investment variables
        binary_vars = {}
        binary_names = []
        
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                var_name = f"b_{asset_id}_{chunk_idx}"
                binary_vars[(asset_id, chunk_idx)] = var_name
                binary_names.append(var_name)
        
        problem.variables.add(
            lb=[0] * len(binary_names),
            ub=[1] * len(binary_names),
            names=binary_names,
            types=["B"] * len(binary_names)  # B for binary variables
        )
        
        # 6. Create variables for DCOPF for each chunk
        # For each asset and chunk, we need separate DCOPF variables
        chunk_dcopf_vars = {}
        
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                chunk_key = (asset_id, chunk_idx)
                periods = representative_periods[asset_id][chunk_idx]['periods']
                
                # Create DCOPF variables for this chunk
                chunk_dcopf_vars[chunk_key] = {
                    'gen_vars': {},
                    'theta_vars': {},
                    'flow_vars': {}
                }
                
                # a) Generation variables
                for g in G:
                    for t in periods:
                        # Find generator data for this period
                        gen_row = gen_data[(gen_data['id'] == g) & 
                                         (gen_data['time'] == t if 'time' in gen_data.columns else True)]
                        
                        if gen_row.empty:
                            print(f"[INVEST] Missing data for generator={g}, time={t}. Using defaults.")
                            pmin = 0
                            pmax = 100
                            cost = 50
                        else:
                            pmin = gen_row['pmin'].iloc[0]
                            pmax = gen_row['pmax'].iloc[0]
                            cost = gen_row['gencost'].iloc[0]
                        
                        # Variable name with chunk identifier
                        var_name = f"g_{g}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        chunk_dcopf_vars[chunk_key]['gen_vars'][(g, t)] = var_name
                        
                        # Add the variable
                        # If g == asset_id, then generation is limited by binary decision
                        if g == asset_id:
                            problem.variables.add(
                                obj=[safe_float(cost)],
                                lb=[0.0],
                                ub=[safe_float(pmax)],
                                names=[var_name]
                            )
                        else:
                            # Otherwise, normal generation variable
                            problem.variables.add(
                                obj=[safe_float(cost)],
                                lb=[safe_float(pmin)],
                                ub=[safe_float(pmax)],
                                names=[var_name]
                            )
                
                # b) Voltage angle variables
                for i in bus['bus_i'].values:
                    for t in periods:
                        var_name = f"t_{i}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        chunk_dcopf_vars[chunk_key]['theta_vars'][(i, t)] = var_name
                        
                        problem.variables.add(
                            lb=[-cplex.infinity],
                            ub=[cplex.infinity],
                            names=[var_name]
                        )
                
                # c) Flow variables
                for _, row in branch.iterrows():
                    i = int(row['fbus'])
                    j = int(row['tbus'])
                    for t in periods:
                        var_name = f"flow_{i}_{j}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        chunk_dcopf_vars[chunk_key]['flow_vars'][(i, j, t)] = var_name
                        
                        problem.variables.add(
                            lb=[-cplex.infinity],
                            ub=[cplex.infinity],
                            names=[var_name]
                        )
        
        # 7. Add constraints
        print("[INVEST] Adding constraints...")
        
        # For each asset and chunk
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                chunk_key = (asset_id, chunk_idx)
                binary_var = binary_vars[chunk_key]
                periods = representative_periods[asset_id][chunk_idx]['periods']
                
                # a) Investment-related constraints
                # The asset's generation in this chunk is limited by the binary decision
                for t in periods:
                    gen_var = chunk_dcopf_vars[chunk_key]['gen_vars'].get((asset_id, t))
                    
                    if gen_var is not None:
                        # Get the max capacity for this asset
                        gen_row = gen_data[(gen_data['id'] == asset_id) & 
                                         (gen_data['time'] == t if 'time' in gen_data.columns else True)]
                        
                        if gen_row.empty:
                            pmax = 100  # Default
                        else:
                            pmax = gen_row['pmax'].iloc[0]
                        
                        # g_{asset_id,t} ≤ Pmax * b_{asset_id,chunk}
                        constraint_name = f"inv_{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        
                        problem.linear_constraints.add(
                            lin_expr=[[[gen_var, binary_var], [1.0, -safe_float(pmax)]]],
                            senses=["L"],
                            rhs=[0.0],
                            names=[constraint_name]
                        )
                
                # b) DCOPF-related constraints for this chunk
                # Power flow constraints
                for _, row_b in branch.iterrows():
                    i = row_b['fbus']
                    j = row_b['tbus']
                    susceptance = row_b['sus']
                    rate_a = safe_float(row_b['ratea'])
                    
                    for t in periods:
                        # Flow = (angle_from - angle_to) / x
                        from_angle_var = chunk_dcopf_vars[chunk_key]['theta_vars'][(i, t)]
                        to_angle_var = chunk_dcopf_vars[chunk_key]['theta_vars'][(j, t)]
                        flow_var = chunk_dcopf_vars[chunk_key]['flow_vars'][(i, j, t)]
                        
                        constraint_name = f"dcflow_{int(i)}_{int(j)}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        
                        var_list = [flow_var, from_angle_var, to_angle_var]
                        coef_list = [1.0, -safe_float(susceptance), safe_float(susceptance)]
                        
                        problem.linear_constraints.add(
                            lin_expr=[[var_list, coef_list]],
                            senses=["E"],
                            rhs=[0.0],
                            names=[constraint_name]
                        )
                        
                        # Flow limits
                        # Upper limit: FLOW[i, j, t] <= rate_a
                        upper_constraint_name = f"upflow_{int(i)}_{int(j)}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        problem.linear_constraints.add(
                            lin_expr=[[[flow_var], [1.0]]],
                            senses=["L"],
                            rhs=[rate_a],
                            names=[upper_constraint_name]
                        )
                        
                        # Lower limit: FLOW[i, j, t] >= -rate_a
                        lower_constraint_name = f"loflow_{int(i)}_{int(j)}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        problem.linear_constraints.add(
                            lin_expr=[[[flow_var], [1.0]]],
                            senses=["G"],
                            rhs=[-rate_a],
                            names=[lower_constraint_name]
                        )
                
                # Power balance constraints
                for t in periods:
                    for i in bus['bus_i'].values:
                        # Collect all variable names and coefficients
                        var_names = []
                        coefficients = []
                        
                        # Add generation
                        for g in G:
                            gen_rows = gen_data.loc[
                                (gen_data['id'] == g) & 
                                (gen_data['time'] == t if 'time' in gen_data.columns else True)
                            ]
                            
                            if not gen_rows.empty and gen_rows['bus'].iloc[0] == i:
                                gen_var = chunk_dcopf_vars[chunk_key]['gen_vars'].get((g, t))
                                if gen_var is not None:
                                    var_names.append(gen_var)
                                    coefficients.append(1.0)
                        
                        # Add flows into the bus
                        for idx_b, row_b in branch.iterrows():
                            if row_b['tbus'] == i:  # Flow into bus i
                                from_bus = int(row_b['fbus'])
                                to_bus = int(row_b['tbus'])
                                flow_var = chunk_dcopf_vars[chunk_key]['flow_vars'].get((from_bus, to_bus, t))
                                if flow_var is not None:
                                    var_names.append(flow_var)
                                    coefficients.append(1.0)
                        
                        # Add flows out of the bus
                        for idx_b, row_b in branch.iterrows():
                            if row_b['fbus'] == i:  # Flow out of bus i
                                from_bus = int(row_b['fbus'])
                                to_bus = int(row_b['tbus'])
                                flow_var = chunk_dcopf_vars[chunk_key]['flow_vars'].get((from_bus, to_bus, t))
                                if flow_var is not None:
                                    var_names.append(flow_var)
                                    coefficients.append(-1.0)
                        
                        # Get demand at bus i
                        pd_val = 0
                        demands_at_bus = demand_time_series.loc[
                            (demand_time_series['bus'] == i) & 
                            (demand_time_series['time'] == t if 'time' in demand_time_series.columns else True),
                            'pd'
                        ]
                        pd_val = demands_at_bus.sum() if not demands_at_bus.empty else 0
                        
                        constraint_name = f"pb_{int(i)}_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                        
                        # Add power balance constraint
                        if var_names:
                            problem.linear_constraints.add(
                                lin_expr=[[var_names, coefficients]],
                                senses=["E"],
                                rhs=[safe_float(pd_val)],
                                names=[constraint_name]
                            )
                
                # Slack bus constraint
                slack_bus = 1  # Bus 1 is the reference
                for t in periods:
                    theta_slack = chunk_dcopf_vars[chunk_key]['theta_vars'][(slack_bus, t)]
                    constraint_name = f"slack_c{asset_id}_{chunk_idx}_t{t.strftime('%Y%m%d%H')}"
                    
                    problem.linear_constraints.add(
                        lin_expr=[[[theta_slack], [1.0]]],
                        senses=["E"],
                        rhs=[0.0],
                        names=[constraint_name]
                    )
        
        # 8. Add investment costs to objective
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                binary_var = binary_vars[(asset_id, chunk_idx)]
                
                # Calculate the annual cost for this asset
                annual_cost = safe_float(asset_capex[asset_id]) / safe_float(asset_lifetimes[asset_id])
                
                # Scale by the chunk duration
                chunk_duration = representative_periods[asset_id][chunk_idx]['scaling_factor']
                chunk_cost = annual_cost * chunk_duration
                
                # Add to objective
                problem.objective.set_linear(binary_var, chunk_cost)
        
        # 9. Solve the problem
        print("[INVEST] About to solve the problem with CPLEX...")
        problem.solve()
        
        # Check solution status
        status = problem.solution.get_status()
        status_string = problem.solution.get_status_string()
        print(f"[INVEST] Solver returned status code = {status}, interpreted as '{status_string}'")
        
        if status != problem.solution.status.optimal and status != problem.solution.status.MIP_optimal:
            print(f"[INVEST] Not optimal => returning None.")
            return None
        
        # 10. Extract results
        # a) Extract objective value
        objective_value = problem.solution.get_objective_value()
        print(f"[INVEST] Final cost = {objective_value}, status = {status_string}")
        
        # b) Extract investment decisions
        investment_decisions = []
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                binary_var = binary_vars[(asset_id, chunk_idx)]
                val = problem.solution.get_values(binary_var)
                
                # Get chunk period info
                chunk_period = chunk_periods[asset_id][chunk_idx]
                start_year = chunk_period[0] + start_year
                end_year = chunk_period[1] + start_year
                
                investment_decisions.append({
                    'asset_id': asset_id,
                    'chunk_idx': chunk_idx,
                    'start_year': start_year,
                    'end_year': end_year,
                    'decision': 1 if val > 0.5 else 0
                })
        
        investment_df = pd.DataFrame(investment_decisions)
        
        # c) Extract generation for each chunk
        generation_by_chunk = {}
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                chunk_key = (asset_id, chunk_idx)
                periods = representative_periods[asset_id][chunk_idx]['periods']
                
                generation = []
                for g in G:
                    for t in periods:
                        gen_var = chunk_dcopf_vars[chunk_key]['gen_vars'].get((g, t))
                        if gen_var is not None:
                            val = problem.solution.get_values(gen_var)
                            
                            # Find the bus for this generator
                            gen_row = gen_data[(gen_data['id'] == g) & 
                                             (gen_data['time'] == t if 'time' in gen_data.columns else True)]
                            g_bus = gen_row['bus'].iloc[0] if not gen_row.empty else None
                            
                            generation.append({
                                'time': t,
                                'id': g,
                                'node': g_bus,
                                'gen': 0 if math.isnan(val) else val,
                                'asset_id': asset_id,
                                'chunk_idx': chunk_idx
                            })
                
                generation_by_chunk[chunk_key] = pd.DataFrame(generation)
        
        # d) Extract flows for each chunk
        flows_by_chunk = {}
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                chunk_key = (asset_id, chunk_idx)
                periods = representative_periods[asset_id][chunk_idx]['periods']
                
                flows = []
                for _, row_b in branch.iterrows():
                    i = int(row_b['fbus'])
                    j = int(row_b['tbus'])
                    for t in periods:
                        flow_var = chunk_dcopf_vars[chunk_key]['flow_vars'].get((i, j, t))
                        if flow_var is not None:
                            val = problem.solution.get_values(flow_var)
                            flows.append({
                                'time': t,
                                'from_bus': i,
                                'to_bus': j,
                                'flow': 0 if math.isnan(val) else val,
                                'asset_id': asset_id,
                                'chunk_idx': chunk_idx
                            })
                
                flows_by_chunk[chunk_key] = pd.DataFrame(flows)
        
        # e) Calculate investment costs by chunk
        investment_costs = {}
        for asset_id in assets:
            for chunk_idx in chunks_by_asset[asset_id]:
                binary_var = binary_vars[(asset_id, chunk_idx)]
                val = problem.solution.get_values(binary_var)
                
                if val > 0.5:  # Investment decision = yes
                    # Calculate the annual cost
                    annual_cost = safe_float(asset_capex[asset_id]) / safe_float(asset_lifetimes[asset_id])
                    
                    # Scale by the chunk duration
                    chunk_duration = representative_periods[asset_id][chunk_idx]['scaling_factor']
                    chunk_cost = annual_cost * chunk_duration
                    
                    investment_costs[(asset_id, chunk_idx)] = chunk_cost
        
        # f) Summarize costs
        total_investment_cost = sum(investment_costs.values())
        total_operational_cost = objective_value - total_investment_cost
        
        cost_summary = {
            'total_cost': objective_value,
            'investment_cost': total_investment_cost,
            'operational_cost': total_operational_cost
        }
        
        print("[INVEST] Done, returning result dictionary.")
        
        return {
            'investment': investment_df,
            'generation_by_chunk': generation_by_chunk,
            'flows_by_chunk': flows_by_chunk,
            'investment_costs': investment_costs,
            'cost_summary': cost_summary,
            'status': status_string,
            'chunk_periods': chunk_periods,
            'representative_periods': representative_periods
        }
    except Exception as e:
        print(f"[INVEST] Error in investment model: {e}")
        import traceback
        traceback.print_exc()
        return None 