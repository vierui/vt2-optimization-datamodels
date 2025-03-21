#!/usr/bin/env python3

"""
optimizer.py

Mathematical optimization model for power system planning with investment decisions.
This implementation models:

1. Decision Variables:
   - Binary investment variables for asset installation in each lifetime period
   - Continuous generation variables for operational dispatch
   - Power flow variables for network modeling
   - Bus angle variables for DC power flow equations
   - Storage charge/discharge variables and energy state variables

2. Objective Function:
   - Minimize the sum of capital costs (investment) and operational costs (generation)

3. Constraints:
   - Power balance at each bus
   - DC power flow physics
   - Generation capacity limited by investment decisions
   - Network flow limits
   - Storage dynamics constraints
"""

import sys
import os
import pandas as pd
import numpy as np
import math
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

# Import utility functions
from scripts.investment_utils import (
    safe_float, 
    ensure_float_list,
    calculate_lifetime_periods,
    create_lifetime_periods_mapping,
    create_representative_periods,
    create_typical_periods,
    create_gen_data_for_investment
)

def run_investment_model(
    gen_time_series, branch, bus, demand_time_series, 
    planning_horizon=10, start_year=2023, 
    asset_lifetimes=None, asset_capex=None,
    operational_periods_per_year=4, hours_per_period=24,
    delta_t=1
):
    """
    Mathematical optimization model for power system planning with investment decisions.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        planning_horizon: Planning horizon in years
        start_year: Starting year for the planning horizon
        asset_lifetimes: Dict mapping asset IDs to lifetimes
        asset_capex: Dict mapping asset IDs to capital costs
        operational_periods_per_year: Number of representative operational periods per year
        hours_per_period: Hours in each operational period
        delta_t: Time step duration in hours
        
    Returns:
        Dictionary with optimization results
    """
    print("[OPTIMIZE] Entering optimization model...")
    
    # Ensure planning_horizon is an integer
    planning_horizon = int(planning_horizon)
    
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
            asset_lifetimes[asset_id] = int(lifetime)  # Ensure it's an integer
    
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
            asset_capex[asset_id] = safe_float(capex)  # Ensure it's a float
    
    try:
        # Create CPLEX problem
        problem = cplex.Cplex()
        
        # Minimize objective function
        problem.objective.set_sense(problem.objective.sense.minimize)
        
        # Set MIP parameters
        problem.parameters.mip.display.set(4)  # Detailed display
        
        # 1. Prepare lifetime period data
        lifetime_periods = create_lifetime_periods_mapping(asset_lifetimes, planning_horizon)
        
        # 2. Create representative operational periods
        start_date = datetime(start_year, 1, 1)
        operational_periods = create_typical_periods(
            start_date, 
            num_periods=operational_periods_per_year, 
            hours_per_period=hours_per_period
        )
        representative_periods = create_representative_periods(lifetime_periods, operational_periods)
        
        # 3. Create variables for operational decisions
        # Storage for variable indices
        var_gen = {}  # Generation variables
        var_invest = {}  # Investment decision variables
        var_flow = {}  # Power flow variables
        var_angle = {}  # Bus angle variables
        var_pch = {}   # Storage charge variables
        var_pdis = {}  # Storage discharge variables
        var_e = {}     # Storage energy state variables
        
        # Identify storage vs. non-storage units
        S = gen_data[gen_data['emax'] > 0]['id'].unique()  # set of storage IDs
        
        # Investment variables - binary decision variables for each asset and lifetime period
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                # Binary variable name: inv_assetID_periodIdx
                var_name = f"inv_{asset_id}_{period_idx}"
                
                # Add binary variable for investment decision
                problem.variables.add(
                    obj=[safe_float(asset_capex[asset_id])],  # Use the capex as the objective coefficient
                    lb=[0],  # Lower bound
                    ub=[1],  # Upper bound
                    types=[problem.variables.type.binary],  # Binary variable
                    names=[var_name]
                )
                
                # Store the variable index
                var_invest[(asset_id, period_idx)] = var_name
                
                # For each representative period within this lifetime period
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    # Get scaling factor for this lifetime period
                    scaling_factor = safe_float(representative_periods[asset_id][period_idx]['scaling_factor'])
                    
                    # Add generation variables for each generator for each time period
                    for _, gen_row in gen_data[gen_data['time'] == t].iterrows():
                        gen_id = int(gen_row['id'])
                        
                        # Skip storage units here - they'll be handled separately
                        if gen_id in S:
                            continue
                            
                        gen_name = f"g_{gen_id}_{asset_id}_{period_idx}_{t_idx}"
                        
                        # Generation costs should be scaled by the time period's weight
                        gen_cost = safe_float(gen_row['gencost']) * scaling_factor
                        gen_pmin = safe_float(gen_row['pmin'])
                        gen_pmax = safe_float(gen_row['pmax'])
                        
                        # Add generation variable
                        problem.variables.add(
                            obj=[gen_cost],  # Generation cost in objective
                            lb=[0],  # Lower bound
                            ub=[gen_pmax],  # Upper bound
                            types=[problem.variables.type.continuous],
                            names=[gen_name]
                        )
                        
                        # Store the variable index
                        var_gen[(gen_id, asset_id, period_idx, t_idx)] = gen_name
                    
                    # Add storage variables (charge, discharge, energy state)
                    for s in S:
                        # Get storage parameters
                        stor_row = gen_data[(gen_data['id'] == s) & (gen_data['time'] == t)]
                        if stor_row.empty:
                            continue
                            
                        stor_row = stor_row.iloc[0]
                        P_max = safe_float(stor_row['pmax'])
                        E_max = safe_float(stor_row['emax'])
                        
                        # Charge variable
                        pch_name = f"pc_{s}_{asset_id}_{period_idx}_{t_idx}"
                        problem.variables.add(
                            obj=[0.001 * scaling_factor],  # Small cost to prevent unnecessary cycling
                            lb=[0],
                            ub=[P_max],
                            types=[problem.variables.type.continuous],
                            names=[pch_name]
                        )
                        var_pch[(s, asset_id, period_idx, t_idx)] = pch_name
                        
                        # Discharge variable
                        pdis_name = f"pd_{s}_{asset_id}_{period_idx}_{t_idx}"
                        problem.variables.add(
                            obj=[0.001 * scaling_factor],  # Small cost to prevent unnecessary cycling
                            lb=[0],
                            ub=[P_max],
                            types=[problem.variables.type.continuous],
                            names=[pdis_name]
                        )
                        var_pdis[(s, asset_id, period_idx, t_idx)] = pdis_name
                        
                        # Energy state variable
                        e_name = f"e_{s}_{asset_id}_{period_idx}_{t_idx}"
                        problem.variables.add(
                            obj=[0],  # No cost in objective
                            lb=[0],
                            ub=[E_max],
                            types=[problem.variables.type.continuous],
                            names=[e_name]
                        )
                        var_e[(s, asset_id, period_idx, t_idx)] = e_name
                    
                    # Add power flow variables for each branch and time period
                    for _, branch_row in branch.iterrows():
                        from_bus = int(branch_row['fbus'])
                        to_bus = int(branch_row['tbus'])
                        flow_name = f"f_{from_bus}_{to_bus}_{asset_id}_{period_idx}_{t_idx}"
                        
                        # Line limit
                        line_limit = safe_float(branch_row['ratea'])
                        
                        # Add flow variable
                        problem.variables.add(
                            obj=[0],  # No cost in objective
                            lb=[-line_limit],  # Lower bound - can be negative
                            ub=[line_limit],  # Upper bound
                            types=[problem.variables.type.continuous],
                            names=[flow_name]
                        )
                        
                        # Store the variable index
                        var_flow[(from_bus, to_bus, asset_id, period_idx, t_idx)] = flow_name
                    
                    # Add bus angle variables for each bus and time period
                    for _, bus_row in bus.iterrows():
                        bus_i = int(bus_row['bus_i'])
                        angle_name = f"theta_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                        
                        # Add angle variable
                        problem.variables.add(
                            obj=[0],  # No cost in objective
                            lb=[-math.pi] if bus_i != 1 else [0],  # Fix angle of reference bus to 0
                            ub=[math.pi] if bus_i != 1 else [0],  # Fix angle of reference bus to 0
                            types=[problem.variables.type.continuous],
                            names=[angle_name]
                        )
                        
                        # Store the variable index
                        var_angle[(bus_i, asset_id, period_idx, t_idx)] = angle_name
        
        # 4. Add constraints
        
        # Power balance constraints: generation - demand = net flow at each bus
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    for _, bus_row in bus.iterrows():
                        bus_i = int(bus_row['bus_i'])
                        
                        # Get demand at this bus for this time period
                        bus_demand = safe_float(demand_time_series[(demand_time_series['time'] == t) & 
                                                      (demand_time_series['bus'] == bus_i)]['pd'].sum())
                        
                        # Debug output for troubleshooting
                        print(f"[OPTIMIZE] Bus {bus_i}, time {t}, demand: {bus_demand}")
                        
                        # Get all generators at this bus
                        bus_gens = gen_data[(gen_data['time'] == t) & (gen_data['bus'] == bus_i)]
                        
                        # Get all lines connected to this bus
                        outgoing_lines = branch[branch['fbus'] == bus_i]
                        incoming_lines = branch[branch['tbus'] == bus_i]
                        
                        # Constraint name
                        con_name = f"balance_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                        
                        # Variables and coefficients for constraint
                        constraint_vars = []
                        constraint_coefs = []
                        
                        # Add generator variables at this bus (non-storage)
                        for _, gen_row in bus_gens.iterrows():
                            gen_id = int(gen_row['id'])
                            if gen_id not in S and (gen_id, asset_id, period_idx, t_idx) in var_gen:
                                constraint_vars.append(var_gen[(gen_id, asset_id, period_idx, t_idx)])
                                constraint_coefs.append(1.0)  # Positive contribution
                        
                        # Add storage discharge (positive) and charge (negative) variables
                        for s in S:
                            s_row = bus_gens[bus_gens['id'] == s]
                            if not s_row.empty and (s, asset_id, period_idx, t_idx) in var_pdis and (s, asset_id, period_idx, t_idx) in var_pch:
                                # Discharge (positive contribution)
                                constraint_vars.append(var_pdis[(s, asset_id, period_idx, t_idx)])
                                constraint_coefs.append(1.0)
                                
                                # Charge (negative contribution) 
                                constraint_vars.append(var_pch[(s, asset_id, period_idx, t_idx)])
                                constraint_coefs.append(-1.0)
                        
                        # Add outgoing line flow variables (negative contribution)
                        for _, line in outgoing_lines.iterrows():
                            to_bus = int(line['tbus'])
                            if (bus_i, to_bus, asset_id, period_idx, t_idx) in var_flow:
                                constraint_vars.append(var_flow[(bus_i, to_bus, asset_id, period_idx, t_idx)])
                                constraint_coefs.append(-1.0)  # Negative contribution
                        
                        # Add incoming line flow variables (positive contribution)
                        for _, line in incoming_lines.iterrows():
                            from_bus = int(line['fbus'])
                            if (from_bus, bus_i, asset_id, period_idx, t_idx) in var_flow:
                                constraint_vars.append(var_flow[(from_bus, bus_i, asset_id, period_idx, t_idx)])
                                constraint_coefs.append(1.0)  # Positive contribution
                        
                        # Add slack variables to handle potential infeasibilities
                        # Positive slack for unmet demand
                        slack_pos_name = f"slack_pos_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                        problem.variables.add(
                            obj=[10000.0 * scaling_factor],  # High penalty cost
                            lb=[0],
                            ub=[cplex.infinity],
                            types=[problem.variables.type.continuous],
                            names=[slack_pos_name]
                        )
                        constraint_vars.append(slack_pos_name)
                        constraint_coefs.append(1.0)
                        
                        # Negative slack for excess generation
                        slack_neg_name = f"slack_neg_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                        problem.variables.add(
                            obj=[10000.0 * scaling_factor],  # High penalty cost
                            lb=[0],
                            ub=[cplex.infinity],
                            types=[problem.variables.type.continuous],
                            names=[slack_neg_name]
                        )
                        constraint_vars.append(slack_neg_name)
                        constraint_coefs.append(-1.0)
                        
                        # Add power balance constraint with slack variables
                        if len(constraint_vars) > 0:
                            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(constraint_vars, constraint_coefs)],
                                senses=["E"],  # Equality constraint
                                rhs=[safe_float(bus_demand)],  # Equal to demand
                                names=[con_name]
                            )
        
        # Power flow constraints: flow proportional to angle difference
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    for _, branch_row in branch.iterrows():
                        from_bus = int(branch_row['fbus'])
                        to_bus = int(branch_row['tbus'])
                        
                        # Get branch reactance and susceptance
                        x = safe_float(branch_row['x'])
                        susceptance = 1.0 / x
                        
                        # Constraint name
                        con_name = f"flow_{from_bus}_{to_bus}_{asset_id}_{period_idx}_{t_idx}"
                        
                        # Variables and coefficients for constraint
                        constraint_vars = []
                        constraint_coefs = []
                        
                        # Flow variable
                        if (from_bus, to_bus, asset_id, period_idx, t_idx) in var_flow:
                            constraint_vars.append(var_flow[(from_bus, to_bus, asset_id, period_idx, t_idx)])
                            constraint_coefs.append(1.0)
                        
                        # From bus angle
                        if (from_bus, asset_id, period_idx, t_idx) in var_angle:
                            constraint_vars.append(var_angle[(from_bus, asset_id, period_idx, t_idx)])
                            constraint_coefs.append(-susceptance)
                        
                        # To bus angle
                        if (to_bus, asset_id, period_idx, t_idx) in var_angle:
                            constraint_vars.append(var_angle[(to_bus, asset_id, period_idx, t_idx)])
                            constraint_coefs.append(susceptance)
                        
                        # Add flow constraint
                        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(constraint_vars, constraint_coefs)],
                            senses=["E"],  # Equality constraint
                            rhs=[0.0],  # Right-hand side
                            names=[con_name]
                        )
        
        # Generation capacity constraints based on investment decisions
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    for _, gen_row in gen_data[gen_data['time'] == t].iterrows():
                        gen_id = int(gen_row['id'])
                        gen_pmax = safe_float(gen_row['pmax'])
                        
                        # Skip if not in var_gen (storage units)
                        if gen_id in S:
                            continue
                            
                        if (gen_id, asset_id, period_idx, t_idx) not in var_gen:
                            continue
                        
                        # Constraint name
                        con_name = f"gencap_{gen_id}_{asset_id}_{period_idx}_{t_idx}"
                        
                        # Variables and coefficients for constraint
                        constraint_vars = []
                        constraint_coefs = []
                        
                        # Generation variable
                        constraint_vars.append(var_gen[(gen_id, asset_id, period_idx, t_idx)])
                        constraint_coefs.append(1.0)
                        
                        # Investment decision variable
                        if (asset_id, period_idx) in var_invest:
                            constraint_vars.append(var_invest[(asset_id, period_idx)])
                            constraint_coefs.append(-gen_pmax)  # Multiply by negative pmax
                        
                        # Add generation capacity constraint
                        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(constraint_vars, constraint_coefs)],
                            senses=["L"],  # Less than or equal
                            rhs=[0.0],  # Right-hand side
                            names=[con_name]
                        )
            
        # Storage capacity constraints based on investment decisions
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    for s in S:
                        stor_row = gen_data[(gen_data['id'] == s) & (gen_data['time'] == t)]
                        if stor_row.empty:
                            continue
                            
                        P_max = safe_float(stor_row.iloc[0]['pmax'])
                        E_max = safe_float(stor_row.iloc[0]['emax'])
                        
                        # Skip if storage variables don't exist
                        if (s, asset_id, period_idx, t_idx) not in var_pch or \
                           (s, asset_id, period_idx, t_idx) not in var_pdis or \
                           (s, asset_id, period_idx, t_idx) not in var_e:
                            continue
                            
                        # Constraint for charge power capacity
                        con_name_ch = f"storch_{s}_{asset_id}_{period_idx}_{t_idx}"
                        problem.linear_constraints.add(
                            lin_expr=[[
                                [var_pch[(s, asset_id, period_idx, t_idx)], var_invest[(asset_id, period_idx)]],
                                [1.0, -P_max]
                            ]],
                            senses=["L"],
                            rhs=[0.0],
                            names=[con_name_ch]
                        )
                        
                        # Constraint for discharge power capacity
                        con_name_dis = f"stordis_{s}_{asset_id}_{period_idx}_{t_idx}"
                        problem.linear_constraints.add(
                            lin_expr=[[
                                [var_pdis[(s, asset_id, period_idx, t_idx)], var_invest[(asset_id, period_idx)]],
                                [1.0, -P_max]
                            ]],
                            senses=["L"],
                            rhs=[0.0],
                            names=[con_name_dis]
                        )
                        
                        # Constraint for energy capacity
                        con_name_e = f"store_{s}_{asset_id}_{period_idx}_{t_idx}"
                        problem.linear_constraints.add(
                            lin_expr=[[
                                [var_e[(s, asset_id, period_idx, t_idx)], var_invest[(asset_id, period_idx)]],
                                [1.0, -E_max]
                            ]],
                            senses=["L"],
                            rhs=[0.0],
                            names=[con_name_e]
                        )
        
        # Storage dynamics constraints
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                times = representative_periods[asset_id][period_idx]['periods']
                for s in S:
                    # Get storage parameters
                    stor_rows = gen_data[gen_data['id'] == s]
                    if stor_rows.empty:
                        continue
                        
                    # Get efficiency, assuming same for all time periods
                    eta = safe_float(stor_rows.iloc[0]['eta'])
                    einitial = safe_float(stor_rows.iloc[0]['einitial'])
                    
                    # Initial storage state constraint
                    if (s, asset_id, period_idx, 0) in var_e:
                        con_name_init = f"stor_init_{s}_{asset_id}_{period_idx}"
                        problem.linear_constraints.add(
                            lin_expr=[[
                                [var_e[(s, asset_id, period_idx, 0)], var_invest[(asset_id, period_idx)]],
                                [1.0, -einitial]
                            ]],
                            senses=["E"],
                            rhs=[0.0],
                            names=[con_name_init]
                        )
                    
                    # Final storage state constraint (equal to initial)
                    if (s, asset_id, period_idx, len(times)-1) in var_e:
                        con_name_final = f"stor_final_{s}_{asset_id}_{period_idx}"
                        problem.linear_constraints.add(
                            lin_expr=[[
                                [var_e[(s, asset_id, period_idx, len(times)-1)], var_invest[(asset_id, period_idx)]],
                                [1.0, -einitial]
                            ]],
                            senses=["E"],
                            rhs=[0.0],
                            names=[con_name_final]
                        )
                    
                    # Storage dynamics constraints for each time period
                    for t_idx in range(len(times)-1):
                        if (s, asset_id, period_idx, t_idx) in var_e and \
                           (s, asset_id, period_idx, t_idx+1) in var_e and \
                           (s, asset_id, period_idx, t_idx) in var_pch and \
                           (s, asset_id, period_idx, t_idx) in var_pdis:
                            
                            con_name_dyn = f"stor_dyn_{s}_{asset_id}_{period_idx}_{t_idx}"
                            
                            # E(t+1) = E(t) + eta*Pch(t) - (1/eta)*Pdis(t)
                            problem.linear_constraints.add(
                                lin_expr=[[
                                    [
                                        var_e[(s, asset_id, period_idx, t_idx+1)],
                                        var_e[(s, asset_id, period_idx, t_idx)],
                                        var_pch[(s, asset_id, period_idx, t_idx)],
                                        var_pdis[(s, asset_id, period_idx, t_idx)]
                                    ],
                                    [1.0, -1.0, -eta*delta_t, (1.0/eta)*delta_t]
                                ]],
                                senses=["E"],
                                rhs=[0.0],
                                names=[con_name_dyn]
                            )
        
        # 5. Solve the model
        problem.solve()
        
        # 6. Process the results
        status = problem.solution.get_status()
        print(f"[OPTIMIZE] Solver status: {status}")
        
        # If solved successfully
        if status in [1, 101, 102]:  # Optimal, integer optimal, or optimal with rounding
            obj_value = problem.solution.get_objective_value()
            print(f"[OPTIMIZE] Final cost: ${obj_value:,.2f}")
            
            # Extract investment decisions
            investment_decisions = []
            investment_costs = {}
            
            for (asset_id, period_idx), var_name in var_invest.items():
                decision_value = problem.solution.get_values(var_name)
                if decision_value > 0.5:  # Binary variable is 1
                    full_periods, remainder, _ = calculate_lifetime_periods(
                        asset_lifetimes[asset_id], planning_horizon
                    )
                    
                    # Calculate years covered by this period
                    if period_idx < full_periods:
                        period_length = asset_lifetimes[asset_id]
                    else:
                        period_length = remainder
                    
                    start_year_period = start_year + period_idx * asset_lifetimes[asset_id]
                    end_year_period = start_year_period + period_length - 1
                    
                    investment_decisions.append({
                        'asset_id': asset_id,
                        'lifetime_period': period_idx,
                        'start_year': start_year_period,
                        'end_year': end_year_period,
                        'length': period_length,
                        'decision': 1
                    })
                    
                    # Store the investment cost
                    cost = safe_float(asset_capex[asset_id])
                    investment_costs[(asset_id, period_idx)] = cost
                else:
                    investment_decisions.append({
                        'asset_id': asset_id,
                        'lifetime_period': period_idx,
                        'decision': 0
                    })
            
            # Extract generation by asset and period
            generation_by_period = {}
            
            # For conventional generators
            for (gen_id, asset_id, period_idx, t_idx), var_name in var_gen.items():
                gen_value = problem.solution.get_values(var_name)
                
                # Initialize if not exists
                if (asset_id, period_idx) not in generation_by_period:
                    generation_by_period[(asset_id, period_idx)] = []
                
                # Get the time for this period
                t = representative_periods[asset_id][period_idx]['periods'][t_idx]
                
                # Add to generation results
                generation_by_period[(asset_id, period_idx)].append({
                    'id': gen_id,
                    'time': t,
                    'gen': gen_value
                })
                
            # For storage (net output = discharge - charge)
            for s in S:
                for asset_id, periods in lifetime_periods.items():
                    for period_idx in periods:
                        for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                            if (s, asset_id, period_idx, t_idx) in var_pdis and (s, asset_id, period_idx, t_idx) in var_pch:
                                # Get discharge and charge values
                                pdis_value = problem.solution.get_values(var_pdis[(s, asset_id, period_idx, t_idx)])
                                pch_value = problem.solution.get_values(var_pch[(s, asset_id, period_idx, t_idx)])
                                
                                # Calculate net output
                                net_output = pdis_value - pch_value
                                
                                # Initialize if not exists
                                if (asset_id, period_idx) not in generation_by_period:
                                    generation_by_period[(asset_id, period_idx)] = []
                                
                                # Add to generation results
                                generation_by_period[(asset_id, period_idx)].append({
                                    'id': s,
                                    'time': t,
                                    'gen': net_output
                                })
            
            # Convert to DataFrames
            for key in generation_by_period:
                generation_by_period[key] = pd.DataFrame(generation_by_period[key])
            
            # Extract storage state information
            storage_by_period = {}
            
            for s in S:
                for asset_id, periods in lifetime_periods.items():
                    for period_idx in periods:
                        for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                            if (s, asset_id, period_idx, t_idx) in var_e and \
                               (s, asset_id, period_idx, t_idx) in var_pch and \
                               (s, asset_id, period_idx, t_idx) in var_pdis:
                                
                                # Get variable values
                                e_value = problem.solution.get_values(var_e[(s, asset_id, period_idx, t_idx)])
                                pch_value = problem.solution.get_values(var_pch[(s, asset_id, period_idx, t_idx)])
                                pdis_value = problem.solution.get_values(var_pdis[(s, asset_id, period_idx, t_idx)])
                                
                                # Initialize if not exists
                                if (asset_id, period_idx) not in storage_by_period:
                                    storage_by_period[(asset_id, period_idx)] = []
                                
                                # Add to storage results
                                storage_by_period[(asset_id, period_idx)].append({
                                    'storage_id': s,
                                    'time': t,
                                    'E': e_value,
                                    'P_charge': pch_value,
                                    'P_discharge': pdis_value
                                })
            
            # Convert to DataFrames
            for key in storage_by_period:
                storage_by_period[key] = pd.DataFrame(storage_by_period[key])
            
            # Extract marginal prices from power balance constraints
            marginal_prices = []
            
            for asset_id, periods in lifetime_periods.items():
                for period_idx in periods:
                    for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                        for _, bus_row in bus.iterrows():
                            bus_i = int(bus_row['bus_i'])
                            
                            constraint_name = f"balance_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                            try:
                                constraint_idx = problem.linear_constraints.get_indices(constraint_name)
                                dual_value = problem.solution.get_dual_values(constraint_idx)
                                
                                # Adjusting for the scaling factor to get per-hour price
                                scaling_factor = safe_float(representative_periods[asset_id][period_idx]['scaling_factor'])
                                adjusted_price = dual_value / scaling_factor if scaling_factor > 0 else dual_value
                                
                                marginal_prices.append({
                                    'asset_id': asset_id,
                                    'lifetime_period': period_idx,
                                    'time': t,
                                    'bus': bus_i,
                                    'price': adjusted_price
                                })
                            except CplexError:
                                print(f"[OPTIMIZE] Warning: Could not get dual value for {constraint_name}")
            
            marginal_prices_df = pd.DataFrame(marginal_prices)
            
            # Extract slack variable information
            slack_info = []
            
            for asset_id, periods in lifetime_periods.items():
                for period_idx in periods:
                    for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                        for _, bus_row in bus.iterrows():
                            bus_i = int(bus_row['bus_i'])
                            
                            # Get slack variable values
                            slack_pos_name = f"slack_pos_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                            slack_neg_name = f"slack_neg_{bus_i}_{asset_id}_{period_idx}_{t_idx}"
                            
                            try:
                                slack_pos_value = problem.solution.get_values(slack_pos_name)
                                slack_neg_value = problem.solution.get_values(slack_neg_name)
                                
                                # Only record non-zero slacks
                                if slack_pos_value > 1e-6 or slack_neg_value > 1e-6:
                                    slack_info.append({
                                        'asset_id': asset_id,
                                        'lifetime_period': period_idx,
                                        'time': t,
                                        'bus': bus_i,
                                        'unmet_demand': slack_pos_value,
                                        'excess_generation': slack_neg_value
                                    })
                            except CplexError:
                                pass
            
            slack_info_df = pd.DataFrame(slack_info) if slack_info else pd.DataFrame()
            
            # Calculate operational and investment costs
            operational_cost = 0
            investment_cost = 0
            
            # For each variable in the objective function
            for i in range(problem.variables.get_num()):
                var_name = problem.variables.get_names(i)
                var_value = problem.solution.get_values(i)
                obj_coef = problem.objective.get_linear(i)
                
                # Calculate contribution to objective
                contribution = var_value * obj_coef
                
                # Categorize cost
                if var_name.startswith('inv_'):
                    investment_cost += contribution
                elif var_name.startswith('g_') or var_name.startswith('pc_') or var_name.startswith('pd_'):
                    operational_cost += contribution
            
            # Prepare results
            results = {
                'investment': pd.DataFrame(investment_decisions),
                'generation_by_period': generation_by_period,
                'storage_by_period': storage_by_period,
                'investment_costs': investment_costs,
                'marginal_prices': marginal_prices_df,
                'slack_info': slack_info_df,
                'cost_summary': {
                    'total_cost': obj_value,
                    'operational_cost': operational_cost,
                    'investment_cost': investment_cost
                }
            }
            
            return results
        else:
            print(f"[OPTIMIZE] Optimization failed with status code {status}")
            return None
            
    except CplexError as e:
        print(f"[OPTIMIZE] Error in investment model: {e}")
        return None

def run_two_stage_investment_model(
    gen_time_series, branch, bus, demand_time_series, 
    planning_horizon=10, start_year=2023, 
    asset_lifetimes=None, asset_capex=None,
    operational_periods_per_year=4, hours_per_period=24,
    delta_t=1,
    min_usage_threshold=0.1  # Minimum threshold to consider an asset used (10% of capacity by default)
):
    """
    Two-stage optimization model for power system planning with investment decisions.
    
    Stage 1: Run DCOPF for all time periods to determine which assets are actually needed.
    Stage 2: Make investment decisions based on DCOPF results.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        planning_horizon: Planning horizon in years
        start_year: Starting year for the planning horizon
        asset_lifetimes: Dict mapping asset IDs to lifetimes
        asset_capex: Dict mapping asset IDs to capital costs
        operational_periods_per_year: Number of representative operational periods per year
        hours_per_period: Hours in each operational period
        delta_t: Time step duration in hours
        min_usage_threshold: Minimum threshold of capacity usage to consider an asset needed
        
    Returns:
        Dictionary with optimization results
    """
    print("[OPTIMIZE] Entering two-stage optimization model...")
    
    # Ensure planning_horizon is an integer
    planning_horizon = int(planning_horizon)
    
    # Prepare generator data with investment-related fields if needed
    gen_data = create_gen_data_for_investment(gen_time_series, planning_horizon, start_year)
    
    # Extract mandatory assets (those with investment_required = 0)
    mandatory_assets = []
    for _, row in gen_data.iterrows():
        if 'investment_required' in row and row['investment_required'] == 0:
            if row['id'] not in mandatory_assets:
                mandatory_assets.append(int(row['id']))
    
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
            asset_lifetimes[asset_id] = int(lifetime)  # Ensure it's an integer
    
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
            asset_capex[asset_id] = safe_float(capex)  # Ensure it's a float
    
    # 1. Prepare lifetime period data
    lifetime_periods = create_lifetime_periods_mapping(asset_lifetimes, planning_horizon)
    
    # 2. Create representative operational periods
    start_date = datetime(start_year, 1, 1)
    operational_periods = create_typical_periods(
        start_date, 
        num_periods=operational_periods_per_year, 
        hours_per_period=hours_per_period
    )
    representative_periods = create_representative_periods(lifetime_periods, operational_periods)
    
    # Identify storage vs. non-storage units
    S = gen_data[gen_data['emax'] > 0]['id'].unique()  # set of storage IDs
    
    # --------------------------
    # STAGE 1: Run DCOPF to determine asset utilization
    # --------------------------
    
    asset_utilization = {}  # Dictionary to store utilization by asset ID
    asset_max_capacity = {}  # Dictionary to store maximum capacity by asset ID
    asset_period_counts = {}  # Track number of periods per asset for proper normalization
    
    # Initialize utilization and max capacity for each asset
    for _, row in gen_data.iterrows():
        asset_id = int(row['id'])
        if asset_id not in asset_utilization:
            asset_utilization[asset_id] = 0.0
            asset_max_capacity[asset_id] = 0.0
            asset_period_counts[asset_id] = 0
        
        # Update max capacity if higher than current value
        capacity = safe_float(row['pmax'])
        if capacity > asset_max_capacity[asset_id]:
            asset_max_capacity[asset_id] = capacity
    
    try:
        # For each lifetime period and representative operational period
        for asset_id, periods in lifetime_periods.items():
            for period_idx in periods:
                # Create a CPLEX problem for this period
                dcopf_model = cplex.Cplex()
                dcopf_model.objective.set_sense(dcopf_model.objective.sense.minimize)
                
                # Variables for DCOPF
                var_gen = {}  # Generation variables
                var_flow = {}  # Power flow variables
                var_angle = {}  # Bus angle variables
                var_pch = {}   # Storage charge variables
                var_pdis = {}  # Storage discharge variables
                var_e = {}     # Storage energy state variables
                
                # For each time period in this lifetime period
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    # Get scaling factor for this lifetime period
                    scaling_factor = safe_float(representative_periods[asset_id][period_idx]['scaling_factor'])
                    
                    # Add generation variables for each generator for each time period
                    for _, gen_row in gen_data[gen_data['time'] == t].iterrows():
                        gen_id = int(gen_row['id'])
                        
                        # Skip storage units here - they'll be handled separately
                        if gen_id in S:
                            continue
                        
                        gen_name = f"g_{gen_id}_{t_idx}"
                        
                        # Generation costs should be scaled by the time period's weight
                        gen_cost = safe_float(gen_row['gencost']) * scaling_factor
                        gen_pmin = safe_float(gen_row['pmin'])
                        gen_pmax = safe_float(gen_row['pmax'])
                        
                        # Add generation variable
                        dcopf_model.variables.add(
                            obj=[gen_cost],  # Generation cost in objective
                            lb=[0],  # Lower bound
                            ub=[gen_pmax],  # Upper bound
                            types=[dcopf_model.variables.type.continuous],
                            names=[gen_name]
                        )
                        
                        # Store the variable index
                        var_gen[(gen_id, t_idx)] = gen_name
                    
                    # Add storage variables (charge, discharge, energy state)
                    for s in S:
                        # Get storage parameters
                        stor_row = gen_data[(gen_data['id'] == s) & (gen_data['time'] == t)]
                        if stor_row.empty:
                            continue
                        
                        stor_row = stor_row.iloc[0]
                        P_max = safe_float(stor_row['pmax'])
                        E_max = safe_float(stor_row['emax'])
                        
                        # Charge variable
                        pch_name = f"pc_{s}_{t_idx}"
                        dcopf_model.variables.add(
                            obj=[0.001 * scaling_factor],  # Small cost to prevent unnecessary cycling
                            lb=[0],
                            ub=[P_max],
                            types=[dcopf_model.variables.type.continuous],
                            names=[pch_name]
                        )
                        var_pch[(s, t_idx)] = pch_name
                        
                        # Discharge variable
                        pdis_name = f"pd_{s}_{t_idx}"
                        dcopf_model.variables.add(
                            obj=[0.001 * scaling_factor],  # Small cost to prevent unnecessary cycling
                            lb=[0],
                            ub=[P_max],
                            types=[dcopf_model.variables.type.continuous],
                            names=[pdis_name]
                        )
                        var_pdis[(s, t_idx)] = pdis_name
                        
                        # Energy state variable
                        e_name = f"e_{s}_{t_idx}"
                        dcopf_model.variables.add(
                            obj=[0],  # No cost in objective
                            lb=[0],
                            ub=[E_max],
                            types=[dcopf_model.variables.type.continuous],
                            names=[e_name]
                        )
                        var_e[(s, t_idx)] = e_name
                    
                    # Add power flow variables for each branch and time period
                    for _, branch_row in branch.iterrows():
                        from_bus = int(branch_row['fbus'])
                        to_bus = int(branch_row['tbus'])
                        flow_name = f"f_{from_bus}_{to_bus}_{t_idx}"
                        
                        # Line limit
                        line_limit = safe_float(branch_row['ratea'])
                        
                        # Add flow variable
                        dcopf_model.variables.add(
                            obj=[0],  # No cost in objective
                            lb=[-line_limit],  # Lower bound - can be negative
                            ub=[line_limit],  # Upper bound
                            types=[dcopf_model.variables.type.continuous],
                            names=[flow_name]
                        )
                        
                        # Store the variable index
                        var_flow[(from_bus, to_bus, t_idx)] = flow_name
                    
                    # Add bus angle variables for each bus and time period
                    for _, bus_row in bus.iterrows():
                        bus_i = int(bus_row['bus_i'])
                        angle_name = f"theta_{bus_i}_{t_idx}"
                        
                        # Add angle variable
                        dcopf_model.variables.add(
                            obj=[0],  # No cost in objective
                            lb=[-math.pi] if bus_i != 1 else [0],  # Fix angle of reference bus to 0
                            ub=[math.pi] if bus_i != 1 else [0],  # Fix angle of reference bus to 0
                            types=[dcopf_model.variables.type.continuous],
                            names=[angle_name]
                        )
                        
                        # Store the variable index
                        var_angle[(bus_i, t_idx)] = angle_name
                
                # Add constraints for DCOPF
                
                # Power balance constraints: generation - demand = net flow at each bus
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    for _, bus_row in bus.iterrows():
                        bus_i = int(bus_row['bus_i'])
                        
                        # Get demand at this bus for this time period
                        bus_demand = safe_float(demand_time_series[(demand_time_series['time'] == t) & 
                                                    (demand_time_series['bus'] == bus_i)]['pd'].sum())
                        
                        # Get all generators at this bus
                        bus_gens = gen_data[(gen_data['time'] == t) & (gen_data['bus'] == bus_i)]
                        
                        # Get all lines connected to this bus
                        outgoing_lines = branch[branch['fbus'] == bus_i]
                        incoming_lines = branch[branch['tbus'] == bus_i]
                        
                        # Constraint name
                        con_name = f"balance_{bus_i}_{t_idx}"
                        
                        # Variables and coefficients for constraint
                        constraint_vars = []
                        constraint_coefs = []
                        
                        # Add generator variables at this bus (non-storage)
                        for _, gen_row in bus_gens.iterrows():
                            gen_id = int(gen_row['id'])
                            if gen_id not in S and (gen_id, t_idx) in var_gen:
                                constraint_vars.append(var_gen[(gen_id, t_idx)])
                                constraint_coefs.append(1.0)  # Positive contribution
                        
                        # Add storage discharge (positive) and charge (negative) variables
                        for s in S:
                            s_row = bus_gens[bus_gens['id'] == s]
                            if not s_row.empty and (s, t_idx) in var_pdis and (s, t_idx) in var_pch:
                                # Discharge (positive contribution)
                                constraint_vars.append(var_pdis[(s, t_idx)])
                                constraint_coefs.append(1.0)
                                
                                # Charge (negative contribution) 
                                constraint_vars.append(var_pch[(s, t_idx)])
                                constraint_coefs.append(-1.0)
                        
                        # Add outgoing line flow variables (negative contribution)
                        for _, line in outgoing_lines.iterrows():
                            to_bus = int(line['tbus'])
                            if (bus_i, to_bus, t_idx) in var_flow:
                                constraint_vars.append(var_flow[(bus_i, to_bus, t_idx)])
                                constraint_coefs.append(-1.0)  # Negative contribution
                        
                        # Add incoming line flow variables (positive contribution)
                        for _, line in incoming_lines.iterrows():
                            from_bus = int(line['fbus'])
                            if (from_bus, bus_i, t_idx) in var_flow:
                                constraint_vars.append(var_flow[(from_bus, bus_i, t_idx)])
                                constraint_coefs.append(1.0)  # Positive contribution
                        
                        # Add slack variables to handle potential infeasibilities
                        # Positive slack for unmet demand
                        slack_pos_name = f"slack_pos_{bus_i}_{t_idx}"
                        dcopf_model.variables.add(
                            obj=[10000.0 * scaling_factor],  # High penalty cost
                            lb=[0],
                            ub=[cplex.infinity],
                            types=[dcopf_model.variables.type.continuous],
                            names=[slack_pos_name]
                        )
                        constraint_vars.append(slack_pos_name)
                        constraint_coefs.append(1.0)
                        
                        # Negative slack for excess generation
                        slack_neg_name = f"slack_neg_{bus_i}_{t_idx}"
                        dcopf_model.variables.add(
                            obj=[10000.0 * scaling_factor],  # High penalty cost
                            lb=[0],
                            ub=[cplex.infinity],
                            types=[dcopf_model.variables.type.continuous],
                            names=[slack_neg_name]
                        )
                        constraint_vars.append(slack_neg_name)
                        constraint_coefs.append(-1.0)
                        
                        # Add power balance constraint with slack variables
                        if len(constraint_vars) > 0:
                            dcopf_model.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(constraint_vars, constraint_coefs)],
                                senses=["E"],  # Equality constraint
                                rhs=[safe_float(bus_demand)],  # Equal to demand
                                names=[con_name]
                            )
                
                # Power flow constraints: flow proportional to angle difference
                for t_idx, t in enumerate(representative_periods[asset_id][period_idx]['periods']):
                    for _, branch_row in branch.iterrows():
                        from_bus = int(branch_row['fbus'])
                        to_bus = int(branch_row['tbus'])
                        
                        # Get branch reactance and susceptance
                        x = safe_float(branch_row['x'])
                        susceptance = 1.0 / x
                        
                        # Constraint name
                        con_name = f"flow_{from_bus}_{to_bus}_{t_idx}"
                        
                        # Variables and coefficients for constraint
                        constraint_vars = []
                        constraint_coefs = []
                        
                        # Flow variable
                        if (from_bus, to_bus, t_idx) in var_flow:
                            constraint_vars.append(var_flow[(from_bus, to_bus, t_idx)])
                            constraint_coefs.append(1.0)
                        
                        # From bus angle
                        if (from_bus, t_idx) in var_angle:
                            constraint_vars.append(var_angle[(from_bus, t_idx)])
                            constraint_coefs.append(-susceptance)
                        
                        # To bus angle
                        if (to_bus, t_idx) in var_angle:
                            constraint_vars.append(var_angle[(to_bus, t_idx)])
                            constraint_coefs.append(susceptance)
                        
                        # Add flow constraint
                        dcopf_model.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(constraint_vars, constraint_coefs)],
                            senses=["E"],  # Equality constraint
                            rhs=[0.0],  # Right-hand side
                            names=[con_name]
                        )
                
                # Storage dynamics constraints
                for s in S:
                    # Get storage parameters
                    stor_rows = gen_data[gen_data['id'] == s]
                    if stor_rows.empty:
                        continue
                    
                    # Get efficiency, assuming same for all time periods
                    eta = safe_float(stor_rows.iloc[0]['eta'])
                    einitial = safe_float(stor_rows.iloc[0]['einitial'])
                    
                    # Initial storage state constraint
                    if (s, 0) in var_e:
                        con_name_init = f"stor_init_{s}"
                        dcopf_model.linear_constraints.add(
                            lin_expr=[[
                                [var_e[(s, 0)]],
                                [1.0]
                            ]],
                            senses=["E"],
                            rhs=[einitial],
                            names=[con_name_init]
                        )
                    
                    # Final storage state constraint (equal to initial)
                    times = representative_periods[asset_id][period_idx]['periods']
                    if (s, len(times)-1) in var_e:
                        con_name_final = f"stor_final_{s}"
                        dcopf_model.linear_constraints.add(
                            lin_expr=[[
                                [var_e[(s, len(times)-1)]],
                                [1.0]
                            ]],
                            senses=["E"],
                            rhs=[einitial],
                            names=[con_name_final]
                        )
                    
                    # Storage dynamics constraints for each time period
                    for t_idx in range(len(times)-1):
                        if (s, t_idx) in var_e and \
                           (s, t_idx+1) in var_e and \
                           (s, t_idx) in var_pch and \
                           (s, t_idx) in var_pdis:
                            
                            con_name_dyn = f"stor_dyn_{s}_{t_idx}"
                            
                            # E(t+1) = E(t) + eta*Pch(t) - (1/eta)*Pdis(t)
                            dcopf_model.linear_constraints.add(
                                lin_expr=[[
                                    [
                                        var_e[(s, t_idx+1)],
                                        var_e[(s, t_idx)],
                                        var_pch[(s, t_idx)],
                                        var_pdis[(s, t_idx)]
                                    ],
                                    [1.0, -1.0, -eta*delta_t, (1.0/eta)*delta_t]
                                ]],
                                senses=["E"],
                                rhs=[0.0],
                                names=[con_name_dyn]
                            )
                
                # Solve DCOPF
                dcopf_model.solve()
                
                # Check if solved successfully
                dcopf_status = dcopf_model.solution.get_status()
                
                if dcopf_status in [1, 101, 102]:  # Optimal, integer optimal, or optimal with rounding
                    # Count period for each asset
                    period_count = len(representative_periods[asset_id][period_idx]['periods'])
                    
                    # Extract generation results for each asset
                    for (gen_id, t_idx), var_name in var_gen.items():
                        gen_value = dcopf_model.solution.get_values(var_name)
                        
                        # Update utilization and period count
                        asset_utilization[gen_id] += gen_value
                        if t_idx == 0:  # Only count each period once per asset
                            asset_period_counts[gen_id] += 1
                    
                    # Extract storage results
                    for s in S:
                        for t_idx, _ in enumerate(representative_periods[asset_id][period_idx]['periods']):
                            if (s, t_idx) in var_pdis and (s, t_idx) in var_pch:
                                pdis_value = dcopf_model.solution.get_values(var_pdis[(s, t_idx)])
                                pch_value = dcopf_model.solution.get_values(var_pch[(s, t_idx)])
                                
                                # Update utilization based on maximum of charge and discharge
                                asset_utilization[s] += max(pdis_value, pch_value)
                                if t_idx == 0:  # Only count each period once per asset
                                    asset_period_counts[s] += 1
                
                else:
                    print(f"[OPTIMIZE] DCOPF failed for period {period_idx} with status {dcopf_status}")
                    # If DCOPF fails, we'll assume all assets are needed
                    for gen_id in asset_utilization.keys():
                        asset_utilization[gen_id] = asset_max_capacity[gen_id] * operational_periods_per_year * hours_per_period
        
        # Normalize utilization by dividing by the number of periods and hours
        # This gives us the average MW used over all periods
        with open('debug_normalization.txt', 'w') as f:
            f.write("===== NORMALIZATION DEBUG =====\n\n")
            for asset_id in asset_utilization.keys():
                # Calculate total hours across all periods
                total_hours = asset_period_counts[asset_id] * hours_per_period
                
                # Only normalize if we have periods
                if total_hours > 0:
                    # Normalize to get average MW used
                    asset_utilization[asset_id] = asset_utilization[asset_id] / total_hours
                
                f.write(f"Asset {asset_id}:\n")
                f.write(f"  - Raw utilization: {asset_utilization[asset_id]:.2f} MW\n")
                f.write(f"  - Max capacity: {asset_max_capacity[asset_id]:.2f} MW\n")
                f.write(f"  - Periods counted: {asset_period_counts[asset_id]}\n")
                f.write(f"  - Total hours: {total_hours}\n")
                
                # Calculate utilization percentage
                if asset_max_capacity[asset_id] > 0:
                    utilization_pct = asset_utilization[asset_id] / asset_max_capacity[asset_id] * 100
                    f.write(f"  - Utilization %: {utilization_pct:.2f}%\n\n")
                else:
                    f.write(f"  - Utilization %: 0.00% (capacity is zero)\n\n")
        
        # --------------------------
        # STAGE 2: Make investment decisions based on DCOPF results
        # --------------------------
        
        # Determine which assets are needed based on utilization
        assets_needed = []
        
        # Save debug information to a file
        with open('debug_utilization.txt', 'w') as f:
            f.write("===== DEBUG INFORMATION =====\n\n")
            
            # Raw utilization values
            f.write("[DEBUG] Raw utilization values:\n")
            for asset_id, utilization in asset_utilization.items():
                f.write(f"Asset {asset_id}: Raw utilization = {utilization:.2f}, Capacity = {asset_max_capacity[asset_id]:.2f}\n")
            
            # Lifetime periods structure
            f.write("\n[DEBUG] Lifetime periods structure:\n")
            for asset_id, periods in lifetime_periods.items():
                f.write(f"Asset {asset_id}: Periods = {periods}\n")
            
            # Calculate utilization percentages
            f.write("\n[DEBUG] Normalized utilization values:\n")
            for asset_id, utilization in asset_utilization.items():
                utilization_pct = utilization / asset_max_capacity[asset_id] if asset_max_capacity[asset_id] > 0 else 0
                f.write(f"Asset {asset_id}: Utilization % = {utilization_pct*100:.2f}%\n")
            
            # Mandatory assets
            f.write("\n[DEBUG] Mandatory assets:\n")
            f.write(f"{mandatory_assets}\n")
        
        print("\n[OPTIMIZE] Asset utilization from DCOPF stage:")
        for asset_id, utilization in asset_utilization.items():
            # Simple normalization - just compare to maximum capacity
            utilization_pct = utilization / asset_max_capacity[asset_id] if asset_max_capacity[asset_id] > 0 else 0
            
            print(f"Asset {asset_id}: {utilization:.2f} MW used out of {asset_max_capacity[asset_id]:.2f} MW capacity ({utilization_pct*100:.2f}%)")
            
            # Check if the asset is needed (utilization > threshold or mandatory)
            if utilization_pct >= min_usage_threshold or asset_id in mandatory_assets:
                assets_needed.append(asset_id)
        
        print(f"[OPTIMIZE] Assets needed based on DCOPF results: {assets_needed}")
        
        # Create a new CPLEX problem for the investment model
        problem = cplex.Cplex()
        problem.objective.set_sense(problem.objective.sense.minimize)
        problem.parameters.mip.display.set(4)  # Detailed display
        
        # Investment variables - binary decision variables for needed assets only
        var_invest = {}  # Investment decision variables
        investment_decisions = []
        investment_costs = {}
        
        for asset_id in assets_needed:
            # Get the periods for this asset
            periods = lifetime_periods[asset_id]
            
            for period_idx in periods:
                # Binary variable name: inv_assetID_periodIdx
                var_name = f"inv_{asset_id}_{period_idx}"
                
                # Add binary variable for investment decision
                problem.variables.add(
                    obj=[safe_float(asset_capex[asset_id])],  # Use the capex as the objective coefficient
                    lb=[0],  # Lower bound
                    ub=[1],  # Upper bound
                    types=[problem.variables.type.binary],  # Binary variable
                    names=[var_name]
                )
                
                # Store the variable index
                var_invest[(asset_id, period_idx)] = var_name
        
        # Add constraints for mandatory assets (must be invested in)
        for asset_id in mandatory_assets:
            if asset_id in assets_needed:
                periods = lifetime_periods[asset_id]
                for period_idx in periods:
                    if (asset_id, period_idx) in var_invest:
                        # Constraint name
                        con_name = f"mandatory_{asset_id}_{period_idx}"
                        
                        # Add constraint to force investment
                        problem.linear_constraints.add(
                            lin_expr=[[
                                [var_invest[(asset_id, period_idx)]],
                                [1.0]
                            ]],
                            senses=["E"],  # Equality constraint
                            rhs=[1.0],  # Must equal 1 (mandatory)
                            names=[con_name]
                        )
        
        # Solve the investment model
        problem.solve()
        
        # Process the results
        status = problem.solution.get_status()
        print(f"[OPTIMIZE] Investment solver status: {status}")
        
        # If solved successfully
        if status in [1, 101, 102]:  # Optimal, integer optimal, or optimal with rounding
            obj_value = problem.solution.get_objective_value()
            print(f"[OPTIMIZE] Final investment cost: ${obj_value:,.2f}")
            
            # Extract investment decisions
            for (asset_id, period_idx), var_name in var_invest.items():
                decision_value = problem.solution.get_values(var_name)
                if decision_value > 0.5:  # Binary variable is 1
                    full_periods, remainder, _ = calculate_lifetime_periods(
                        asset_lifetimes[asset_id], planning_horizon
                    )
                    
                    # Calculate years covered by this period
                    if period_idx < full_periods:
                        period_length = asset_lifetimes[asset_id]
                    else:
                        period_length = remainder
                    
                    start_year_period = start_year + period_idx * asset_lifetimes[asset_id]
                    end_year_period = start_year_period + period_length - 1
                    
                    investment_decisions.append({
                        'asset_id': asset_id,
                        'lifetime_period': period_idx,
                        'start_year': start_year_period,
                        'end_year': end_year_period,
                        'length': period_length,
                        'decision': 1
                    })
                    
                    # Store the investment cost
                    cost = safe_float(asset_capex[asset_id])
                    investment_costs[(asset_id, period_idx)] = cost
                else:
                    investment_decisions.append({
                        'asset_id': asset_id,
                        'lifetime_period': period_idx,
                        'decision': 0
                    })
            
            # Calculate total costs
            investment_cost = sum(investment_costs.values())
            
            # Prepare results
            results = {
                'investment': pd.DataFrame(investment_decisions),
                'investment_costs': investment_costs,
                'asset_utilization': asset_utilization,
                'assets_needed': assets_needed,
                'cost_summary': {
                    'total_cost': obj_value,
                    'investment_cost': investment_cost
                }
            }
            
            return results
        else:
            print(f"[OPTIMIZE] Investment optimization failed with status code {status}")
            return None
            
    except CplexError as e:
        print(f"[OPTIMIZE] Error in two-stage investment model: {e}")
        return None 