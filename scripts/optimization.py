#!/usr/bin/env python3

"""
dcopf.py

A DC Optimal Power Flow (DCOPF) solver using CPLEX.
This implementation focuses on extracting and analyzing marginal prices.
It also includes investment decisions for generation and storage assets.
"""

import sys
import os
import pandas as pd
import numpy as np
import math
from pandas.tseries.offsets import DateOffset

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

def dcopf(gen_time_series, branch, bus, demand_time_series, delta_t=1, 
          include_investment=False, planning_horizon=1, asset_lifetimes=None, asset_capex=None, 
          existing_investment=None):
    """
    DC Optimal Power Flow solver using CPLEX with optional investment decisions.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        delta_t: Time step duration in hours
        include_investment: Whether to include investment decisions
        planning_horizon: Planning horizon in years (only used if include_investment=True)
        asset_lifetimes: Dictionary mapping asset IDs to their lifetimes in years
        asset_capex: Dictionary mapping asset IDs to their capital costs
        existing_investment: Dictionary mapping asset IDs to boolean values indicating if they are already installed
    
    Returns:
        Dictionary with DCOPF results including generation, flows, prices, etc.
        If include_investment=True, also includes investment decisions.
    """
    # Function to safely convert any value to float
    def safe_float(value):
        return float(value)

    try:
        print("[DCOPF] Using CPLEX solver")
    except ImportError:
        print("[DCOPF] Error: CPLEX not available.")
        return None

    # Create CPLEX problem
    problem = cplex.Cplex()
    
    # Minimize objective function
    problem.objective.set_sense(problem.objective.sense.minimize)
    
    # Identify storage vs. non-storage units
    storage_data = gen_time_series[gen_time_series['emax'] > 0]
    S = storage_data['id'].unique()     # set of storage IDs
    non_storage_data = gen_time_series[gen_time_series['emax'] == 0]
    G = non_storage_data['id'].unique() # set of non-storage gen IDs

    print(f"[DCOPF] Found storage units: {S}, non-storage units: {G}")
    print("[DCOPF] Storage data sample:")
    if not storage_data.empty:
        print(storage_data[['id', 'bus', 'emax', 'pmax', 'eta']].head())

    # Time and bus sets
    N = bus['bus_i'].values
    T = sorted(demand_time_series['time'].unique())
    if not T:
        print("[DCOPF] No time steps found in demand_time_series. Returning None.")
        return None

    next_time = T[-1] + DateOffset(hours=delta_t)
    extended_T = list(T) + [next_time]
    
    # Check if we need to include investment decisions
    if include_investment:
        if asset_lifetimes is None or asset_capex is None:
            print("[DCOPF] Error: asset_lifetimes and asset_capex must be provided when include_investment=True")
            return None
        
        # Identify assets that require investment decisions
        # Defaults to all green assets (those with gencost=0 in our framework)
        investment_required = []
        print("[DCOPF] Determining assets requiring investment decisions:")
        
        for asset_id in list(G) + list(S):
            asset_data = gen_time_series[gen_time_series['id'] == asset_id]
            if 'investment_required' in asset_data.columns:
                # Use explicit flag if available
                if asset_data['investment_required'].iloc[0] == 1:
                    investment_required.append(asset_id)
                    print(f"  - Asset {asset_id}: Investment required (explicit flag)")
            elif asset_data['gencost'].iloc[0] == 0:
                # Otherwise, assume green assets (zero operational cost) need investment
                investment_required.append(asset_id)
                print(f"  - Asset {asset_id}: Investment required (green asset with zero operational cost)")
            else:
                print(f"  - Asset {asset_id}: No investment required (grey asset with operational cost)")
        
        print(f"[DCOPF] Assets requiring investment: {investment_required}")
    
    # 1. Create variables
    
    # Variable names must be valid CPLEX identifiers (no special chars)
    
    # a) Generation variables for non-storage generators
    gen_vars = {}
    gen_names = []
    gen_costs = []
    gen_lbs = []
    gen_ubs = []
    
    for g in G:
        for t in T:
            gen_row = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]
            if gen_row.empty:
                print(f"[DCOPF] Missing data for generator={g}, time={t}. Returning None.")
                return None
                
            pmin = safe_float(gen_row['pmin'].iloc[0])
            pmax = safe_float(gen_row['pmax'].iloc[0])
            cost = safe_float(gen_row['gencost'].iloc[0])
            
            var_name = f"g{g}t{t.strftime('%Y%m%d%H')}"
            gen_vars[g, t] = var_name
            gen_names.append(var_name)
            gen_costs.append(cost)
            gen_lbs.append(pmin)
            gen_ubs.append(pmax)
    
    problem.variables.add(
        obj=gen_costs,
        lb=gen_lbs,
        ub=gen_ubs,
        names=gen_names
    )
    
    # b) Voltage angle variables
    theta_vars = {}
    theta_names = []
    theta_lbs = []
    theta_ubs = []
    
    for i in N:
        for t in T:
            var_name = f"t{i}t{t.strftime('%Y%m%d%H')}"
            theta_vars[i, t] = var_name
            theta_names.append(var_name)
            theta_lbs.append(-cplex.infinity)
            theta_ubs.append(cplex.infinity)
    
    problem.variables.add(
        lb=theta_lbs,
        ub=theta_ubs,
        names=theta_names
    )
    
    # c) Flow variables
    flow_vars = {}
    flow_names = []
    flow_lbs = []
    flow_ubs = []
    
    for _, row in branch.iterrows():
        i = int(row['fbus'])
        j = int(row['tbus'])
        for t in T:
            var_name = f"f{i}{j}t{t.strftime('%Y%m%d%H')}"
            flow_vars[i, j, t] = var_name
            flow_names.append(var_name)
            flow_lbs.append(-cplex.infinity)
            flow_ubs.append(cplex.infinity)
    
    problem.variables.add(
        lb=flow_lbs,
        ub=flow_ubs,
        names=flow_names
    )
    
    # d) Storage variables (P_charge, P_discharge, E)
    pch_vars = {}
    pdis_vars = {}
    e_vars = {}
    
    # Add charge/discharge variables
    for s in S:
        s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
        P_max = safe_float(s_row['pmax'])  # Discharging max
        E_max = safe_float(s_row['emax'])  # Energy capacity
        
        for t in T:
            # Charge variables
            pch_name = f"pc{s}t{t.strftime('%Y%m%d%H')}"
            pch_vars[s, t] = pch_name
            problem.variables.add(
                lb=[0.0],
                ub=[abs(P_max)],
                names=[pch_name]
            )
            
            # Discharge variables
            pdis_name = f"pd{s}t{t.strftime('%Y%m%d%H')}"
            pdis_vars[s, t] = pdis_name
            problem.variables.add(
                lb=[0.0],
                ub=[abs(P_max)],
                names=[pdis_name]
            )
        
        # Energy variables
        for t in extended_T:
            e_name = f"e{s}t{t.strftime('%Y%m%d%H')}"
            e_vars[s, t] = e_name
            problem.variables.add(
                lb=[0.0],
                ub=[E_max],
                names=[e_name]
            )
    
    # e) Binary investment decision variables (if include_investment=True)
    invest_vars = {}
    
    if include_investment:
        print("[DCOPF] Adding investment decision variables")
        for asset_id in investment_required:
            var_name = f"b{asset_id}"
            invest_vars[asset_id] = var_name
            
            # Check if this asset is already installed
            is_already_installed = False
            if existing_investment and asset_id in existing_investment:
                is_already_installed = existing_investment[asset_id]
            
            # Add binary variable with coefficient = annual capex cost
            annual_capex = safe_float(asset_capex[asset_id]) / safe_float(asset_lifetimes[asset_id])
            annual_capex_horizon = annual_capex * safe_float(planning_horizon)  # Use safe_float for multiplication
            
            # If asset is already installed, fix binary variable to 1 with zero cost
            if is_already_installed:
                lb_value = 1.0  # Fixed to 1
                ub_value = 1.0  # Fixed to 1
                obj_value = 0.0  # No additional cost for already installed assets
                print(f"  - Asset {asset_id} is already installed, fixing binary variable to 1")
            else:
                lb_value = 0.0  # Can be 0 or 1
                ub_value = 1.0  # Can be 0 or 1
                obj_value = annual_capex_horizon  # Normal cost
            
            problem.variables.add(
                obj=[obj_value],  # Total capex over planning horizon or 0 if already installed
                lb=[lb_value],  # Explicit float
                ub=[ub_value],  # Explicit float
                types=["B"],  # Binary variable
                names=[var_name]
            )
            
            print(f"  - Added binary variable {var_name} with annual cost: ${annual_capex:,.2f}")
    
    # 2. Add constraints
    
    # a) DC Power Flow Constraints
    for idx_b, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        susceptance = safe_float(row_b['sus'])
        rate_a = safe_float(row_b['ratea'])  # Convert limit to float
        
        for t in T:
            # flow_i_j = susceptance * (theta_i - theta_j)
            flow_var = flow_vars[i, j, t]
            from_angle_var = theta_vars[i, t]
            to_angle_var = theta_vars[j, t]
            
            # Flow = (angle_from - angle_to) / x
            names = [flow_var, from_angle_var, to_angle_var]
            coeffs = [1.0, -susceptance, susceptance]
            
            problem.linear_constraints.add(
                lin_expr=[[names, coeffs]],
                senses=["E"],
                rhs=[0.0],
                names=[f"dc_flow_{i}_{j}_{t}"]
            )
            
            # Upper limit: FLOW[i, j, t] <= rate_a
            problem.linear_constraints.add(
                lin_expr=[[[flow_var], [1.0]]],
                senses=["L"],
                rhs=[rate_a],
                names=[f"up{i}{j}t{t.strftime('%Y%m%d%H')}"]
            )
            
            # Lower limit: FLOW[i, j, t] >= -rate_a
            problem.linear_constraints.add(
                lin_expr=[[[flow_var], [1.0]]],
                senses=["G"],
                rhs=[-rate_a],
                names=[f"lo{i}{j}t{t.strftime('%Y%m%d%H')}"]
            )
    
    # b) Initial storage state constraint
    for s in S:
        s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
        E_initial = safe_float(s_row['einitial'])
        
        # If investment is included and this is a storage asset requiring investment
        if include_investment and s in investment_required:
            # Initial SoC: E[T[0]] = E_initial * binary_variable
            problem.linear_constraints.add(
                lin_expr=[[[e_vars[s, T[0]], invest_vars[s]], [1.0, -E_initial]]],
                senses=["E"],
                rhs=[0.0],
                names=[f"inite{s}"]
            )
        else:
            # Initial SoC: E[T[0]] = E_initial (standard case)
            problem.linear_constraints.add(
                lin_expr=[[[e_vars[s, T[0]]], [1.0]]],
                senses=["E"],
                rhs=[E_initial],
                names=[f"inite{s}"]
            )
    
    # c) Storage dynamics
    for s in S:
        s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
        eta = safe_float(s_row['eta'])
        
        for idx_t, t in enumerate(T):
            next_t = extended_T[idx_t + 1]
            
            # E[next] = E[t] + eta*charge - (1/eta)*discharge
            e_t = e_vars[s, t]
            pch_t = pch_vars[s, t]
            pdis_t = pdis_vars[s, t]
            problem.linear_constraints.add(
                lin_expr=[[
                    [e_t, e_vars[s, next_t], pch_t, pdis_t],
                    [1.0, -1.0, -eta * safe_float(delta_t), safe_float(1.0/eta) * safe_float(delta_t)]
                ]],
                senses=["E"],
                rhs=[0.0],
                names=[f"sd{s}t{t.strftime('%Y%m%d%H')}"]
            )
    
    # d) Final storage state constraint
    for s in S:
        s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
        E_initial = safe_float(s_row['einitial'])
        
        # If investment is included and this is a storage asset requiring investment
        if include_investment and s in investment_required:
            # Final SoC: E[extended_T[-1]] = E_initial * binary_variable
            problem.linear_constraints.add(
                lin_expr=[[[e_vars[s, extended_T[-1]], invest_vars[s]], [1.0, -E_initial]]],
                senses=["E"],
                rhs=[0.0],
                names=[f"finsoc{s}"]
            )
        else:
            # Final SoC: E[extended_T[-1]] = E_initial (standard case)
            problem.linear_constraints.add(
                lin_expr=[[[e_vars[s, extended_T[-1]]], [1.0]]],
                senses=["E"],
                rhs=[E_initial],
                names=[f"finsoc{s}"]
            )
    
    # e) Slack bus constraint
    slack_bus = 1  # Bus 1 is the reference
    for t in T:
        problem.linear_constraints.add(
            lin_expr=[[[theta_vars[slack_bus, t]], [1.0]]],
            senses=["E"],
            rhs=[0.0],
            names=[f"slack{t.strftime('%Y%m%d%H')}"]
        )
    
    # f) Power balance constraints
    for t in T:
        for i in N:  # Include all buses, even those without generators
            # Prepare the power balance constraint
            lin_vars = []
            lin_coefs = []
            
            # Add non-storage generation
            for g in G:
                gen_rows = gen_time_series.loc[
                    (gen_time_series['id'] == g) & (gen_time_series['time'] == t)
                ]
                if not gen_rows.empty and gen_rows['bus'].iloc[0] == i:
                    lin_vars.append(gen_vars[g, t])
                    lin_coefs.append(1.0)
            
            # Add storage (discharge - charge)
            for s in S:
                stor_rows = gen_time_series.loc[
                    (gen_time_series['id'] == s) & (gen_time_series['time'] == t)
                ]
                if not stor_rows.empty and stor_rows['bus'].iloc[0] == i:
                    lin_vars.append(pdis_vars[s, t])
                    lin_coefs.append(1.0)
                    lin_vars.append(pch_vars[s, t])
                    lin_coefs.append(-1.0)
            
            # Add flows into the bus
            for idx_b, row_b in branch.iterrows():
                if row_b['tbus'] == i:  # Flow into bus i
                    lin_vars.append(flow_vars[int(row_b['fbus']), int(row_b['tbus']), t])
                    lin_coefs.append(1.0)
            
            # Add flows out of the bus
            for idx_b, row_b in branch.iterrows():
                if row_b['fbus'] == i:  # Flow out of bus i
                    lin_vars.append(flow_vars[int(row_b['fbus']), int(row_b['tbus']), t])
                    lin_coefs.append(-1.0)
            
            # Get demand at bus i
            pd_val = 0
            demands_at_bus = demand_time_series.loc[
                (demand_time_series['bus'] == i) & (demand_time_series['time'] == t),
                'pd'
            ]
            pd_val = demands_at_bus.sum() if not demands_at_bus.empty else 0
            
            # Add power balance constraint
            if lin_vars:
                problem.linear_constraints.add(
                    lin_expr=[[lin_vars, lin_coefs]],
                    senses=["E"],
                    rhs=[safe_float(pd_val)],
                    names=[f"pb{i}t{t.strftime('%Y%m%d%H')}"]
                )
    
    # g) Flow limits
    for idx_b, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        rate_a = safe_float(row_b['ratea'])
        
        for t in T:
            # Upper limit: FLOW[i, j, t] <= rate_a
            flow_var = flow_vars[i, j, t]
            problem.linear_constraints.add(
                lin_expr=[[[flow_var], [1.0]]],
                senses=["L"],
                rhs=[rate_a],
                names=[f"up{i}{j}t{t.strftime('%Y%m%d%H')}"]
            )
            
            # Lower limit: FLOW[i, j, t] >= -rate_a
            problem.linear_constraints.add(
                lin_expr=[[[flow_var], [1.0]]],
                senses=["G"],
                rhs=[-rate_a],
                names=[f"lo{i}{j}t{t.strftime('%Y%m%d%H')}"]
            )
    
    # h) Add storage cost to objective
    if len(S) > 0:
        # Simple storage cost to prevent unnecessary cycling
        for s in S:
            for t in T:
                problem.objective.set_linear(pch_vars[s, t], 0.001)
                problem.objective.set_linear(pdis_vars[s, t], 0.001)
    
    # i) Add investment constraints for generators and storage
    if include_investment:
        print("[DCOPF] Adding capacity constraints based on investment decisions")
        
        # For generators: 0 <= P <= Pmax * b
        for g in G:
            if g in investment_required:
                for t in T:
                    gen_row = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]
                    pmax = safe_float(gen_row['pmax'].iloc[0])
                    
                    # Add big-M constraint: P <= Pmax * b
                    problem.linear_constraints.add(
                        lin_expr=[[[gen_vars[g, t], invest_vars[g]], [1.0, -pmax]]],
                        senses=["L"],
                        rhs=[0.0],
                        names=[f"cap_g{g}t{t.strftime('%Y%m%d%H')}"]
                    )
                    
                    print(f"  - Added capacity constraint for generator {g} with Pmax={pmax}")
        
        # For storage: Power capacity constraints only (not energy)
        for s in S:
            if s in investment_required:
                s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
                P_max = safe_float(s_row['pmax'])  # Max power capacity
                
                # Power capacity constraints
                for t in T:
                    # Charge power <= Pmax * b
                    problem.linear_constraints.add(
                        lin_expr=[[[pch_vars[s, t], invest_vars[s]], [1.0, -P_max]]],
                        senses=["L"],
                        rhs=[0.0],
                        names=[f"cap_ch{s}t{t.strftime('%Y%m%d%H')}"]
                    )
                    
                    # Discharge power <= Pmax * b
                    problem.linear_constraints.add(
                        lin_expr=[[[pdis_vars[s, t], invest_vars[s]], [1.0, -P_max]]],
                        senses=["L"],
                        rhs=[0.0],
                        names=[f"cap_dis{s}t{t.strftime('%Y%m%d%H')}"]
                    )
                
                print(f"  - Added power capacity constraints for storage {s} with Pmax={P_max}")
    
    # 3. Solve the problem
    print("[DCOPF] About to solve the problem with CPLEX...")
    problem.solve()
    
    # Check solution status
    status = problem.solution.get_status()
    status_string = problem.solution.get_status_string()
    print(f"[DCOPF] Solver returned status code = {status}, interpreted as '{status_string}'")
    
    # Accept multiple status codes indicating optimality:
    # - 1: optimal (LP)
    # - 101: integer optimal solution (MIP)
    # - 102: optimal with rounding (MIP)
    if status != problem.solution.status.optimal and status != 101 and status != 102:
        print(f"[DCOPF] Not optimal => returning None.")
        return None
    
    # 4. Extract results
    
    # a) Extract objective value
    objective_value = problem.solution.get_objective_value()
    print(f"[DCOPF] Final cost = {objective_value}, status = {status_string}")
    
    # b) Extract generation (non-storage)
    generation = []
    for g in G:
        g_bus = gen_time_series.loc[gen_time_series['id'] == g, 'bus'].iloc[0]
        for t in T:
            val = problem.solution.get_values(gen_vars[g, t])
            generation.append({
                'time': t,
                'id': g,
                'node': g_bus,
                'gen': 0 if math.isnan(val) else val
            })
    
    # c) Extract storage net output
    storage_generation = []
    for s in S:
        s_bus = gen_time_series.loc[gen_time_series['id'] == s, 'bus'].iloc[0]
        for t in T:
            ch = problem.solution.get_values(pch_vars[s, t])
            dis = problem.solution.get_values(pdis_vars[s, t])
            if math.isnan(ch):
                ch = 0
            if math.isnan(dis):
                dis = 0
            net_out = dis - ch
            storage_generation.append({
                'time': t,
                'id': s,
                'node': s_bus,
                'gen': net_out
            })
    
    # Combine non-storage and storage generation
    generation_df = pd.DataFrame(generation)
    storage_generation_df = pd.DataFrame(storage_generation)
    
    if not generation_df.empty and not storage_generation_df.empty:
        generation_df = pd.concat([generation_df, storage_generation_df], ignore_index=True)
    elif not storage_generation_df.empty:
        generation_df = storage_generation_df
    
    # d) Extract angles
    angles = []
    for i_bus in N:
        for t in T:
            val_theta = problem.solution.get_values(theta_vars[i_bus, t])
            angles.append({
                'time': t,
                'bus': i_bus,
                'theta': 0 if math.isnan(val_theta) else val_theta
            })
    angles_df = pd.DataFrame(angles)
    
    # e) Extract flows
    flows = []
    for idx_b, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        for t in T:
            val_flow = problem.solution.get_values(flow_vars[i, j, t])
            flows.append({
                'time': t,
                'from_bus': i,
                'to_bus': j,
                'flow': 0 if math.isnan(val_flow) else val_flow
            })
    flows_df = pd.DataFrame(flows)
    
    # f) Extract storage states
    storage_list = []
    for s in S:
        for idx_t, tt in enumerate(extended_T):
            E_val = problem.solution.get_values(e_vars[s, tt])
            Pch = problem.solution.get_values(pch_vars[s, tt]) if tt in T else None
            Pdis = problem.solution.get_values(pdis_vars[s, tt]) if tt in T else None
            storage_list.append({
                'storage_id': s,
                'time': tt,
                'E': 0 if math.isnan(E_val) else E_val,
                'P_charge': None if (Pch is None or math.isnan(Pch)) else Pch,
                'P_discharge': None if (Pdis is None or math.isnan(Pdis)) else Pdis
            })
    
    # Always define columns so groupby('storage_id') won't fail
    storage_df = pd.DataFrame(
        storage_list,
        columns=["storage_id", "time", "E", "P_charge", "P_discharge"]
    )
    
    if len(S) > 0 and not storage_df.empty:
        # Shift E if you want SoC at start of each interval
        storage_corrected = []
        for s_id, group in storage_df.groupby('storage_id'):
            group = group.sort_values('time').reset_index(drop=True)
            group['E'] = group['E'].shift(-1)
            # remove last row
            group = group.iloc[:-1]
            storage_corrected.append(group)
        storage_df = pd.concat(storage_corrected, ignore_index=True)
    
    # g) Extract marginal prices from power balance constraints
    marginal_prices = []
    for t in T:
        for i in N:
            constraint_name = f"pb{i}t{t.strftime('%Y%m%d%H')}"
            try:
                constraint_idx = problem.linear_constraints.get_indices(constraint_name)
                dual_value = problem.solution.get_dual_values(constraint_idx)
                marginal_prices.append({
                    'time': t,
                    'bus': i,
                    'price': dual_value
                })
            except CplexError:
                print(f"[DCOPF] Warning: Could not get dual value for {constraint_name}")
    marginal_prices_df = pd.DataFrame(marginal_prices)
    
    # h) Extract congestion (shadow prices on flow limit constraints)
    congestion_shadow_prices = []
    for idx_b, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        for t in T:
            upper_constraint = f"up{i}{j}t{t.strftime('%Y%m%d%H')}"
            lower_constraint = f"lo{i}{j}t{t.strftime('%Y%m%d%H')}"
            
            try:
                upper_idx = problem.linear_constraints.get_indices(upper_constraint)
                upper_dual = problem.solution.get_dual_values(upper_idx)
            except CplexError:
                upper_dual = 0
                
            try:
                lower_idx = problem.linear_constraints.get_indices(lower_constraint)
                lower_dual = problem.solution.get_dual_values(lower_idx)
            except CplexError:
                lower_dual = 0
            
            congestion_shadow_prices.append({
                'time': t,
                'from_bus': i,
                'to_bus': j,
                'upper_limit_price': upper_dual,
                'lower_limit_price': lower_dual,
                'is_congested': abs(upper_dual) > 1e-6 or abs(lower_dual) > 1e-6
            })
    congestion_shadow_prices_df = pd.DataFrame(congestion_shadow_prices)
    
    # i) Extract investment decisions
    investment_decisions = {}
    investment_cost = 0
    operational_cost = objective_value
    
    if include_investment:
        for asset_id in investment_required:
            try:
                decision_value = problem.solution.get_values(invest_vars[asset_id])
                # Round to handle numerical issues (values very close to 0 or 1)
                decision = round(decision_value)
                investment_decisions[asset_id] = decision
                
                if decision == 1:
                    # Calculate annual capex
                    annual_capex = safe_float(asset_capex[asset_id]) / safe_float(asset_lifetimes[asset_id])
                    asset_investment_cost = annual_capex * planning_horizon
                    investment_cost += asset_investment_cost
                    
                    # Adjust operational cost
                    operational_cost -= asset_investment_cost
                    
                    print(f"[DCOPF] Asset {asset_id} selected for investment. Cost: ${asset_investment_cost:,.2f}")
                else:
                    print(f"[DCOPF] Asset {asset_id} NOT selected for investment")
            except CplexError:
                print(f"[DCOPF] Warning: Could not get investment decision for asset {asset_id}")
                investment_decisions[asset_id] = 0
        
        print(f"[DCOPF] Total investment cost: ${investment_cost:,.2f}")
        print(f"[DCOPF] Total operational cost: ${operational_cost:,.2f}")
    
    print("[DCOPF] Done, returning result dictionary.")
    
    result = {
        'generation': generation_df,
        'angles': angles_df,
        'flows': flows_df,
        'storage': storage_df,
        'cost': objective_value,
        'status': status_string,
        'marginal_prices': marginal_prices_df,
        'congestion': congestion_shadow_prices_df
    }
    
    if include_investment:
        result.update({
            'investment_decisions': investment_decisions,
            'investment_cost': investment_cost,
            'operational_cost': operational_cost
        })
    
    return result


def investment_dcopf(gen_time_series, branch, bus, demand_time_series, planning_horizon,
                   asset_lifetimes, asset_capex, delta_t=1, mip_gap=0.01, mip_time_limit=1800,
                   existing_investment=None):
    """
    Wrapper function for the DCOPF with investment decisions.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        planning_horizon: Planning horizon in years
        asset_lifetimes: Dictionary mapping asset IDs to their lifetimes in years
        asset_capex: Dictionary mapping asset IDs to their capital costs
        delta_t: Time step duration in hours
        mip_gap: MIP gap for the solver
        mip_time_limit: Time limit for the solver in seconds
        existing_investment: Dictionary mapping asset IDs to boolean values indicating if they are already installed
    
    Returns:
        Dictionary with DCOPF results including investment decisions
    """
    return dcopf(
        gen_time_series=gen_time_series,
        branch=branch,
        bus=bus,
        demand_time_series=demand_time_series,
        delta_t=delta_t,
        include_investment=True,
        planning_horizon=planning_horizon,
        asset_lifetimes=asset_lifetimes,
        asset_capex=asset_capex,
        existing_investment=existing_investment
    )

def investment_dcopf_planning(gen_time_series, branch, bus, demand_time_series, planning_horizon, asset_lifetimes, 
                            asset_capex, start_year=2023, delta_t=1, mip_gap=0.01, mip_time_limit=1800):
    """
    Solves the DC Optimal Power Flow problem with investment decisions across a planning horizon.
    
    This function extends the investment_dcopf function to create an installation plan across
    the planning horizon, analyzing whether assets should be installed or reinstalled based on
    economic considerations.
    
    Args:
        gen_time_series (pd.DataFrame): Generator data with time series information.
        branch (pd.DataFrame): Branch data.
        bus (pd.DataFrame): Bus data.
        demand_time_series (pd.DataFrame): Demand data with time series information.
        planning_horizon (int): Number of years in the planning horizon.
        asset_lifetimes (dict): Dictionary mapping asset IDs to their lifetimes in years.
        asset_capex (dict): Dictionary mapping asset IDs to their capital expenditure costs.
        start_year (int, optional): Starting year for the planning horizon. Defaults to 2023.
        delta_t (float, optional): Time step in hours. Defaults to 1 (1 hour).
        mip_gap (float, optional): MIP gap for the solver. Defaults to 0.01 (1%).
        mip_time_limit (int, optional): Time limit for the solver in seconds. Defaults to 1800 (30 minutes).
    
    Returns:
        dict: Dictionary containing the results of the optimization, including:
            - installation_timeline: Dictionary mapping asset IDs to dictionaries of {year: action},
              where action is 'Install', 'Reinstall', or 'Retire'.
            - active_assets_by_year: Dictionary mapping years to lists of active asset IDs.
            - investment_required: List of asset IDs that require investment decisions.
            - planning_horizon: Number of years in the planning horizon.
            - start_year: Starting year for the planning horizon.
            - asset_lifetimes: Dictionary mapping asset IDs to their lifetimes in years.
    """
    import cplex
    import pandas as pd
    import numpy as np
    from cplex.exceptions import CplexError
    
    print("Solving DC Optimal Power Flow with Investment Planning...")
    
    # Identify which generators require investment decisions
    investment_required = []
    for asset_id in gen_time_series['id'].unique():
        asset_data = gen_time_series[gen_time_series['id'] == asset_id].iloc[0]
        if asset_data['investment_required'] == 1:
            investment_required.append(asset_id)
            capex = asset_capex.get(asset_id, 0)
            print(f"Asset {asset_id} requires investment decision, CAPEX: ${capex:,.2f}")
    
    # Dictionary to track installation actions for each asset across the planning horizon
    installation_timeline = {asset_id: {} for asset_id in investment_required}
    
    # Dictionary to track active assets by year
    active_assets_by_year = {}
    
    # Track installed assets from previous years
    installed_assets = []
    
    # For each year in the planning horizon, run the investment model
    for year in range(start_year, start_year + planning_horizon):
        print(f"\n===== Year {year} =====")
        
        # Identify which assets are due for retirement this year
        assets_to_retire = []
        for asset_id in installed_assets:
            # Find when this asset was last installed
            last_installation = max([y for y, action in installation_timeline[asset_id].items() 
                                    if action in ['Install', 'Reinstall']], default=None)
            
            if last_installation is not None:
                # Check if asset reaches end of life this year
                if year == last_installation + asset_lifetimes[asset_id]:
                    assets_to_retire.append(asset_id)
                    installation_timeline[asset_id][year] = 'Retire'
                    print(f"Asset {asset_id} reaches end of life in {year}")
        
        # Remove retired assets from installed assets
        for asset_id in assets_to_retire:
            installed_assets.remove(asset_id)
        
        # Mark existing assets as already installed
        existing_investment = {}
        for asset_id in investment_required:
            # If the asset is already installed, it doesn't need to be installed again
            existing_investment[asset_id] = asset_id in installed_assets
        
        # Run the investment model for this year
        investment_results = investment_dcopf(
            gen_time_series=gen_time_series,
            branch=branch,
            bus=bus,
            demand_time_series=demand_time_series,
            planning_horizon=planning_horizon,
            asset_lifetimes=asset_lifetimes,
            asset_capex=asset_capex,
            existing_investment=existing_investment,
            delta_t=delta_t,
            mip_gap=mip_gap,
            mip_time_limit=mip_time_limit
        )
        
        if investment_results is None:
            print(f"Investment optimization failed for year {year}")
            continue
        
        # Extract investment decisions
        investment_decisions = investment_results['investment_decisions']
        
        # Update installation timeline and track newly installed assets
        for asset_id, decision in investment_decisions.items():
            if decision and asset_id not in installed_assets:
                # Check if this is a new installation or reinstallation
                if any(action in ['Install', 'Reinstall'] for action in installation_timeline[asset_id].values()):
                    installation_timeline[asset_id][year] = 'Reinstall'
                    print(f"Asset {asset_id} reinstalled in {year}")
                else:
                    installation_timeline[asset_id][year] = 'Install'
                    print(f"Asset {asset_id} installed in {year}")
                
                installed_assets.append(asset_id)
        
        # Record active assets for this year
        active_assets_by_year[year] = installed_assets.copy()
        print(f"Active assets in {year}: {active_assets_by_year[year]}")
    
    # Prepare the results
    planning_results = {
        'installation_timeline': installation_timeline,
        'active_assets_by_year': active_assets_by_year,
        'investment_required': investment_required,
        'planning_horizon': planning_horizon,
        'start_year': start_year,
        'asset_lifetimes': asset_lifetimes
    }
    
    # Print a summary of the installation plan
    print("\n===== Installation Plan Summary =====")
    for asset_id in investment_required:
        installations = [year for year, action in installation_timeline[asset_id].items() 
                        if action in ['Install', 'Reinstall']]
        retirements = [year for year, action in installation_timeline[asset_id].items() 
                      if action == 'Retire']
        
        if installations:
            print(f"Asset {asset_id}: Installed/Reinstalled in years {installations}, Retired in years {retirements}")
        else:
            print(f"Asset {asset_id}: Never installed")
    
    return planning_results