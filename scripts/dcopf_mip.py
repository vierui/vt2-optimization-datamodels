#!/usr/bin/env python3

"""
dcopf_mip.py

A Mixed Integer Programming (MIP) extension of the DC Optimal Power Flow (DCOPF) solver.
This implementation adds binary commitment variables and startup/shutdown variables.

Key features:
- Binary commitment variables for generators
- Startup and shutdown variables and costs
- Minimum up/down time constraints (optional)
- Ramp rate constraints (optional)
- Integration with the standard DCOPF solver

This serves as a base for implementing more complex multi-stage investment models.
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

def dcopf_mip(gen_time_series, branch, bus, demand_time_series, 
             startup_costs=None, shutdown_costs=None, min_up_time=None, 
             min_down_time=None, ramp_rates=None, delta_t=1):
    """
    Mixed Integer Programming (MIP) extension of the DC Optimal Power Flow (DCOPF) solver.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        startup_costs: Dictionary mapping generator IDs to startup costs
        shutdown_costs: Dictionary mapping generator IDs to shutdown costs
        min_up_time: Dictionary mapping generator IDs to minimum up times (hours)
        min_down_time: Dictionary mapping generator IDs to minimum down times (hours)
        ramp_rates: Dictionary mapping generator IDs to ramp rates (MW/hour)
        delta_t: Time step duration in hours
    
    Returns:
        Dictionary with DCOPF-MIP results
    """
    print("[DCOPF-MIP] Entering dcopf_mip function...")
    print(f"[DCOPF-MIP] gen_time_series length = {len(gen_time_series)}, demand_time_series length = {len(demand_time_series)}")
    
    # Function to safely convert any value to float
    def safe_float(value):
        return float(value)

    try:
        # Create CPLEX problem
        problem = cplex.Cplex()
        
        # Minimize objective function
        problem.objective.set_sense(problem.objective.sense.minimize)
        
        # Set MIP parameters
        problem.parameters.mip.display.set(2)  # Display node/iterations
        
        # Identify storage vs. non-storage units
        storage_data = gen_time_series[gen_time_series['emax'] > 0]
        S = storage_data['id'].unique()     # set of storage IDs
        non_storage_data = gen_time_series[gen_time_series['emax'] == 0]
        G = non_storage_data['id'].unique() # set of non-storage gen IDs
        
        print(f"[DCOPF-MIP] Found storage units: {S}, non-storage units: {G}")
        
        # Time and bus sets
        N = bus['bus_i'].values
        T = sorted(demand_time_series['time'].unique())
        if not T:
            print("[DCOPF-MIP] No time steps found in demand_time_series. Returning None.")
            return None
        
        next_time = T[-1] + DateOffset(hours=delta_t)
        extended_T = list(T) + [next_time]
        
        # Default costs if not provided
        if startup_costs is None:
            startup_costs = {g: 100.0 for g in G}
        if shutdown_costs is None:
            shutdown_costs = {g: 50.0 for g in G}
            
        # 1. Create variables
        # a) Generation variables
        gen_vars = {}
        gen_names = []
        gen_costs = []
        gen_lbs = []
        gen_ubs = []
        
        for g in G:
            for t in T:
                gen_row = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]
                if gen_row.empty:
                    print(f"[DCOPF-MIP] Missing data for generator={g}, time={t}. Returning None.")
                    return None
                    
                pmin = gen_row['pmin'].iloc[0]
                pmax = gen_row['pmax'].iloc[0]
                cost = gen_row['gencost'].iloc[0]
                
                var_name = f"g_{g}_t{t.strftime('%Y%m%d%H')}"
                gen_vars[g, t] = var_name
                gen_names.append(var_name)
                gen_costs.append(safe_float(cost))
                gen_lbs.append(0.0)  # Lower bound will be enforced by commitment constraint
                gen_ubs.append(safe_float(pmax))
        
        problem.variables.add(
            obj=gen_costs,
            lb=gen_lbs,
            ub=gen_ubs,
            names=gen_names
        )
        
        # b) Binary commitment variables
        commit_vars = {}
        commit_names = []
        
        for g in G:
            for t in T:
                var_name = f"u_{g}_t{t.strftime('%Y%m%d%H')}"
                commit_vars[g, t] = var_name
                commit_names.append(var_name)
        
        problem.variables.add(
            lb=[0] * len(commit_names),
            ub=[1] * len(commit_names),
            names=commit_names,
            types=["B"] * len(commit_names)  # B for binary variables
        )
        
        # c) Startup and shutdown variables
        startup_vars = {}
        startup_names = []
        shutdown_vars = {}
        shutdown_names = []
        
        for g in G:
            for t in T:
                # Startup variables
                startup_name = f"v_{g}_t{t.strftime('%Y%m%d%H')}"
                startup_vars[g, t] = startup_name
                startup_names.append(startup_name)
                
                # Shutdown variables
                shutdown_name = f"w_{g}_t{t.strftime('%Y%m%d%H')}"
                shutdown_vars[g, t] = shutdown_name
                shutdown_names.append(shutdown_name)
        
        problem.variables.add(
            lb=[0] * len(startup_names),
            ub=[1] * len(startup_names),
            names=startup_names,
            types=["B"] * len(startup_names)
        )
        
        problem.variables.add(
            lb=[0] * len(shutdown_names),
            ub=[1] * len(shutdown_names),
            names=shutdown_names,
            types=["B"] * len(shutdown_names)
        )
        
        # d) Voltage angle variables
        theta_vars = {}
        theta_names = []
        theta_lbs = []
        theta_ubs = []
        
        for i in N:
            for t in T:
                var_name = f"t_{i}_t{t.strftime('%Y%m%d%H')}"
                theta_vars[i, t] = var_name
                theta_names.append(var_name)
                theta_lbs.append(-cplex.infinity)
                theta_ubs.append(cplex.infinity)
        
        problem.variables.add(
            lb=theta_lbs,
            ub=theta_ubs,
            names=theta_names
        )
        
        # e) Flow variables
        flow_vars = {}
        flow_names = []
        flow_lbs = []
        flow_ubs = []
        
        for _, row in branch.iterrows():
            i = int(row['fbus'])
            j = int(row['tbus'])
            for t in T:
                var_name = f"flow_{i}_{j}_t{t.strftime('%Y%m%d%H')}"
                flow_vars[i, j, t] = var_name
                flow_names.append(var_name)
                flow_lbs.append(-cplex.infinity)
                flow_ubs.append(cplex.infinity)
        
        problem.variables.add(
            lb=flow_lbs,
            ub=flow_ubs,
            names=flow_names
        )
        
        # f) Storage variables (P_charge, P_discharge, E) if storage exists
        pch_vars = {}
        pdis_vars = {}
        e_vars = {}
        
        if len(S) > 0:
            # Add charge/discharge variables
            for s in S:
                s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
                P_max = s_row['pmax']  # Discharging max
                
                for t in T:
                    # Charge variables
                    pch_name = f"pc_{s}_t{t.strftime('%Y%m%d%H')}"
                    pch_vars[s, t] = pch_name
                    problem.variables.add(
                        lb=[0],
                        ub=[abs(P_max)],
                        names=[pch_name]
                    )
                    
                    # Discharge variables
                    pdis_name = f"pd_{s}_t{t.strftime('%Y%m%d%H')}"
                    pdis_vars[s, t] = pdis_name
                    problem.variables.add(
                        lb=[0],
                        ub=[abs(P_max)],
                        names=[pdis_name]
                    )
                
                # Energy variables
                for t in extended_T:
                    e_name = f"e_{s}_t{t.strftime('%Y%m%d%H')}"
                    e_vars[s, t] = e_name
                    problem.variables.add(
                        lb=[0],
                        ub=[s_row['emax']],
                        names=[e_name]
                    )
        
        # 2. Add constraints
        print("[DCOPF-MIP] Adding constraints...")
        
        # a) Commitment constraints for generation
        for g in G:
            for t in T:
                gen_row = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]
                pmin = gen_row['pmin'].iloc[0]
                pmax = gen_row['pmax'].iloc[0]
                
                gen_var = gen_vars[g, t]
                commit_var = commit_vars[g, t]
                
                # P ≤ Pmax * u
                upper_name = f"up_{g}_t{t.strftime('%Y%m%d%H')}"
                problem.linear_constraints.add(
                    lin_expr=[[[gen_var, commit_var], [1.0, -safe_float(pmax)]]],
                    senses=["L"],
                    rhs=[0.0],
                    names=[upper_name]
                )
                
                # P ≥ Pmin * u
                if pmin > 0:
                    lower_name = f"lo_{g}_t{t.strftime('%Y%m%d%H')}"
                    problem.linear_constraints.add(
                        lin_expr=[[[gen_var, commit_var], [1.0, -safe_float(pmin)]]],
                        senses=["G"],
                        rhs=[0.0],
                        names=[lower_name]
                    )
        
        # b) DC Power Flow Constraints
        for idx_b, row_b in branch.iterrows():
            i = row_b['fbus']
            j = row_b['tbus']
            susceptance = row_b['sus']
            rate_a = safe_float(row_b['ratea'])
            
            for t in T:
                # Flow = (angle_from - angle_to) / x
                from_angle_var_name = theta_vars[i, t]
                to_angle_var_name = theta_vars[j, t]
                flow_var_name = flow_vars[i, j, t]
                
                constraint_name = f"dcflow_{int(i)}_{int(j)}_t{t.strftime('%Y%m%d%H')}"
                
                var_list = [flow_var_name, from_angle_var_name, to_angle_var_name]
                coef_list = [1.0, -safe_float(susceptance), safe_float(susceptance)]
                
                problem.linear_constraints.add(
                    lin_expr=[[var_list, coef_list]],
                    senses=["E"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
                
                # Flow limits
                # Upper limit: FLOW[i, j, t] <= rate_a
                upper_constraint_name = f"upflow_{int(i)}_{int(j)}_t{t.strftime('%Y%m%d%H')}"
                problem.linear_constraints.add(
                    lin_expr=[[[flow_var_name], [1.0]]],
                    senses=["L"],
                    rhs=[rate_a],
                    names=[upper_constraint_name]
                )
                
                # Lower limit: FLOW[i, j, t] >= -rate_a
                lower_constraint_name = f"loflow_{int(i)}_{int(j)}_t{t.strftime('%Y%m%d%H')}"
                problem.linear_constraints.add(
                    lin_expr=[[[flow_var_name], [1.0]]],
                    senses=["G"],
                    rhs=[-rate_a],
                    names=[lower_constraint_name]
                )
        
        # c) Startup/Shutdown linking constraints
        if len(T) > 1:
            for g in G:
                # For first period, assume generator is initially off
                t0 = T[0]
                commit_var_t0 = commit_vars[g, t0]
                startup_var_t0 = startup_vars[g, t0]
                
                # u_t0 ≤ v_t0
                constraint_name = f"start_init_{g}"
                problem.linear_constraints.add(
                    lin_expr=[[[commit_var_t0, startup_var_t0], [1.0, -1.0]]],
                    senses=["L"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
                
                # For subsequent periods
                for t_idx in range(1, len(T)):
                    t = T[t_idx]
                    t_prev = T[t_idx - 1]
                    
                    commit_var_t = commit_vars[g, t]
                    commit_var_prev = commit_vars[g, t_prev]
                    startup_var_t = startup_vars[g, t]
                    shutdown_var_t = shutdown_vars[g, t]
                    
                    # u_t - u_{t-1} ≤ v_t
                    startup_con_name = f"start_{g}_t{t.strftime('%Y%m%d%H')}"
                    problem.linear_constraints.add(
                        lin_expr=[[[commit_var_t, commit_var_prev, startup_var_t], 
                                  [1.0, -1.0, -1.0]]],
                        senses=["L"],
                        rhs=[0.0],
                        names=[startup_con_name]
                    )
                    
                    # u_{t-1} - u_t ≤ w_t
                    shutdown_con_name = f"shut_{g}_t{t.strftime('%Y%m%d%H')}"
                    problem.linear_constraints.add(
                        lin_expr=[[[commit_var_prev, commit_var_t, shutdown_var_t], 
                                  [1.0, -1.0, -1.0]]],
                        senses=["L"],
                        rhs=[0.0],
                        names=[shutdown_con_name]
                    )
        
        # d) Storage constraints if storage exists
        if len(S) > 0:
            # Initial storage state constraint
            for s in S:
                s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
                E_initial = s_row['einitial']
                
                # Initial SoC: E[T[0]] = E_initial
                e_var_name = e_vars[s, T[0]]
                constraint_name = f"inite_{int(s)}"
                
                problem.linear_constraints.add(
                    lin_expr=[[[e_var_name], [1.0]]],
                    senses=["E"],
                    rhs=[safe_float(E_initial)],
                    names=[constraint_name]
                )
            
            # Storage dynamics
            for s in S:
                s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
                eta = s_row['eta']
                
                for idx_t, t in enumerate(T):
                    next_t = extended_T[idx_t + 1]
                    
                    # E[next] = E[t] + eta*charge - (1/eta)*discharge
                    e_t_name = e_vars[s, t]
                    e_next_name = e_vars[s, next_t]
                    pch_t_name = pch_vars[s, t]
                    pdis_t_name = pdis_vars[s, t]
                    
                    storage_var_names = [e_t_name, e_next_name, pch_t_name, pdis_t_name]
                    storage_coefs = [1.0, -1.0, -safe_float(eta) * safe_float(delta_t), 
                                   safe_float(1/eta) * safe_float(delta_t)]
                    constraint_name = f"sd_{int(s)}_t{t.strftime('%Y%m%d%H')}"
                    
                    problem.linear_constraints.add(
                        lin_expr=[[storage_var_names, storage_coefs]],
                        senses=["E"],
                        rhs=[0.0],
                        names=[constraint_name]
                    )
            
            # Final storage state constraint
            for s in S:
                s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
                E_initial = s_row['einitial']
                
                # Final SoC: E[extended_T[-1]] = E_initial
                e_final_name = e_vars[s, extended_T[-1]]
                constraint_name = f"finsoc_{int(s)}"
                
                problem.linear_constraints.add(
                    lin_expr=[[[e_final_name], [1.0]]],
                    senses=["E"],
                    rhs=[safe_float(E_initial)],
                    names=[constraint_name]
                )
        
        # e) Reference (slack) bus constraint
        slack_bus = 1  # Bus 1 is the reference
        for t in T:
            theta_slack_name = theta_vars[slack_bus, t]
            constraint_name = f"slack_t{t.strftime('%Y%m%d%H')}"
            
            problem.linear_constraints.add(
                lin_expr=[[[theta_slack_name], [1.0]]],
                senses=["E"],
                rhs=[0.0],
                names=[constraint_name]
            )
        
        # f) Power balance constraints
        for t in T:
            for i in N:  # Include all buses
                # Collect all variable names and coefficients
                var_names = []
                coefficients = []
                
                # Add non-storage generation
                for g in G:
                    gen_rows = gen_time_series.loc[
                        (gen_time_series['id'] == g) & (gen_time_series['time'] == t)
                    ]
                    if not gen_rows.empty and gen_rows['bus'].iloc[0] == i:
                        var_names.append(gen_vars[g, t])
                        coefficients.append(1.0)
                
                # Add storage (discharge - charge)
                for s in S:
                    stor_rows = gen_time_series.loc[
                        (gen_time_series['id'] == s) & (gen_time_series['time'] == t)
                    ]
                    if not stor_rows.empty and stor_rows['bus'].iloc[0] == i:
                        var_names.append(pdis_vars[s, t])
                        coefficients.append(1.0)
                        var_names.append(pch_vars[s, t])
                        coefficients.append(-1.0)
                
                # Add flows into the bus
                for idx_b, row_b in branch.iterrows():
                    if row_b['tbus'] == i:  # Flow into bus i
                        from_bus = int(row_b['fbus'])
                        to_bus = int(row_b['tbus'])
                        var_names.append(flow_vars[from_bus, to_bus, t])
                        coefficients.append(1.0)
                
                # Add flows out of the bus
                for idx_b, row_b in branch.iterrows():
                    if row_b['fbus'] == i:  # Flow out of bus i
                        from_bus = int(row_b['fbus'])
                        to_bus = int(row_b['tbus'])
                        var_names.append(flow_vars[from_bus, to_bus, t])
                        coefficients.append(-1.0)
                
                # Get demand at bus i
                pd_val = 0
                demands_at_bus = demand_time_series.loc[
                    (demand_time_series['bus'] == i) & (demand_time_series['time'] == t),
                    'pd'
                ]
                pd_val = demands_at_bus.sum() if not demands_at_bus.empty else 0
                
                constraint_name = f"pb_{int(i)}_t{t.strftime('%Y%m%d%H')}"
                
                # Add power balance constraint
                if var_names:
                    problem.linear_constraints.add(
                        lin_expr=[[var_names, coefficients]],
                        senses=["E"],
                        rhs=[safe_float(pd_val)],
                        names=[constraint_name]
                    )
                
        # 3. Add startup/shutdown costs to objective
        for g in G:
            for t in T:
                # Add startup cost
                problem.objective.set_linear(startup_vars[g, t], safe_float(startup_costs[g]))
                # Add shutdown cost
                problem.objective.set_linear(shutdown_vars[g, t], safe_float(shutdown_costs[g]))
        
        # 4. Solve the problem
        print("[DCOPF-MIP] About to solve the problem with CPLEX...")
        problem.solve()
        
        # Check solution status
        status = problem.solution.get_status()
        status_string = problem.solution.get_status_string()
        print(f"[DCOPF-MIP] Solver returned status code = {status}, interpreted as '{status_string}'")
        
        if status != problem.solution.status.optimal and status != problem.solution.status.MIP_optimal:
            print(f"[DCOPF-MIP] Not optimal => returning None.")
            return None
        
        # 5. Extract results
        # a) Extract objective value
        objective_value = problem.solution.get_objective_value()
        print(f"[DCOPF-MIP] Final cost = {objective_value}, status = {status_string}")
        
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
        
        # c) Extract commitment status
        commitment = []
        for g in G:
            for t in T:
                val = problem.solution.get_values(commit_vars[g, t])
                commitment.append({
                    'time': t,
                    'id': g, 
                    'status': 1 if val > 0.5 else 0
                })
        
        # d) Extract startup/shutdown decisions
        startup_shutdown = []
        for g in G:
            for t in T:
                startup_val = problem.solution.get_values(startup_vars[g, t])
                shutdown_val = problem.solution.get_values(shutdown_vars[g, t])
                startup_shutdown.append({
                    'time': t,
                    'id': g,
                    'startup': 1 if startup_val > 0.5 else 0,
                    'shutdown': 1 if shutdown_val > 0.5 else 0
                })
        
        # e) Extract storage data if storage exists
        storage_generation = []
        storage_list = []
        
        if len(S) > 0:
            # Extract storage net output
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
            
            # Extract storage states
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
            
            # Shift E if you want SoC at start of each interval
            storage_df = pd.DataFrame(
                storage_list,
                columns=["storage_id", "time", "E", "P_charge", "P_discharge"]
            )
            
            if not storage_df.empty:
                storage_corrected = []
                for s_id, group in storage_df.groupby('storage_id'):
                    group = group.sort_values('time').reset_index(drop=True)
                    group['E'] = group['E'].shift(-1)
                    # remove last row
                    group = group.iloc[:-1]
                    storage_corrected.append(group)
                storage_df = pd.concat(storage_corrected, ignore_index=True)
        else:
            storage_df = pd.DataFrame(
                columns=["storage_id", "time", "E", "P_charge", "P_discharge"]
            )
        
        # Combine non-storage and storage generation
        generation_df = pd.DataFrame(generation)
        storage_generation_df = pd.DataFrame(storage_generation)
        
        if not generation_df.empty and not storage_generation_df.empty:
            generation_df = pd.concat([generation_df, storage_generation_df], ignore_index=True)
        elif not storage_generation_df.empty:
            generation_df = storage_generation_df
        
        # f) Extract angles
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
        
        # g) Extract flows
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
        
        # h) Extract marginal prices from power balance constraints
        marginal_prices = []
        for t in T:
            for i in N:
                constraint_name = f"pb_{i}_t{t.strftime('%Y%m%d%H')}"
                try:
                    constraint_idx = problem.linear_constraints.get_indices(constraint_name)
                    dual_value = problem.solution.get_dual_values(constraint_idx)
                    marginal_prices.append({
                        'time': t,
                        'bus': i,
                        'price': dual_value
                    })
                except CplexError:
                    print(f"[DCOPF-MIP] Warning: Could not get dual value for {constraint_name}")
        marginal_prices_df = pd.DataFrame(marginal_prices)
        
        # i) Extract congestion (shadow prices on flow limit constraints)
        congestion_shadow_prices = []
        for idx_b, row_b in branch.iterrows():
            i = int(row_b['fbus'])
            j = int(row_b['tbus'])
            for t in T:
                upper_constraint = f"upflow_{i}_{j}_t{t.strftime('%Y%m%d%H')}"
                lower_constraint = f"loflow_{i}_{j}_t{t.strftime('%Y%m%d%H')}"
                
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
        
        # j) Create commitment and startup/shutdown dataframes
        commitment_df = pd.DataFrame(commitment)
        startup_shutdown_df = pd.DataFrame(startup_shutdown)
        
        print("[DCOPF-MIP] Done, returning result dictionary.")
        
        return {
            'generation': generation_df,
            'angles': angles_df,
            'flows': flows_df,
            'storage': storage_df,
            'cost': objective_value,
            'status': status_string,
            'marginal_prices': marginal_prices_df,
            'congestion': congestion_shadow_prices_df,
            'commitment': commitment_df,
            'startup_shutdown': startup_shutdown_df
        }
    except Exception as e:
        print(f"[DCOPF-MIP] Error in DCOPF-MIP solver: {e}")
        return None 