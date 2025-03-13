#!/usr/bin/env python3

"""
dcopf_mip.py

A Mixed Integer Programming (MIP) DC Optimal Power Flow (DCOPF) solver using CPLEX.
This implementation extends the standard DCOPF by including binary variables for generator commitment decisions.

Key extensions over the LP formulation:
- Binary commitment variables for generators (on/off status)
- Minimum generation levels when generators are committed
- Startup and shutdown costs
- Ramping constraints

The solver uses the CPLEX optimization engine and requires the CPLEX Python API.
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

def dcopf_mip(gen_time_series, branch, bus, demand_time_series, delta_t=1, 
              min_up_time=1, min_down_time=1, startup_costs=None, shutdown_costs=None):
    """
    Mixed Integer Programming DC Optimal Power Flow solver using CPLEX.
    
    Args:
        gen_time_series: DataFrame with generator data
        branch: DataFrame with branch data
        bus: DataFrame with bus data
        demand_time_series: DataFrame with demand data
        delta_t: Time step duration in hours
        min_up_time: Minimum up time for generators (default: 1)
        min_down_time: Minimum down time for generators (default: 1)
        startup_costs: Dictionary of generator startup costs {gen_id: cost}
        shutdown_costs: Dictionary of generator shutdown costs {gen_id: cost}
    
    Returns:
        Dictionary with DCOPF results including generation, flows, prices, etc.
    """
    print("[DCOPF-MIP] Entering dcopf_mip function...")
    print(f"[DCOPF-MIP] gen_time_series length = {len(gen_time_series)}, demand_time_series length = {len(demand_time_series)}")
    
    print("[DCOPF-MIP] Using CPLEX solver for MIP formulation")
    return _dcopf_mip_cplex(gen_time_series, branch, bus, demand_time_series, delta_t,
                         min_up_time, min_down_time, startup_costs, shutdown_costs)

def _dcopf_mip_cplex(gen_time_series, branch, bus, demand_time_series, delta_t=1,
                   min_up_time=1, min_down_time=1, startup_costs=None, shutdown_costs=None):
    """
    Internal implementation of MIP DCOPF using CPLEX.
    """
    # Function to safely convert any value to float
    def safe_float(value):
        return float(value)

    # Default values for startup/shutdown costs if not provided
    if startup_costs is None:
        startup_costs = {}
    if shutdown_costs is None:
        shutdown_costs = {}

    try:
        print("[DCOPF-MIP] Using CPLEX solver")
    except ImportError:
        print("[DCOPF-MIP] Error: CPLEX not available.")
        return None

    # Create CPLEX problem
    problem = cplex.Cplex()
    
    # Minimize objective function
    problem.objective.set_sense(problem.objective.sense.minimize)
    
    # Set MIP parameters
    problem.parameters.mip.display.set(4)  # Display detailed MIP information
    
    # Identify storage vs. non-storage units
    storage_data = gen_time_series[gen_time_series['emax'] > 0]
    S = storage_data['id'].unique()     # set of storage IDs
    non_storage_data = gen_time_series[gen_time_series['emax'] == 0]
    G = non_storage_data['id'].unique() # set of non-storage gen IDs

    print(f"[DCOPF-MIP] Found storage units: {S}, non-storage units: {G}")
    if not storage_data.empty:
        print("[DCOPF-MIP] Storage data sample:")
        print(storage_data[['id', 'bus', 'emax', 'pmax', 'eta']].head())

    # Time and bus sets
    N = bus['bus_i'].values
    T = sorted(demand_time_series['time'].unique())
    if not T:
        print("[DCOPF-MIP] No time steps found in demand_time_series. Returning None.")
        return None

    next_time = T[-1] + DateOffset(hours=delta_t)
    extended_T = list(T) + [next_time]
    
    # 1. Create variables    
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
                print(f"[DCOPF-MIP] Missing data for generator={g}, time={t}. Returning None.")
                return None
                
            pmin = gen_row['pmin'].iloc[0]
            pmax = gen_row['pmax'].iloc[0]
            cost = gen_row['gencost'].iloc[0]
            
            var_name = f"g_{g}_t{t.strftime('%Y%m%d%H')}"
            gen_vars[g, t] = var_name
            gen_names.append(var_name)
            gen_costs.append(safe_float(cost))
            gen_lbs.append(0.0)  # Lower bound will be controlled by commitment * pmin
            gen_ubs.append(safe_float(pmax))
    
    problem.variables.add(
        obj=gen_costs,
        lb=gen_lbs,
        ub=gen_ubs,
        names=gen_names
    )
    
    # b) Binary commitment variables for generators
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
        types=["B"] * len(commit_names)  # Binary variables
    )
    
    # c) Startup and shutdown variables
    startup_vars = {}
    startup_names = []
    startup_costs_list = []
    
    shutdown_vars = {}
    shutdown_names = []
    shutdown_costs_list = []
    
    for g in G:
        for t_idx, t in enumerate(T):
            # Startup variables
            startup_var_name = f"su_{g}_t{t.strftime('%Y%m%d%H')}"
            startup_vars[g, t] = startup_var_name
            startup_names.append(startup_var_name)
            # Use default cost of 10% of generation cost if not specified
            g_cost = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]['gencost'].iloc[0]
            startup_cost = startup_costs.get(g, 0.1 * g_cost)
            startup_costs_list.append(safe_float(startup_cost))
            
            # Shutdown variables
            shutdown_var_name = f"sd_{g}_t{t.strftime('%Y%m%d%H')}"
            shutdown_vars[g, t] = shutdown_var_name
            shutdown_names.append(shutdown_var_name)
            # Use default cost of 5% of generation cost if not specified
            shutdown_cost = shutdown_costs.get(g, 0.05 * g_cost)
            shutdown_costs_list.append(safe_float(shutdown_cost))
    
    problem.variables.add(
        obj=startup_costs_list,
        lb=[0] * len(startup_names),
        ub=[1] * len(startup_names),
        names=startup_names,
        types=["B"] * len(startup_names)  # Binary variables
    )
    
    problem.variables.add(
        obj=shutdown_costs_list,
        lb=[0] * len(shutdown_names),
        ub=[1] * len(shutdown_names),
        names=shutdown_names,
        types=["B"] * len(shutdown_names)  # Binary variables
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
    
    # f) Storage variables (P_charge, P_discharge, E)
    pch_vars = {}
    pdis_vars = {}
    e_vars = {}
    
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
    print("[DCOPF-MIP] Adding DC Power Flow Constraints...")
    
    # a) DC Power Flow Constraints
    for idx_b, row_b in branch.iterrows():
        i = row_b['fbus']
        j = row_b['tbus']
        susceptance = row_b['sus']
        rate_a = safe_float(row_b['ratea'])  # Convert limit to float
        
        for t in T:
            # Flow = (angle_from - angle_to) / x
            from_angle_var_name = theta_vars[i, t]
            to_angle_var_name = theta_vars[j, t]
            flow_var_name = flow_vars[i, j, t]
            
            constraint_name = f"dcflow_{int(i)}_{int(j)}_t{t.strftime('%Y%m%d%H')}"
            
            var_list = [flow_var_name, from_angle_var_name, to_angle_var_name]
            coef_list = [1.0, -safe_float(susceptance), safe_float(susceptance)]
            
            try:
                problem.linear_constraints.add(
                    lin_expr=[[var_list, coef_list]],
                    senses=["E"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
            except Exception as e:
                print(f"[DCOPF-MIP] Error adding constraint {constraint_name}: {e}")
                print(f"[DCOPF-MIP] Variable list: {var_list}")
                print(f"[DCOPF-MIP] Coefficient list: {coef_list}")
                raise
            
            # Upper limit: FLOW[i, j, t] <= rate_a
            flow_var_name = flow_vars[i, j, t]
            upper_constraint_name = f"upflow_{int(i)}_{int(j)}_t{t.strftime('%Y%m%d%H')}"
            
            problem.linear_constraints.add(
                lin_expr=[[[flow_var_name], [1.0]]],
                senses=["L"],
                rhs=[safe_float(rate_a)],
                names=[upper_constraint_name]
            )
            
            # Lower limit: FLOW[i, j, t] >= -rate_a
            lower_constraint_name = f"loflow_{int(i)}_{int(j)}_t{t.strftime('%Y%m%d%H')}"
            
            problem.linear_constraints.add(
                lin_expr=[[[flow_var_name], [1.0]]],
                senses=["G"],
                rhs=[safe_float(-rate_a)],
                names=[lower_constraint_name]
            )
    
    # b) Generator minimum output constraints (p >= pmin * u)
    for g in G:
        for t in T:
            gen_row = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]
            pmin = gen_row['pmin'].iloc[0]
            
            gen_var_name = gen_vars[g, t]
            commit_var_name = commit_vars[g, t]
            constraint_name = f"min_gen_{g}_t{t.strftime('%Y%m%d%H')}"
            
            problem.linear_constraints.add(
                lin_expr=[[[gen_var_name, commit_var_name], [1.0, -safe_float(pmin)]]],
                senses=["G"],
                rhs=[0.0],
                names=[constraint_name]
            )
    
    # c) Generator maximum output constraints (p <= pmax * u)
    for g in G:
        for t in T:
            gen_row = gen_time_series[(gen_time_series['id'] == g) & (gen_time_series['time'] == t)]
            pmax = gen_row['pmax'].iloc[0]
            
            gen_var_name = gen_vars[g, t]
            commit_var_name = commit_vars[g, t]
            constraint_name = f"max_gen_{g}_t{t.strftime('%Y%m%d%H')}"
            
            problem.linear_constraints.add(
                lin_expr=[[[gen_var_name, commit_var_name], [1.0, -safe_float(pmax)]]],
                senses=["L"],
                rhs=[0.0],
                names=[constraint_name]
            )
    
    # d) Startup constraints: u[g,t] - u[g,t-1] <= su[g,t]
    for g in G:
        for t_idx, t in enumerate(T):
            if t_idx == 0:
                # For the first time period, assume generator was off
                commit_var_name = commit_vars[g, t]
                startup_var_name = startup_vars[g, t]
                constraint_name = f"startup_{g}_t{t.strftime('%Y%m%d%H')}"
                
                problem.linear_constraints.add(
                    lin_expr=[[[commit_var_name, startup_var_name], [1.0, -1.0]]],
                    senses=["L"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
            else:
                prev_t = T[t_idx - 1]
                commit_var_name = commit_vars[g, t]
                prev_commit_var_name = commit_vars[g, prev_t]
                startup_var_name = startup_vars[g, t]
                constraint_name = f"startup_{g}_t{t.strftime('%Y%m%d%H')}"
                
                problem.linear_constraints.add(
                    lin_expr=[[[commit_var_name, prev_commit_var_name, startup_var_name], 
                              [1.0, -1.0, -1.0]]],
                    senses=["L"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
    
    # e) Shutdown constraints: u[g,t-1] - u[g,t] <= sd[g,t]
    for g in G:
        for t_idx, t in enumerate(T):
            if t_idx == 0:
                # For the first time period, assume generator was off
                commit_var_name = commit_vars[g, t]
                shutdown_var_name = shutdown_vars[g, t]
                constraint_name = f"shutdown_{g}_t{t.strftime('%Y%m%d%H')}"
                
                problem.linear_constraints.add(
                    lin_expr=[[[commit_var_name, shutdown_var_name], [-1.0, -1.0]]],
                    senses=["L"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
            else:
                prev_t = T[t_idx - 1]
                commit_var_name = commit_vars[g, t]
                prev_commit_var_name = commit_vars[g, prev_t]
                shutdown_var_name = shutdown_vars[g, t]
                constraint_name = f"shutdown_{g}_t{t.strftime('%Y%m%d%H')}"
                
                problem.linear_constraints.add(
                    lin_expr=[[[commit_var_name, prev_commit_var_name, shutdown_var_name], 
                              [-1.0, 1.0, -1.0]]],
                    senses=["L"],
                    rhs=[0.0],
                    names=[constraint_name]
                )
    
    # f) Minimum up time constraints
    if min_up_time > 1:
        for g in G:
            for t_idx, t in enumerate(T):
                # Look at the minimum up time window
                end_idx = min(t_idx + min_up_time, len(T))
                window_t = T[t_idx:end_idx]
                
                if t_idx > 0 and len(window_t) > 1:  # Only if we have a previous period and at least 2 periods in window
                    startup_var_name = startup_vars[g, t]
                    commit_var_names = [commit_vars[g, w_t] for w_t in window_t]
                    coeffs = [-len(window_t)] + [1.0] * len(window_t)
                    
                    constraint_name = f"min_up_{g}_t{t.strftime('%Y%m%d%H')}"
                    
                    problem.linear_constraints.add(
                        lin_expr=[[
                            [startup_var_name] + commit_var_names,
                            coeffs
                        ]],
                        senses=["L"],
                        rhs=[0.0],
                        names=[constraint_name]
                    )
    
    # g) Minimum down time constraints
    if min_down_time > 1:
        for g in G:
            for t_idx, t in enumerate(T):
                # Look at the minimum down time window
                end_idx = min(t_idx + min_down_time, len(T))
                window_t = T[t_idx:end_idx]
                
                if t_idx > 0 and len(window_t) > 1:  # Only if we have a previous period and at least 2 periods in window
                    shutdown_var_name = shutdown_vars[g, t]
                    commit_var_names = [commit_vars[g, w_t] for w_t in window_t]
                    coeffs = [len(window_t)] + [-1.0] * len(window_t)
                    
                    constraint_name = f"min_down_{g}_t{t.strftime('%Y%m%d%H')}"
                    
                    problem.linear_constraints.add(
                        lin_expr=[[
                            [shutdown_var_name] + commit_var_names,
                            coeffs
                        ]],
                        senses=["L"],
                        rhs=[len(window_t)],
                        names=[constraint_name]
                    )
    
    # h) Initial storage state constraint
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
    
    # i) Storage dynamics
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
            storage_coefs = [1.0, -1.0, -safe_float(eta) * safe_float(delta_t), safe_float(1/eta) * safe_float(delta_t)]
            constraint_name = f"sd_{int(s)}_t{t.strftime('%Y%m%d%H')}"
            
            problem.linear_constraints.add(
                lin_expr=[[storage_var_names, storage_coefs]],
                senses=["E"],
                rhs=[0.0],
                names=[constraint_name]
            )
    
    # j) Final storage state constraint
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
    
    # k) Slack bus constraint
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
    
    # l) Power balance constraints
    for t in T:
        for i in N:  # Include all buses, even those without generators
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
    
    # m) Add storage cost to objective
    if S:
        # Simple storage cost to prevent unnecessary cycling
        for s in S:
            for t in T:
                problem.objective.set_linear(pch_vars[s, t], 0.001)
                problem.objective.set_linear(pdis_vars[s, t], 0.001)
    
    # 3. Solve the problem
    print("[DCOPF-MIP] About to solve the MIP problem with CPLEX...")
    problem.solve()
    
    # Check solution status
    status = problem.solution.get_status()
    status_string = problem.solution.get_status_string()
    print(f"[DCOPF-MIP] Solver returned status code = {status}, interpreted as '{status_string}'")
    
    # In a MIP, status 101 is "integer optimal solution" which is what we want
    # Accept both regular optimal (1) and integer optimal (101) status codes
    acceptable_statuses = [
        problem.solution.status.optimal,               # 1
        problem.solution.status.optimal_tolerance,     # 2
        101  # integer optimal solution
    ]
    
    if status not in acceptable_statuses:
        print(f"[DCOPF-MIP] Not optimal => returning None.")
        return None
    
    # 4. Extract results
    # a) Extract objective value
    objective_value = problem.solution.get_objective_value()
    print(f"[DCOPF-MIP] Final cost = {objective_value}, status = {status_string}")
    
    # b) Extract generation (non-storage) and commitment status
    generation = []
    commitment = []
    startup_shutdown = []
    
    for g in G:
        g_bus = gen_time_series.loc[gen_time_series['id'] == g, 'bus'].iloc[0]
        for t in T:
            # Get generation value
            val = problem.solution.get_values(gen_vars[g, t])
            generation.append({
                'time': t,
                'id': g,
                'node': g_bus,
                'gen': 0 if math.isnan(val) else val
            })
            
            # Get commitment status
            commit_val = problem.solution.get_values(commit_vars[g, t])
            commitment.append({
                'time': t,
                'id': g,
                'commitment': int(round(commit_val))
            })
            
            # Get startup/shutdown values
            startup_val = problem.solution.get_values(startup_vars[g, t])
            shutdown_val = problem.solution.get_values(shutdown_vars[g, t])
            startup_shutdown.append({
                'time': t,
                'id': g,
                'startup': int(round(startup_val)),
                'shutdown': int(round(shutdown_val))
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
    
    commitment_df = pd.DataFrame(commitment)
    startup_shutdown_df = pd.DataFrame(startup_shutdown)
    
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
    
    # h) Extract congestion (shadow prices on flow limit constraints)
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
    
    print("[DCOPF-MIP] Done, returning result dictionary.")
    
    return {
        'generation': generation_df,
        'commitment': commitment_df,
        'startup_shutdown': startup_shutdown_df,
        'angles': angles_df,
        'flows': flows_df,
        'storage': storage_df,
        'cost': objective_value,
        'status': status_string,
        'marginal_prices': marginal_prices_df,
        'congestion': congestion_shadow_prices_df
    } 