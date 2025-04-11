#!/usr/bin/env python3
"""
Optimization module for simplified multi-year DC OPF with annualized investments
No layering: create, solve, extract numeric results, store in integrated_network.
"""

import cvxpy as cp
import pandas as pd
import numpy as np

def create_integrated_dcopf_problem(integrated_network):
    """
    Create an integrated multi-year DC OPF problem with:
      - One binary variable per asset & year for "installed".
      - Generator dispatch & storage variables for each year+season.
      - Summed operational + annualized capital cost objective.
      - No re-installation or replacement logic.
        
    Returns:
      problem_dict = {
        'objective': cp.Minimize(...),
        'constraints': [...],
        'variables': {
          'gen_installed': {...},
          'storage_installed': {...},
          'season_variables': {season: {...}}
        },
        'years': [...],
        'seasons': [...],
        'season_costs': {}
      }
    """
    years = integrated_network.years
    seasons = list(integrated_network.season_networks.keys())
    first_network = list(integrated_network.season_networks.values())[0] if seasons else None

    generators = first_network.generators.index if first_network else []
    storage_units = first_network.storage_units.index if first_network else []
    buses = first_network.buses.index if first_network else []
    lines = first_network.lines.index if first_network else []

    # -- 1) Create global variables
    gen_installed = {(g, y): cp.Variable(boolean=True) for g in generators for y in years}
    storage_installed = {(s, y): cp.Variable(boolean=True) for s in storage_units for y in years}

    # -- 2) Add monotonic constraints (if wanted): once installed, remains installed
    global_constraints = []
    for g in generators:
        for i in range(1, len(years)):
            global_constraints.append(gen_installed[(g, years[i])] >= gen_installed[(g, years[i-1])])
    for s in storage_units:
        for i in range(1, len(years)):
            global_constraints.append(storage_installed[(s, years[i])] >= storage_installed[(s, years[i-1])])

    # -- 3) Create per-season dispatch variables and constraints 
    season_variables = {}  # dict: (season, year) -> dict of variables
    season_constraints = {}  # dict: season -> list of constraints

    for season in seasons:
        net = integrated_network.season_networks[season]
        T = net.T

        # Dictionary to store the variables for this season
        season_variables[season] = {}
        # List to store constraints for this season across all years
        season_constraints[season] = []

        # 3.1 Generator dispatch variables p_gen[(g,y)][t]
        p_gen = {}
        for g in generators:
            for y in years:
                # Create a dispatch variable for each hour t
                p_gen[(g, y)] = cp.Variable(T, nonneg=True)

                # Generators can only dispatch if installed this year
                g_nom = net.generators.at[g, 'p_nom'] if g in net.generators.index else 0
                # p_gen can't exceed g_nom for each t
                for t in range(T):
                    season_constraints[season].append(p_gen[(g, y)][t] <= g_nom * gen_installed[(g, y)])

        # 3.2 Line flow variables p_line[(l,y)][t]
        p_line = {}
        for l in lines:
            for y in years:
                # Create a real number variable for line flows
                p_line[(l, y)] = cp.Variable(T)
                
                # Line limits
                if l in net.lines.index:
                    line_cap = net.lines.at[l, 's_nom'] if hasattr(net.lines, 's_nom') else 0
                    # Bidirectional line capacity
                    for t in range(T):
                        season_constraints[season].append(cp.abs(p_line[(l, y)][t]) <= line_cap)

        # 3.3 Storage variables and constraints
        p_charge = {}
        p_discharge = {}
        soc = {}
        for s in storage_units:
            for y in years:
                # Storage variables: charging, discharging, state of charge
                p_charge[(s, y)] = cp.Variable(T, nonneg=True)
                p_discharge[(s, y)] = cp.Variable(T, nonneg=True)
                soc[(s, y)] = cp.Variable(T, nonneg=True)
                
                # Get storage parameters
                if s in net.storage_units.index:
                    s_p_nom = net.storage_units.at[s, 'p_nom']
                    s_max_hours = net.storage_units.at[s, 'max_hours']
                    eff_in = net.storage_units.at[s, 'efficiency_store']
                    eff_out = net.storage_units.at[s, 'efficiency_dispatch']
                else:
                    s_p_nom, s_max_hours, eff_in, eff_out = 0, 0, 0, 0
                
                # Max power charge/discharge
                for t in range(T):
                    season_constraints[season].append(
                        p_charge[(s, y)][t] <= s_p_nom * storage_installed[(s, y)]
                    )
                    season_constraints[season].append(
                        p_discharge[(s, y)][t] <= s_p_nom * storage_installed[(s, y)]
                    )
                    # Max energy stored (SOC)
                    s_e_nom = s_p_nom * s_max_hours
                    season_constraints[season].append(
                        soc[(s, y)][t] <= s_e_nom * storage_installed[(s, y)]
                    )
                
                # SOC dynamics
                for t in range(1, T):
                    season_constraints[season].append(
                        soc[(s, y)][t] == soc[(s, y)][t-1] 
                        + eff_in * p_charge[(s, y)][t-1] 
                        - (1.0/eff_out) * p_discharge[(s, y)][t-1]
                    )
                
                # Initial SOC = 0 and final SOC = 0
                # This is a simplification to avoid cross-season coupling
                season_constraints[season].append(soc[(s, y)][0] == 0)
                season_constraints[season].append(soc[(s, y)][T-1] == 0)

        # Store all the variables
        season_variables[season]['p_gen'] = p_gen
        season_variables[season]['p_line'] = p_line
        season_variables[season]['p_charge'] = p_charge
        season_variables[season]['p_discharge'] = p_discharge
        season_variables[season]['soc'] = soc

        # 3.4 Nodal power balance constraints
        # We gather loads from net.loads_t or net.loads
        # Create load dictionary for each bus
        bus_load_dict = {}
        
        # Print some diagnostic information
        print(f"Season: {season}, Bus types: {[type(b) for b in buses]}")
        if not net.loads.empty:
            print(f"Load data columns: {net.loads.columns.tolist()}")
            print(f"Load bus column type: {net.loads['bus'].dtype}")
            print(f"Load bus values: {net.loads['bus'].tolist()}")
            print(f"Load p_mw values: {net.loads['p_mw'].tolist()}")
        
        for b in buses:
            bus_load_dict[b] = np.zeros(T)
            # sum all loads that belong to bus b
            if not net.loads.empty:
                for ld in net.loads.index:
                    # Ensure we're comparing the same types
                    load_bus = net.loads.at[ld, 'bus']
                    # Convert types if needed to ensure comparison works
                    if isinstance(b, int) and isinstance(load_bus, str):
                        try:
                            load_bus = int(load_bus)
                        except ValueError:
                            print(f"Warning: Could not convert load bus {load_bus} to int for comparison with bus {b}")
                    elif isinstance(b, str) and isinstance(load_bus, int):
                        b_str = str(b)
                        if load_bus == int(b_str):
                            load_bus = b_str
                    
                    if load_bus == b:
                        # First check if there's a time series for this load
                        if 'p' in net.loads_t and ld in net.loads_t['p']:
                            load_ts = net.loads_t['p'][ld]
                            bus_load_dict[b] += load_ts.values[:T]
                            print(f"Added time-series load {ld} at bus {b}")
                        else:
                            # Use static load value
                            static_val = net.loads.at[ld,'p_mw']
                            if not isinstance(static_val, (int, float)) or np.isnan(static_val):
                                raise ValueError(f"Load {ld} at bus {b} has invalid p_mw value: {static_val}")
                            bus_load_dict[b] += np.ones(T) * static_val
                            print(f"Added static load {ld} at bus {b} with value {static_val} MW")
        
        # Debug print to verify loads are correctly mapped
        for b in buses:
            if np.any(bus_load_dict[b] > 0):
                print(f"DEBUG: Bus {b} has load: max={np.max(bus_load_dict[b])}, mean={np.mean(bus_load_dict[b])}")
            else:
                print(f"DEBUG: Bus {b} has NO load (all zeros)")
                
        # Print the first few timesteps for debugging
        print(f"===== Season {season} debug: first 5 time periods =====")
        for b in buses:
            print(f"  Bus {b}:")
            for t in range(min(5, T)):
                print(f"    t={t}: load={bus_load_dict[b][t]:.2f} MW")

        for y in years:
            for t in range(T):
                for b in buses:
                    # generation
                    g_at_bus = [g for g in generators
                                if g in net.generators.index 
                                and net.generators.at[g,'bus'] == b]
                    gen_sum = sum(season_variables[season]['p_gen'][(g,y)][t] for g in g_at_bus)

                    # storage net
                    s_at_bus = [s for s in storage_units
                                if s in net.storage_units.index
                                and net.storage_units.at[s,'bus'] == b]
                    st_net = sum( season_variables[season]['p_discharge'][(s,y)][t]
                                  - season_variables[season]['p_charge'][(s,y)][t]
                                  for s in s_at_bus )

                    # flows in/out
                    lines_from = [l_ for l_ in lines
                                  if l_ in net.lines.index
                                  and net.lines.at[l_,'from_bus'] == b]
                    lines_to = [l_ for l_ in lines
                                if l_ in net.lines.index
                                and net.lines.at[l_,'to_bus'] == b]
                    flow_out = sum( season_variables[season]['p_line'][(l_,y)][t] for l_ in lines_from)
                    flow_in = sum( season_variables[season]['p_line'][(l_,y)][t] for l_ in lines_to)

                    # load
                    load_val = bus_load_dict[b][t] if b in bus_load_dict else 0

                    # balance
                    season_constraints[season].append(
                        gen_sum + st_net + flow_in == load_val + flow_out
                    )

    # -- 4) Build the objective function: 
    # operation cost + annualized capital cost
    # We'll do a simple approach: sum of generator dispatch * marginal cost
    # across seasons, plus sum of installed capacity * annual capex

    operational_obj = 0
    for season in seasons:
        net = integrated_network.season_networks[season]
        T = net.T
        weight_weeks = integrated_network.season_weights.get(season, 0)
        # or a fraction (weight_weeks / 52)...

        for y in years:
            for g in generators:
                if g in net.generators.index:
                    mc = net.generators.at[g,'marginal_cost']
                else:
                    mc = 0
                # sum of p_gen*gencost
                for t in range(T):
                    operational_obj += weight_weeks * mc * season_variables[season]['p_gen'][(g,y)][t]

    capital_obj = 0
    if first_network:
        # for each year we pay annual capex for each installed generator & storage
        for y in years:
            for g in generators:
                # annual capex = (capex_per_mw * p_nom) / lifetime
                capex_per_mw = first_network.generators.at[g,'capex_per_mw'] if g in first_network.generators.index else 0
                p_nom = first_network.generators.at[g,'p_nom'] if g in first_network.generators.index else 0
                lifetime = first_network.generators.at[g,'lifetime_years'] if g in first_network.generators.index else 20
                annual_capex_g = (capex_per_mw * p_nom)/lifetime
                capital_obj += annual_capex_g * gen_installed[(g,y)]

            for s in storage_units:
                capex_pm = first_network.storage_units.at[s,'capex_per_mw'] if s in first_network.storage_units.index else 0
                p_nom_s = first_network.storage_units.at[s,'p_nom'] if s in first_network.storage_units.index else 0
                lifetime_s = first_network.storage_units.at[s,'lifetime_years'] if s in first_network.storage_units.index else 10
                # optionally also handle capex_per_mwh if you want
                annual_capex_s = (capex_pm * p_nom_s)/lifetime_s
                capital_obj += annual_capex_s * storage_installed[(s,y)]

    total_cost = operational_obj + capital_obj
    objective = cp.Minimize(total_cost)
    
    # Combine constraints
    all_constraints = []
    all_constraints.extend(global_constraints)
    for s in seasons:
        all_constraints.extend(season_constraints[s])

    # Return dict
    return {
        'objective': objective,
        'constraints': all_constraints,
        'variables': {
            'gen_installed': gen_installed,
            'storage_installed': storage_installed,
            'season_variables': season_variables
        },
        'years': years,
        'seasons': seasons
    }


def solve_multi_year_investment(integrated_network, solver_options=None):
    """
    Solve the integrated multi-year problem in one function:
      1) create problem
      2) solve with CPLEX
      3) extract numeric variable values
      4) store results in integrated_network.integrated_results
    """
    if solver_options is None:
        solver_options = {}
    
    years = integrated_network.years
    seasons = integrated_network.seasons
    print(f"Solving multi-year investment: {len(seasons)} seasons, {len(years)} years")

    # 1) Create problem
    problem_dict = create_integrated_dcopf_problem(integrated_network)
    prob = cp.Problem(problem_dict['objective'], problem_dict['constraints'])

    # 2) Solve with CPLEX
    cplex_params = {'threads': 12, 'timelimit': 3600}
    cplex_params.update(solver_options)

    try:
        print("Solving with CPLEX ...")
        prob.solve(solver=cp.CPLEX, verbose=True, cplex_params=cplex_params)
    except Exception as e:
        print("Solver failed:", e)
        integrated_network.integrated_results = {
                    'status': 'failed',
            'value': None,
            'success': False,
            'variables': {}
        }
        return integrated_network.integrated_results

    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        print(f"Solve ended with status {status}")
        integrated_network.integrated_results = {
            'status': status,
            'value': None,
            'success': False,
            'variables': {}
        }
        return integrated_network.integrated_results

    # 3) Extract numeric variable values
    val_gen_installed = {}
    for k, var in problem_dict['variables']['gen_installed'].items():
        val_gen_installed[k] = float(var.value) if var.value is not None else 0.0

    val_storage_installed = {}
    for k, var in problem_dict['variables']['storage_installed'].items():
        val_storage_installed[k] = float(var.value) if var.value is not None else 0.0

    # 4) Save solution in integrated_network
    integrated_network.integrated_results = {
        'status': status,
        'value': float(prob.value),
        'success': True,
        'variables': {
            'gen_installed': val_gen_installed,
            'storage_installed': val_storage_installed
        }
    }
    print(f"Solve success. Status={status}, Objective={prob.value:.2f}")

    return integrated_network.integrated_results