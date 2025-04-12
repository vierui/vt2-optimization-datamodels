#!/usr/bin/env python3
"""
Optimization module for simplified multi-year DC OPF with annualized investments.
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
      }
    """
    years = integrated_network.years
    seasons = list(integrated_network.season_networks.keys())
    first_network = list(integrated_network.season_networks.values())[0] if seasons else None

    generators = first_network.generators.index if first_network else []
    storage_units = first_network.storage_units.index if first_network else []
    buses = first_network.buses.index if first_network else []
    lines = first_network.lines.index if first_network else []

    # -- 1) Create global installation variables
    gen_installed = {(g, y): cp.Variable(boolean=True) for g in generators for y in years}
    storage_installed = {(s, y): cp.Variable(boolean=True) for s in storage_units for y in years}

    # -- 2) Add lifetime-based constraints: once installed, remains installed for its lifetime
    global_constraints = []
    
    # For each generator
    for g in generators:
        lifetime_g = first_network.generators.at[g, 'lifetime_years']
        for y_index, y in enumerate(years):
            for delta in range(1, lifetime_g):
                future_index = y_index + delta
                if future_index < len(years):
                    y_future = years[future_index]
                    global_constraints.append(
                        gen_installed[(g, y_future)] >= gen_installed[(g, y)]
                    )

    # For each storage unit
    for s in storage_units:
        lifetime_s = first_network.storage_units.at[s, 'lifetime_years']
        for y_index, y in enumerate(years):
            for delta in range(1, lifetime_s):
                future_index = y_index + delta
                if future_index < len(years):
                    y_future = years[future_index]
                    global_constraints.append(
                        storage_installed[(s, y_future)] >= storage_installed[(s, y)]
                    )

    # -- 3) Create per-season dispatch variables and constraints 
    season_variables = {}  # dict: season -> dict of variables
    season_constraints = {}  # dict: season -> list of constraints

    # Get load growth factors if available
    load_growth_factors = {}
    for y in years:
        # Default to no growth (factor 1.0)
        load_growth_factors[y] = 1.0
        
    if hasattr(integrated_network, 'load_growth'):
        load_growth_factors = integrated_network.load_growth

    for season in seasons:
        net = integrated_network.season_networks[season]
        T = net.T

        # Dictionary to store the variables for this season
        season_variables[season] = {}
        # List to store constraints for this season across all years
        season_constraints[season] = []

        # 3.1 Generator dispatch variables (vectorized over time)
        p_gen = {}
        for g in generators:
            for y in years:
                p_gen[(g, y)] = cp.Variable(T, nonneg=True)
                if g in net.generators.index:
                    g_nom = net.generators.at[g, 'p_nom']
                    gen_type = net.generators.at[g, 'type']
                    
                    # Different constraints based on generator type
                    if gen_type == 'thermal':
                        # Thermal generators have constant maximum output
                        season_constraints[season].append(
                            p_gen[(g, y)] <= g_nom * gen_installed[(g, y)]
                        )
                    elif gen_type in ['wind', 'solar']:
                        # Wind and solar have time-varying maximum output
                        if ('p_max_pu' in net.generators_t and 
                            g in net.generators_t['p_max_pu']):
                            # Get the time-varying availability profile (already in MW)
                            p_max_vector = net.generators_t['p_max_pu'][g].values[:T]
                            
                            # Apply the constraint: p_gen <= p_max_vector * installed
                            season_constraints[season].append(
                                p_gen[(g, y)] <= cp.multiply(p_max_vector, gen_installed[(g, y)])
                            )
                        else:
                            # Fallback if no profile data is available
                            print(f"Warning: No profile found for {gen_type} generator {g} in {season}")
                            season_constraints[season].append(
                                p_gen[(g, y)] <= g_nom * gen_installed[(g, y)]
                            )
                    else:
                        # Default constraint for any other generator type
                        season_constraints[season].append(
                            p_gen[(g, y)] <= g_nom * gen_installed[(g, y)]
                        )
                else:
                    # If generator not in this network, set capacity to 0
                    season_constraints[season].append(
                        p_gen[(g, y)] <= 0
                    )

        # 3.2 Line flow variables (vectorized over time)
        p_line = {}
        for l in lines:
            for y in years:
                p_line[(l, y)] = cp.Variable(T)
                if l in net.lines.index:
                    line_cap = net.lines.at[l, 's_nom'] if hasattr(net.lines, 's_nom') else 0
                    season_constraints[season].append(
                        cp.abs(p_line[(l, y)]) <= line_cap
                    )

        # 3.3 Storage variables and vectorized capacity constraints
        p_charge = {}
        p_discharge = {}
        soc = {}
        for s in storage_units:
            for y in years:
                p_charge[(s, y)] = cp.Variable(T, nonneg=True)
                p_discharge[(s, y)] = cp.Variable(T, nonneg=True)
                soc[(s, y)] = cp.Variable(T, nonneg=True)
                
                if s in net.storage_units.index:
                    s_p_nom = net.storage_units.at[s, 'p_nom']
                    s_max_hours = net.storage_units.at[s, 'max_hours']
                    eff_in = net.storage_units.at[s, 'efficiency_store']
                    eff_out = net.storage_units.at[s, 'efficiency_dispatch']
                else:
                    s_p_nom, s_max_hours, eff_in, eff_out = 0, 0, 0, 0
                
                s_e_nom = s_p_nom * s_max_hours

                # Vectorized capacity constraints for charging, discharging, and SOC
                season_constraints[season].append(
                    p_charge[(s, y)] <= s_p_nom * storage_installed[(s, y)]
                )
                season_constraints[season].append(
                    p_discharge[(s, y)] <= s_p_nom * storage_installed[(s, y)]
                )
                season_constraints[season].append(
                    soc[(s, y)] <= s_e_nom * storage_installed[(s, y)]
                )
                
                # Vectorized SoC dynamics constraint using slicing:
                # Enforce: soc[1:] == soc[:-1] + eff_in * p_charge[:-1] - (1/eff_out)*p_discharge[:-1]
                season_constraints[season].append(
                    soc[(s, y)][1:] == soc[(s, y)][:-1] + eff_in * p_charge[(s, y)][:-1] - (1.0/eff_out) * p_discharge[(s, y)][:-1]
                )
                
                # Initial and final SoC constraints
                season_constraints[season].append(soc[(s, y)][0] == 0)
                season_constraints[season].append(soc[(s, y)][T-1] == 0)

        # Store storage-related variables for the season
        season_variables[season]['p_gen'] = p_gen
        season_variables[season]['p_line'] = p_line
        season_variables[season]['p_charge'] = p_charge
        season_variables[season]['p_discharge'] = p_discharge
        season_variables[season]['soc'] = soc

        # 3.4 Build the bus load dictionary (same as before)
        bus_load_dict = {}
        for b in buses:
            bus_load_dict[b] = np.zeros(T)
            if not net.loads.empty:
                for ld in net.loads.index:
                    load_bus = net.loads.at[ld, 'bus']
                    # Handle differences in type
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
                        if 'p' in net.loads_t and ld in net.loads_t['p']:
                            load_ts = net.loads_t['p'][ld]
                            bus_load_dict[b] += load_ts.values[:T]
                        else:
                            static_val = net.loads.at[ld, 'p_mw']
                            if not isinstance(static_val, (int, float)) or np.isnan(static_val):
                                raise ValueError(f"Load {ld} at bus {b} has invalid p_mw value: {static_val}")
                            bus_load_dict[b] += np.ones(T) * static_val

        # 3.5 Vectorized nodal power balance constraints:
        for y in years:
            # Apply load growth factor for this year
            growth_factor = load_growth_factors.get(y, 1.0)
            
            for b in buses:
                # Scale the load by the growth factor for this year
                scaled_load = bus_load_dict[b] * growth_factor
                
                # Sum of generation at bus b in year y
                g_at_bus = [g for g in generators if g in net.generators.index and net.generators.at[g, 'bus'] == b]
                gen_sum = sum(season_variables[season]['p_gen'][(g, y)] for g in g_at_bus) if g_at_bus else 0
                
                # Sum storage net injection (discharge minus charge) at bus b
                s_at_bus = [s for s in storage_units if s in net.storage_units.index and net.storage_units.at[s, 'bus'] == b]
                st_net = sum(season_variables[season]['p_discharge'][(s, y)] - season_variables[season]['p_charge'][(s, y)]
                             for s in s_at_bus) if s_at_bus else 0
                
                # Sum flows leaving (flow_out) and entering (flow_in) bus b
                lines_from = [l for l in lines if l in net.lines.index and net.lines.at[l, 'from_bus'] == b]
                flow_out = sum(season_variables[season]['p_line'][(l, y)] for l in lines_from) if lines_from else 0

                lines_to = [l for l in lines if l in net.lines.index and net.lines.at[l, 'to_bus'] == b]
                flow_in = sum(season_variables[season]['p_line'][(l, y)] for l in lines_to) if lines_to else 0

                # Create a constant vector for the scaled load at bus b
                load_vec = cp.Constant(scaled_load)
                # Impose the power balance over the entire time horizon at bus b for year y:
                season_constraints[season].append(
                    (gen_sum + st_net + flow_in) == (load_vec + flow_out)
                )

    # -- 4) Build the objective function: Operational cost + annualized capital cost
    operational_obj = 0
    for season in seasons:
        net = integrated_network.season_networks[season]
        T = net.T
        weight_weeks = integrated_network.season_weights.get(season, 0)

        for y in years:
            for g in generators:
                mc = net.generators.at[g, 'marginal_cost'] if g in net.generators.index else 0
                # Use cp.sum to sum over time periods
                operational_obj += weight_weeks * mc * cp.sum(season_variables[season]['p_gen'][(g,y)])

    capital_obj = 0
    if first_network:
        for y in years:
            for g in generators:
                capex_per_mw = first_network.generators.at[g, 'capex_per_mw'] if g in first_network.generators.index else 0
                p_nom = first_network.generators.at[g, 'p_nom'] if g in first_network.generators.index else 0
                lifetime = first_network.generators.at[g, 'lifetime_years'] if g in first_network.generators.index else 20
                annual_capex_g = (capex_per_mw * p_nom) / lifetime
                capital_obj += annual_capex_g * gen_installed[(g,y)]

            for s in storage_units:
                capex_pm = first_network.storage_units.at[s, 'capex_per_mw'] if s in first_network.storage_units.index else 0
                p_nom_s = first_network.storage_units.at[s, 'p_nom'] if s in first_network.storage_units.index else 0
                lifetime_s = first_network.storage_units.at[s, 'lifetime_years'] if s in first_network.storage_units.index else 10
                annual_capex_s = (capex_pm * p_nom_s) / lifetime_s
                capital_obj += annual_capex_s * storage_installed[(s,y)]

    total_cost = operational_obj + capital_obj
    objective = cp.Minimize(total_cost)
    
    # Combine all constraints
    all_constraints = []
    all_constraints.extend(global_constraints)
    for s in seasons:
        all_constraints.extend(season_constraints[s])

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
    Solve the integrated multi-year problem:
      1) Create the problem
      2) Solve with CPLEX
      3) Extract numeric variable values
      4) Store results in integrated_network.integrated_results
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
    cplex_params = {'threads': 10, 'timelimit': 1200}
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

    # 3) Extract variable values
    val_gen_installed = {k: float(var.value) if var.value is not None else 0.0 
                         for k, var in problem_dict['variables']['gen_installed'].items()}
    val_storage_installed = {k: float(var.value) if var.value is not None else 0.0 
                             for k, var in problem_dict['variables']['storage_installed'].items()}

    # 4) Save solution
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