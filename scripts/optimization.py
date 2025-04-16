#!/usr/bin/env python3
"""
Optimization module for simplified multi-year DC OPF with annualized investments.
No layering: create, solve, extract numeric results, store in integrated_network.
"""

import cvxpy as cp
import pandas as pd
import numpy as np
import os
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def compute_crf(lifetime, discount_rate):
    """
    Compute the capital recovery factor (CRF) for an asset.
    
    CRF = i * (1+i)^n / ((1+i)^n - 1)
    where:
      i = discount_rate,
      n = lifetime (years)
      
    Returns 1.0 if lifetime is zero or close to zero.
    """
    if lifetime is None or lifetime <= 0:
        return 1.0
    i = discount_rate
    n = lifetime
    numerator = i * (1 + i)**n
    denominator = (1 + i)**n - 1
    if abs(denominator) < 1e-9:
        return 1.0
    return numerator / denominator

def compute_discount_sum(lifetime, discount_rate):
    """
    Compute the sum of discounted factors over the asset's lifetime:
    
    A = sum_{t=1}^{n} 1/(1+i)^t = (1 - 1/(1+i)^n) / i  if i>0
    """
    if discount_rate <= 0:
        return lifetime
    return (1 - 1 / ((1 + discount_rate) ** lifetime)) / discount_rate

def create_integrated_dcopf_problem(integrated_network):
    """
    Create an integrated multi-year DC OPF problem with:
      - Separate 'build' and 'installed' binary variables per asset & year.
      - Generator dispatch & storage variables for each year+season.
      - Summed operational + annualized capital cost objective (cost based on 'build').
      - Lifetimes handled by linking 'build' to 'installed' status.
        
    Returns:
      problem_dict = {
        'objective': cp.Minimize(...),
        'constraints': [...],
        'variables': {
          'gen_build': {...},        # New build variables
          'storage_build': {...},  # New build variables
          'gen_installed': {...},    # Operational status
          'storage_installed': {...},# Operational status
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

    # -- 1) Create global 'build' and 'installed' variables
    gen_build = {(g, y): cp.Variable(boolean=True, name=f"gen_build_{g}_{y}") for g in generators for y in years}
    storage_build = {(s, y): cp.Variable(boolean=True, name=f"storage_build_{s}_{y}") for s in storage_units for y in years}
    
    gen_installed = {(g, y): cp.Variable(boolean=True, name=f"gen_installed_{g}_{y}") for g in generators for y in years}
    storage_installed = {(s, y): cp.Variable(boolean=True, name=f"storage_installed_{s}_{y}") for s in storage_units for y in years}

    # -- 2) Link 'build' to 'installed' status based on lifetime
    global_constraints = []
    
    # For each generator g, link installed[g, y] to build decisions within its lifetime
    for g in generators:
        lifetime_g = first_network.generators.at[g, 'lifetime_years']
        # Ensure lifetime is treated as an integer for range calculation
        lifetime_g_int = int(lifetime_g) if pd.notna(lifetime_g) else 0 
        for y_idx, y in enumerate(years):
            # An asset is installed in year y if it was built in a year y_build such that y is within its lifetime
            # Condition: yb_idx <= y_idx < yb_idx + lifetime_g_int  (equivalent to yb <= y < yb + lifetime_g)
            relevant_build_years = [yb for yb_idx, yb in enumerate(years) if y_idx >= yb_idx and y_idx < yb_idx + lifetime_g_int]
            if relevant_build_years:
                 build_vars_list = [gen_build[(g, yb)] for yb in relevant_build_years]
                 # This constraint enforces gen_installed <= sum(relevant gen_build)
                 global_constraints.append(
                     gen_installed[(g, y)] <= cp.sum(build_vars_list) # Use the list here
                 )
            else:
                # If no possible build year could cover this year, it cannot be installed
                 global_constraints.append(gen_installed[(g, y)] <= 0)
            # Ensure that if we build in year y, we are installed in year y (might be redundant with above but clearer)
            global_constraints.append(gen_installed[(g,y)] >= gen_build[(g,y)])
            
            # Add a constraint to prevent redundant builds
            # Only allow building in the current year if:
            # - It's the first year, OR
            # - The asset wasn't installed in the previous year
            if y_idx > 0:
                prev_year = years[y_idx - 1]
                global_constraints.append(
                    gen_build[(g, y)] <= 1 - gen_installed[(g, prev_year)] + 
                    # Allow rebuilding if previous installation is expiring
                    cp.sum([gen_build[(g, yb)] for yb_idx, yb in enumerate(years) 
                           if y_idx - 1 == yb_idx + lifetime_g_int - 1 and yb_idx < y_idx])
                )


    # For each storage unit s, link installed[s, y] to build decisions within its lifetime
    for s in storage_units:
        lifetime_s = first_network.storage_units.at[s, 'lifetime_years']
        # Ensure lifetime is treated as an integer
        lifetime_s_int = int(lifetime_s) if pd.notna(lifetime_s) else 0
        for y_idx, y in enumerate(years):
            relevant_build_years = [yb for yb_idx, yb in enumerate(years) if y_idx >= yb_idx and y_idx < yb_idx + lifetime_s_int]
            if relevant_build_years:
                 build_vars_list = [storage_build[(s, yb)] for yb in relevant_build_years]
                 # This constraint enforces storage_installed <= sum(relevant storage_build)
                 global_constraints.append(
                     storage_installed[(s, y)] <= cp.sum(build_vars_list) # Use the list here
                 )
            else:
                 global_constraints.append(storage_installed[(s, y)] <= 0)
            # Ensure that if we build in year y, we are installed in year y
            global_constraints.append(storage_installed[(s,y)] >= storage_build[(s,y)])
            
            # Add a constraint to prevent redundant builds for storage units
            # Only allow building in the current year if:
            # - It's the first year, OR
            # - The asset wasn't installed in the previous year
            if y_idx > 0:
                prev_year = years[y_idx - 1]
                global_constraints.append(
                    storage_build[(s, y)] <= 1 - storage_installed[(s, prev_year)] + 
                    # Allow rebuilding if previous installation is expiring
                    cp.sum([storage_build[(s, yb)] for yb_idx, yb in enumerate(years) 
                           if y_idx - 1 == yb_idx + lifetime_s_int - 1 and yb_idx < y_idx])
                )

    # -- 3) Create per-season dispatch variables and constraints 
    season_variables = {}
    season_constraints = {}

    load_growth_factors = integrated_network.load_growth if hasattr(integrated_network, 'load_growth') else {y: 1.0 for y in years}

    for season in seasons:
        net = integrated_network.season_networks[season]
        T = net.T

        season_variables[season] = {}
        season_constraints[season] = []

        # 3.1 Generator dispatch variables (using gen_installed for capacity)
        p_gen = {}
        for g in generators:
            for y in years:
                p_gen[(g, y)] = cp.Variable(T, nonneg=True, name=f"p_gen_{season}_{g}_{y}")
                if g in net.generators.index:
                    g_nom = net.generators.at[g, 'p_nom']
                    gen_type = net.generators.at[g, 'type']
                    
                    # Use gen_installed[(g, y)] to enable/disable capacity
                    if gen_type == 'thermal':
                        season_constraints[season].append(
                            p_gen[(g, y)] <= g_nom * gen_installed[(g, y)] 
                        )
                    elif gen_type in ['wind', 'solar']:
                        if 'p_max_pu' in net.generators_t and g in net.generators_t['p_max_pu']:
                            p_max_vector = net.generators_t['p_max_pu'][g].values[:T]
                            scaled_p_max = p_max_vector * g_nom
                            season_constraints[season].append(
                                p_gen[(g, y)] <= cp.multiply(scaled_p_max, gen_installed[(g, y)])
                            )
                        else:
                            season_constraints[season].append(
                                p_gen[(g, y)] <= g_nom * gen_installed[(g, y)]
                            )
                    else:
                        season_constraints[season].append(
                            p_gen[(g, y)] <= g_nom * gen_installed[(g, y)]
                        )
                else:
                    season_constraints[season].append(p_gen[(g, y)] <= 0)

        # 3.2 Line flow variables (no change needed)
        p_line = {}
        for l in lines:
            for y in years:
                p_line[(l, y)] = cp.Variable(T, name=f"p_line_{season}_{l}_{y}")
                if l in net.lines.index:
                    line_cap = net.lines.at[l, 's_nom'] if hasattr(net.lines, 's_nom') else 0
                    season_constraints[season].append(cp.abs(p_line[(l, y)]) <= line_cap)

        # 3.3 Storage variables (using storage_installed for capacity)
        p_charge = {}
        p_discharge = {}
        soc = {}
        for s in storage_units:
            for y in years:
                p_charge[(s, y)] = cp.Variable(T, nonneg=True, name=f"p_charge_{season}_{s}_{y}")
                p_discharge[(s, y)] = cp.Variable(T, nonneg=True, name=f"p_discharge_{season}_{s}_{y}")
                soc[(s, y)] = cp.Variable(T, nonneg=True, name=f"soc_{season}_{s}_{y}")
                
                if s in net.storage_units.index:
                    s_p_nom = net.storage_units.at[s, 'p_nom']
                    s_max_hours = net.storage_units.at[s, 'max_hours']
                    eff_in = net.storage_units.at[s, 'efficiency_store']
                    eff_out = net.storage_units.at[s, 'efficiency_dispatch']
                    s_e_nom = s_p_nom * s_max_hours
                else:
                    s_p_nom, s_e_nom, eff_in, eff_out = 0, 0, 1, 1 # Set defaults for safety

                # Use storage_installed[(s, y)] for capacity constraints
                season_constraints[season].append(p_charge[(s, y)] <= s_p_nom * storage_installed[(s, y)])
                season_constraints[season].append(p_discharge[(s, y)] <= s_p_nom * storage_installed[(s, y)])
                season_constraints[season].append(soc[(s, y)] <= s_e_nom * storage_installed[(s, y)])
                
                # SoC dynamics (no change needed in logic)
                season_constraints[season].append(
                    soc[(s, y)][1:] == soc[(s, y)][:-1] + eff_in * p_charge[(s, y)][:-1] - (1.0/eff_out) * p_discharge[(s, y)][:-1]
                )
                season_constraints[season].append(soc[(s, y)][0] == 0)
                # Relax final SoC constraint slightly for feasibility
                season_constraints[season].append(soc[(s, y)][T-1] >= 0)
                season_constraints[season].append(soc[(s, y)][T-1] <= s_e_nom * storage_installed[(s,y)] * 0.1) # e.g. max 10% end SoC


        season_variables[season]['p_gen'] = p_gen
        season_variables[season]['p_line'] = p_line
        season_variables[season]['p_charge'] = p_charge
        season_variables[season]['p_discharge'] = p_discharge
        season_variables[season]['soc'] = soc

        # 3.4 Build the bus load dictionary (no change needed)
        bus_load_dict = {b: np.zeros(T) for b in buses}
        if not net.loads.empty:
            for ld in net.loads.index:
                load_bus_orig = net.loads.at[ld, 'bus']
                # Try to match bus types (int vs str)
                matched_bus = None
                for b in buses:
                   try:
                       if str(load_bus_orig) == str(b):
                           matched_bus = b
                           break
                   except: # Handle potential type errors during comparison
                        pass
                
                if matched_bus is not None:
                    if 'p' in net.loads_t and ld in net.loads_t['p']:
                        load_ts = net.loads_t['p'][ld].values[:T]
                        bus_load_dict[matched_bus] += load_ts
                    else:
                        static_val = net.loads.at[ld, 'p_mw']
                        if pd.notna(static_val):
                            bus_load_dict[matched_bus] += np.ones(T) * static_val
                        else:
                            logger.warning(f"Load {ld} at bus {matched_bus} has invalid p_mw value: {static_val}. Setting to 0.")

        # 3.5 Vectorized nodal power balance constraints (no change needed in logic)
        for y in years:
            growth_factor = load_growth_factors.get(y, 1.0)
            for b in buses:
                scaled_load = bus_load_dict[b] * growth_factor
                
                # Find generators, storage, lines connected to this bus (handle type mismatches)
                g_at_bus = [g for g in generators if g in net.generators.index and str(net.generators.at[g, 'bus']) == str(b)]
                s_at_bus = [s for s in storage_units if s in net.storage_units.index and str(net.storage_units.at[s, 'bus']) == str(b)]
                lines_from = [l for l in lines if l in net.lines.index and str(net.lines.at[l, 'from_bus']) == str(b)]
                lines_to = [l for l in lines if l in net.lines.index and str(net.lines.at[l, 'to_bus']) == str(b)]

                gen_sum = cp.sum([p_gen[(g, y)] for g in g_at_bus]) if g_at_bus else 0
                st_net = cp.sum([p_discharge[(s, y)] - p_charge[(s, y)] for s in s_at_bus]) if s_at_bus else 0
                flow_out = cp.sum([p_line[(l, y)] for l in lines_from]) if lines_from else 0
                flow_in = cp.sum([p_line[(l, y)] for l in lines_to]) if lines_to else 0
                
                load_vec = cp.Constant(scaled_load)
                
                season_constraints[season].append(
                    (gen_sum + st_net + flow_in) == (load_vec + flow_out)
                )

    # -- 4) Build the objective function: Operational cost + annualized capital cost (based on 'build')
    operational_obj = 0
    for season in seasons:
        net = integrated_network.season_networks[season]
        weight_weeks = integrated_network.season_weights.get(season, 0)
        for y in years:
            for g in generators:
                if g in net.generators.index and (g, y) in p_gen: # Check if var exists
                    mc = net.generators.at[g, 'marginal_cost']
                    discount_rate = net.generators.at[g, 'discount_rate']
                    discount_factor = 1.0 / ((1.0 + discount_rate) ** max(0, y - years[0])) # Use relative year for discount
                    operational_obj += weight_weeks * mc * cp.sum(p_gen[(g, y)]) * discount_factor

    capital_obj = 0
    if first_network:
        # Process generator costs using the annuity approach
        for g in generators:
            if g not in first_network.generators.index: continue

            # Retrieve fixed parameters from the network data
            capex = first_network.generators.at[g, 'capex']
            lifetime = first_network.generators.at[g, 'lifetime_years']
            discount_rate = first_network.generators.at[g, 'discount_rate']
            # Check if operating_costs exists, otherwise default to 0
            opex_fraction = 0.0
            if 'operating_costs' in first_network.generators.columns:
                opex_fraction = first_network.generators.at[g, 'operating_costs']
            
            # Default lifetime if needed
            if lifetime is None or pd.isna(lifetime) or lifetime <= 0:
                lifetime = 1

            # Compute the sum of discounted factors over the asset's lifetime
            discount_sum = compute_discount_sum(lifetime, discount_rate)
            # Compute the asset's NPV (here treating CAPEX and operating cost as cash outflows)
            npv = capex + (capex * opex_fraction * discount_sum)
            # Compute the capital recovery factor (CRF)
            crf = compute_crf(lifetime, discount_rate)
            # Annual cost (annuity) for the asset
            annual_asset_cost = npv * crf

            # For every planning year, if the generator is installed, add its full annuity cost (without additional discounting)
            for y in years:
                capital_obj += annual_asset_cost * gen_installed[(g, y)]

        # Process storage costs similarly:
        for s in storage_units:
            if s not in first_network.storage_units.index: continue

            # Retrieve fixed parameters from the network data
            capex = first_network.storage_units.at[s, 'capex']
            lifetime_s = first_network.storage_units.at[s, 'lifetime_years']
            discount_rate_s = first_network.storage_units.at[s, 'discount_rate']
            # Check if operating_costs exists, otherwise default to 0
            opex_fraction_s = 0.0
            if 'operating_costs' in first_network.storage_units.columns:
                opex_fraction_s = first_network.storage_units.at[s, 'operating_costs']
            
            # Default lifetime if needed
            if lifetime_s is None or pd.isna(lifetime_s) or lifetime_s <= 0:
                lifetime_s = 1

            # Compute the sum of discounted factors over the asset's lifetime
            discount_sum_s = compute_discount_sum(lifetime_s, discount_rate_s)
            # Compute the asset's NPV (here treating CAPEX and operating cost as cash outflows)
            npv_s = capex + (capex * opex_fraction_s * discount_sum_s)
            # Compute the capital recovery factor (CRF)
            crf_s = compute_crf(lifetime_s, discount_rate_s)
            # Annual cost (annuity) for the asset
            annual_asset_cost_s = npv_s * crf_s

            # For every planning year, if the storage is installed, add its full annuity cost (without additional discounting)
            for y in years:
                capital_obj += annual_asset_cost_s * storage_installed[(s, y)]

    total_cost = operational_obj + capital_obj
    objective = cp.Minimize(total_cost)
    
    all_constraints = global_constraints + [c for s_constrs in season_constraints.values() for c in s_constrs]

    # Sanity check for NaNs in constraints or objective coefficients
    # (Add checks here if needed, e.g., iterating through constraints and objective terms)

    return {
        'objective': objective,
        'constraints': all_constraints,
        'variables': {
            'gen_build': gen_build,             # Return new build vars
            'storage_build': storage_build,     # Return new build vars
            'gen_installed': gen_installed,     # Return installed vars
            'storage_installed': storage_installed, # Return installed vars
            'season_variables': season_variables
        },
        'years': years,
        'seasons': seasons
    }


def solve_multi_year_investment(integrated_network, solver_options=None):
    """
    Solve the integrated multi-year problem using the build/installed formulation:
      1) Create the problem
      2) Solve with CPLEX
      3) Extract numeric variable values (including build vars)
      4) Store results in integrated_network.integrated_results
    """
    if solver_options is None:
        solver_options = {}
    
    years = integrated_network.years
    seasons = integrated_network.seasons
    first_network = list(integrated_network.season_networks.values())[0] if seasons else None
    logger.info(f"Solving multi-year investment: {len(seasons)} seasons, {len(years)} years using build/installed formulation.")

    # 1) Create problem
    problem_dict = create_integrated_dcopf_problem(integrated_network)
    prob = cp.Problem(problem_dict['objective'], problem_dict['constraints'])

    # 2) Solve with CPLEX
    cplex_params = {'threads': 10, 'timelimit': 1200}
    cplex_params.update(solver_options)

    try:
        logger.info("Solving with CPLEX...")
        # Increase MIP tolerances for potentially challenging problems
        cplex_params.update({'mip.tolerances.mipgap': 0.01, 'mip.tolerances.absmipgap': 1.0})
        prob.solve(solver=cp.CPLEX, verbose=True, cplex_params=cplex_params)
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        # Check if the error message contains DCP details
        if "Problem does not follow DCP rules" in str(e):
            logger.error("DCP Error Details:")
            # (CVXPY often prints DCP errors to stdout/stderr, check console)
            # Extracting specific details programmatically can be complex
            pass 
        integrated_network.integrated_results = {
            'status': 'failed',
            'value': None,
            'success': False,
            'variables': {}
        }
        return integrated_network.integrated_results

    status = prob.status
    objective_value = prob.value if status in ("optimal", "optimal_inaccurate") else None

    if status not in ("optimal", "optimal_inaccurate"):
        logger.warning(f"Solve ended with status {status}. Objective value: {objective_value}")
        # Attempt to analyze infeasibility if CPLEX provides details
        # (This requires specific CPLEX API calls not directly available via CVXPY)
        integrated_network.integrated_results = {
            'status': status,
            'value': objective_value,
            'success': False,
            'variables': {}
        }
        return integrated_network.integrated_results
    else:
        logger.info(f"Solve successful with status {status}. Objective value: {objective_value:.2f}")


    # 3) Extract variable values
    result_vars = {}
    var_values = {}
    problem_vars = problem_dict['variables']
    
    # Helper to safely get variable value
    def get_val(var):
        try:
            return float(var.value) if var.value is not None else 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0 # Default to 0 if value is invalid or missing

    # Extract build variables
    for k, var in problem_vars['gen_build'].items():
        var_values[k] = get_val(var)
        result_vars[f"gen_build_{k[0]}_{k[1]}"] = var_values[k]
    for k, var in problem_vars['storage_build'].items():
        var_values[k] = get_val(var)
        result_vars[f"storage_build_{k[0]}_{k[1]}"] = var_values[k]

    # Extract installed variables
    for k, var in problem_vars['gen_installed'].items():
        var_values[k] = get_val(var)
        result_vars[f"gen_installed_{k[0]}_{k[1]}"] = var_values[k]
    for k, var in problem_vars['storage_installed'].items():
        var_values[k] = get_val(var)
        result_vars[f"storage_installed_{k[0]}_{k[1]}"] = var_values[k]

    # Extract seasonal dispatch variables
    season_vars_dict = problem_vars['season_variables']
    for season, svars in season_vars_dict.items():
        net = integrated_network.season_networks[season]
        T = net.T
        
        for var_type in ['p_gen', 'p_line', 'p_charge', 'p_discharge', 'soc']:
            if var_type in svars:
                for key, var_vec in svars[var_type].items(): # key is (asset_id, year)
                    # Extract values safely
                    try:
                        vals = var_vec.value
                        if vals is None:
                            vals = np.zeros(T)
                        elif not isinstance(vals, np.ndarray):
                            vals = np.full(T, float(vals)) # Handle scalar promotion
                        elif len(vals) != T:
                             logger.warning(f"Length mismatch for {var_type} {key} season {season}. Expected {T}, got {len(vals)}. Padding/truncating.")
                             # Attempt to fix length mismatch, e.g., pad with zeros or truncate
                             if len(vals) < T:
                                 vals = np.pad(vals, (0, T - len(vals)), 'constant')
                             else:
                                 vals = vals[:T]
                    except (AttributeError, TypeError, ValueError):
                         logger.warning(f"Could not get value for {var_type} {key} season {season}. Defaulting to zeros.")
                         vals = np.zeros(T)

                    # Store individual time step values
                    asset_id, year = key
                    for t in range(T):
                        result_vars[f"{var_type}_{season}_{asset_id}_{year}_{t+1}"] = float(vals[t])
            else:
                logger.warning(f"Variable type '{var_type}' not found in season '{season}'")

    # 4) Save solution
    integrated_network.integrated_results = {
        'status': status,
        'value': objective_value,
        'success': True,
        'variables': result_vars
    }
    
    # Log summary using extracted values from result_vars
    num_gen_built = sum(1 for k, v in result_vars.items() if k.startswith('gen_build_') and v > 0.5)
    num_stor_built = sum(1 for k, v in result_vars.items() if k.startswith('storage_build_') and v > 0.5)
    logger.info(f"Total generator build decisions: {num_gen_built}")
    logger.info(f"Total storage build decisions: {num_stor_built}")

    # Detailed installation analysis using 'installed' variables
    logger.info("Analyzing installation patterns (based on 'installed' status):")
    gen_installations = {}
    storage_installations = {}
    
    # Use result_vars which contains the extracted float values
    for k, v in result_vars.items():
        if v > 0.5: # Use a threshold for binary check
            if k.startswith('gen_installed_'):
                parts = k.split('_')
                if len(parts) == 4: # Check format: gen_installed_ID_YEAR
                   try:
                       g = parts[2]
                       y = int(parts[3])
                       if g not in gen_installations: gen_installations[g] = []
                       gen_installations[g].append(y)
                   except (IndexError, ValueError):
                       logger.warning(f"Could not parse key: {k}")
            elif k.startswith('storage_installed_'):
                parts = k.split('_')
                if len(parts) == 4:
                    try:
                        s = parts[2]
                        y = int(parts[3])
                        if s not in storage_installations: storage_installations[s] = []
                        storage_installations[s].append(y)
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse key: {k}")
    
    # Function to get lifetime safely
    def get_lifetime(asset_id, asset_type):
        df = first_network.generators if asset_type == 'gen' else first_network.storage_units
        try:
            # Handle potential integer/string mismatch for index lookup
            if asset_id in df.index:
                return df.at[asset_id, 'lifetime_years']
            elif str(asset_id) in df.index:
                 return df.at[str(asset_id), 'lifetime_years']
            elif int(asset_id) in df.index:
                 return df.at[int(asset_id), 'lifetime_years']
            else:
                return "unknown"
        except (KeyError, ValueError, TypeError):
            return "unknown"

    logger.info("Generator installation patterns:")
    for g, years_list in gen_installations.items():
        lifetime = get_lifetime(g, 'gen')
        years_str = ", ".join(str(y) for y in sorted(list(set(years_list)))) # Ensure unique and sorted
        logger.info(f"  Generator {g} (lifetime={lifetime}): installed in years [{years_str}]")
        
    logger.info("Storage installation patterns:")
    for s, years_list in storage_installations.items():
        lifetime = get_lifetime(s, 'storage')
        years_str = ", ".join(str(y) for y in sorted(list(set(years_list)))) # Ensure unique and sorted
        logger.info(f"  Storage {s} (lifetime={lifetime}): installed in years [{years_str}]")

    return integrated_network.integrated_results