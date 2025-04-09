"""
Optimization module for DC Optimal Power Flow
This module handles the mathematical optimization aspects of the model
"""
import cvxpy as cp
import pandas as pd
import numpy as np

def create_dcopf_problem(network, years=None):
    """
    Create a DC optimal power flow problem for a network
    
    Args:
        network: Network object
        years: List of planning years (optional)
        
    Returns:
        Dictionary with problem data
    """
    # Debug information - log the column names
    print("DEBUG: Network column names:")
    print(f"  Buses columns: {list(network.buses.columns)}")
    print(f"  Generators columns: {list(network.generators.columns)}")
    print(f"  Loads columns: {list(network.loads.columns)}")
    print(f"  Lines/Branches columns: {list(network.lines.columns)}")
    if not network.storage_units.empty:
        print(f"  Storage columns: {list(network.storage_units.columns)}")
    
    # If years not specified, default to a single year
    if years is None:
        years = [1]
        print("No planning years specified, defaulting to a single year")
    
    print(f"Planning horizon: {len(years)} years")
    
    T = network.T  # Number of time periods
    
    # Extract network data
    buses = network.buses
    gens = network.generators
    loads = network.loads
    lines = network.lines
    storage = network.storage_units
    
    # Initialize costs lists
    operational_costs = []
    capital_costs = []
    
    # Create optimization variables
    # Generator power output for each generator, year, and time period
    p_gen = {(g, y): cp.Variable(network.T, nonneg=True)
             for g in gens.index for y in years}
    
    # Bus voltage angle for each bus, year, and time period
    theta = {(b, y): cp.Variable(network.T)
             for b in buses.index for y in years}
    
    # Storage variables for each storage unit, year, and time period
    p_charge = {}
    p_discharge = {}
    soc = {}
    
    # Only create storage variables if we have storage units
    if not storage.empty:
        p_charge = {(s, y): cp.Variable(network.T, nonneg=True)
                    for s in storage.index for y in years}
        p_discharge = {(s, y): cp.Variable(network.T, nonneg=True)
                       for s in storage.index for y in years}
        soc = {(s, y): cp.Variable(network.T, nonneg=True)
               for s in storage.index for y in years}
    
    # Line flow variables
    p_line = {(l, y): cp.Variable(network.T)
              for l in network.lines.index for y in years}
    
    # Binary variables for generator and storage installation decisions
    gen_installed = {(g, y): cp.Variable(boolean=True)
                     for g in gens.index for y in years}
    
    storage_installed = {(s, y): cp.Variable(boolean=True)
                         for s in storage.index for y in years}
    
    # Variables for tracking first installation
    gen_first_install = {(g, y): cp.Variable(boolean=True)
                         for g in gens.index for y in years}
    
    storage_first_install = {(s, y): cp.Variable(boolean=True)
                             for s in storage.index for y in years}
    
    # Variables for replacement decisions
    gen_replacement = {(g, y): cp.Variable(boolean=True)
                      for g in gens.index for y in years}
    
    storage_replacement = {(s, y): cp.Variable(boolean=True)
                          for s in storage.index for y in years}
    
    # Constraints
    constraints = []
    
    # Power balance constraints for each bus, year, and time period
    # Add slack variables to make constraints more flexible
    slack_pos = {(b, y, t): cp.Variable(pos=True) 
                for b in buses.index for y in years for t in range(T)}
    slack_neg = {(b, y, t): cp.Variable(pos=True) 
                for b in buses.index for y in years for t in range(T)}
    
    # Large penalty for using slack variables
    slack_penalty = 10000
    
    for b in buses.index:
        for y in years:
            for t in range(T):
                # Generators at this bus
                gens_at_bus = [g for g in gens.index if gens.loc[g, 'bus'] == b]
                gen_output = sum(p_gen[(g, y)][t] for g in gens_at_bus)
                
                # Loads at this bus
                loads_at_bus = [l for l in loads.index if loads.loc[l, 'bus'] == b]
                load_demand = 0
                for l in loads_at_bus:
                    if l in network.loads_t['p']:
                        load_series = network.loads_t['p'][l]
                        load_demand += load_series.iloc[t] if isinstance(load_series, pd.Series) else load_series[t]
                
                # Storage at this bus
                storage_at_bus = [s for s in storage.index if storage.loc[s, 'bus'] == b]
                storage_net = 0
                if storage_at_bus:
                    storage_net = sum(p_discharge[(s, y)][t] - p_charge[(s, y)][t] for s in storage_at_bus)
                
                # Lines connected to this bus
                lines_from = [l for l in lines.index if lines.loc[l, 'from_bus'] == b]
                lines_to = [l for l in lines.index if lines.loc[l, 'to_bus'] == b]
                
                # Power flow out of the bus (positive means power leaves the bus)
                power_flow_out = sum(p_line[(l, y)][t] for l in lines_from)
                
                # Power flow into the bus (positive means power enters the bus)
                power_flow_in = sum(p_line[(l, y)][t] for l in lines_to)
                
                # Power balance constraint with slack variables: 
                # generation + storage discharge - storage charge - slack_pos + slack_neg = load + net exports
                constraints.append(gen_output + storage_net - slack_pos[(b, y, t)] + slack_neg[(b, y, t)] == 
                                  load_demand + power_flow_out - power_flow_in)
                
                # Add slack variables to objective function with high penalty
                operational_costs.append(slack_penalty * slack_pos[(b, y, t)])
                operational_costs.append(slack_penalty * slack_neg[(b, y, t)])
    
    # Generator capacity constraints
    for g in gens.index:
        gen_data = gens.loc[g]
        # Use p_nom instead of capacity_mw
        capacity = gen_data['p_nom']
        
        for y in years:
            for t in range(T):
                # Generator output must be less than its capacity times its availability factor
                # and the installation status
                max_output = capacity
                if hasattr(network, 'gen_p_max_pu') and not network.gen_p_max_pu.empty and g in network.gen_p_max_pu:
                    max_output = capacity * network.gen_p_max_pu[g][t]
                
                constraints.append(p_gen[(g, y)][t] <= max_output * gen_installed[(g, y)])
    
    # Line capacity constraints
    for l in lines.index:
        line_data = lines.loc[l]
        # Use s_nom instead of capacity_mw
        capacity = line_data['s_nom']
        
        for y in years:
            for t in range(T):
                # Line flow must be less than its capacity
                constraints.append(p_line[(l, y)][t] <= capacity)
                constraints.append(p_line[(l, y)][t] >= -capacity)
    
    # Line flow constraints using PTDF or DC power flow equations
    for l in lines.index:
        line_data = lines.loc[l]
        from_bus = line_data['from_bus']
        to_bus = line_data['to_bus']
        reactance = line_data['x']
        
        for y in years:
            for t in range(T):
                # Power flow equation: p = (theta_from - theta_to) / x
                constraints.append(p_line[(l, y)][t] == (theta[(from_bus, y)][t] - theta[(to_bus, y)][t]) / reactance)
    
    # Storage constraints
    if not storage.empty:
        for s in storage.index:
            storage_data = storage.loc[s]
            # Use p_nom instead of capacity_mw and calculate energy_mwh from max_hours
            power_capacity = storage_data['p_nom']
            energy_capacity = storage_data['max_hours'] * power_capacity
            charge_eff = storage_data['efficiency_store']
            discharge_eff = storage_data['efficiency_dispatch']
            
            for y in years:
                for t in range(T):
                    # Charging and discharging power constraints
                    constraints.append(p_charge[(s, y)][t] <= power_capacity * storage_installed[(s, y)])
                    constraints.append(p_discharge[(s, y)][t] <= power_capacity * storage_installed[(s, y)])
                    
                    # Storage energy state constraint
                    if t == 0:
                        # Initial state of charge (assume half full)
                        constraints.append(soc[(s, y)][t] == 0.5 * energy_capacity * storage_installed[(s, y)] 
                                           + charge_eff * p_charge[(s, y)][t] - p_discharge[(s, y)][t] / discharge_eff)
                    else:
                        # State of charge update
                        constraints.append(soc[(s, y)][t] == soc[(s, y)][t-1] 
                                           + charge_eff * p_charge[(s, y)][t] - p_discharge[(s, y)][t] / discharge_eff)
                    
                    # Storage capacity constraint
                    constraints.append(soc[(s, y)][t] <= energy_capacity * storage_installed[(s, y)])
                
                # End state equal to initial state (cycle constraint)
                constraints.append(soc[(s, y)][T-1] == soc[(s, y)][0])
    
    # Reference bus constraint (set one bus angle to zero)
    ref_bus = buses.index[0]  # Use the first bus as reference
    for y in years:
        for t in range(T):
            constraints.append(theta[(ref_bus, y)][t] == 0)
    
    # Objective function - minimize total cost
    total_cost = cp.sum(operational_costs) + cp.sum(capital_costs)
    
    # Create the optimization problem
    problem = {
        'objective': cp.Minimize(total_cost),
        'constraints': constraints,
        'variables': {
            'p_gen': p_gen,
            'theta': theta,
            'p_line': p_line,
            'gen_installed': gen_installed,
            'storage_installed': storage_installed,
            'gen_first_install': gen_first_install,
            'storage_first_install': storage_first_install,
            'gen_replacement': gen_replacement,
            'storage_replacement': storage_replacement
        }
    }
    
    # Add storage variables if any
    if not storage.empty:
        problem['variables']['p_charge'] = p_charge
        problem['variables']['p_discharge'] = p_discharge
        problem['variables']['soc'] = soc
    
    return problem

def solve_with_cplex(problem):
    """
    Solve the optimization problem with CPLEX
    
    Args:
        problem: Dictionary with variables, constraints, and objective
        
    Returns:
        Dictionary with solution status and value
    """
    # Create the problem
    prob = cp.Problem(problem['objective'], problem['constraints'])
    
    # Set CPLEX parameters to show iterations
    cplex_params = {
        'threads': 10,         # Use 10 threads for parallel processing
        # Removing all parameters with slashes that were causing errors
        # 'display': 2,        # This parameter is not supported and causes an error
        # 'barrier/display': 1,  # Not supported - causes error
        # 'simplex/display': 2,  # Not supported - causes error
        # 'tune/display': 3,     # Not supported - causes error
        # 'mip/display': 5,      # Not supported - causes error
        # 'mip/interval': 1,     # Not supported - causes error
        # 'barrier/convergetol': 1e-8,  # Not supported - causes error
        'timelimit': 3600     # Maximum time limit in seconds (1 hour)
    }
    
    try:
        # Use CPLEX with verbose output to show iterations
        print("Starting optimization with CPLEX...")
        print("Solving optimization problem... (this may take several minutes)")
        
        # Solve with CPLEX and show iterations
        prob.solve(solver=cp.CPLEX, verbose=True, cplex_params=cplex_params)
        
        print(f"Optimization status: {prob.status}")
        if prob.status in ["optimal", "optimal_inaccurate"]:
            print(f"Objective value: {prob.value:.2f}")
    except Exception as e:
        print(f"CPLEX optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'success': False, 'value': None}
        
    # Check if solution is optimal
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return {'status': prob.status, 'success': False, 'value': None}
    
    # Return the solution status and value
    return {
        'status': prob.status,
        'success': True,
        'value': prob.value
    }

def extract_results(network, problem, solution):
    """
    Extract results from the solved problem and store them in the network
    
    Args:
        network: Network object
        problem: Dictionary with variables, constraints, and objective
        solution: Dictionary with solution status and value
        
    Returns:
        None (results are stored in the network object)
    """
    # Extract variables from the problem
    p_gen = problem['variables']['p_gen']
    p_charge = problem['variables']['p_charge']
    p_discharge = problem['variables']['p_discharge']
    soc = problem['variables']['soc']
    p_line = problem['variables']['p_line']
    theta = problem['variables']['theta']
    gen_installed = problem['variables']['gen_installed']
    storage_installed = problem['variables']['storage_installed']
    gen_first_install = problem['variables']['gen_first_install']
    storage_first_install = problem['variables']['storage_first_install']
    
    # Safely get replacement variables, which might be None for single-year problems
    gen_replacement = problem['variables'].get('gen_replacement')
    storage_replacement = problem['variables'].get('storage_replacement')
    
    # If we only have one year, these variables might not exist
    if len(problem.get('years', [0])) <= 1:
        gen_replacement = None
        storage_replacement = None
    
    years = problem.get('years', [1])  # Default to a single year with index 1 since that's what we found in the keys
    
    # Print debugging information
    print("DEBUG - Keys in p_gen:", list(p_gen.keys())[:5])  # Print first 5 keys
    print("DEBUG - Generator indices:", list(network.generators.index))
    print("DEBUG - Year in extract_results:", years)
    
    # Map network years to problem years (the problem might use different year indices)
    # This maps from what the network expects to what the problem actually has
    year_map = {}
    for year in years:
        # Try to find this year in the keys
        sample_gen = list(network.generators.index)[0] if not network.generators.empty else None
        if sample_gen:
            for possible_year in [1, 0, year]:  # Try common values
                if (sample_gen, possible_year) in p_gen:
                    year_map[year] = possible_year
                    break
            
            # If we haven't found a match, use the first year from any key
            if year not in year_map and len(list(p_gen.keys())) > 0:
                first_key = list(p_gen.keys())[0]
                year_map[year] = first_key[1]  # Use the year from the first key
    
    print("DEBUG - Year mapping:", year_map)
    
    # Create containers for time series results by year
    network.generators_t_by_year = {year: {'p': pd.DataFrame(index=range(network.T))} for year in years}
    network.storage_units_t_by_year = {
        year: {
            'p_charge': pd.DataFrame(index=range(network.T)),
            'p_discharge': pd.DataFrame(index=range(network.T)),
            'state_of_charge': pd.DataFrame(index=range(network.T))
        } for year in years
    }
    network.lines_t_by_year = {year: {'p': pd.DataFrame(index=range(network.T))} for year in years}
    network.buses_t_by_year = {year: {'v_ang': pd.DataFrame(index=range(network.T))} for year in years}
    
    # Store installation decisions by year
    network.generators_installed_by_year = {year: {} for year in years}
    network.storage_installed_by_year = {year: {} for year in years}
    network.generators_first_install_by_year = {year: {} for year in years}
    network.storage_first_install_by_year = {year: {} for year in years}
    
    # Store replacement information if available
    if gen_replacement is not None:
        network.generators_replacement_by_year = {year: {} for year in years}
        for year in years:
            for g in network.generators.index:
                network.generators_replacement_by_year[year][g] = gen_replacement[(g, year)].value
    else:
        # Initialize empty to avoid attribute errors
        network.generators_replacement_by_year = {year: {g: 0.0 for g in network.generators.index} for year in years}
    
    if storage_replacement is not None:
        network.storage_replacement_by_year = {year: {} for year in years}
        for year in years:
            for s in network.storage_units.index:
                network.storage_replacement_by_year[year][s] = storage_replacement[(s, year)].value
    else:
        # Initialize empty to avoid attribute errors
        network.storage_replacement_by_year = {year: {s: 0.0 for s in network.storage_units.index} for year in years}
    
    # Create dictionaries to store asset installation history
    network.asset_installation_history = {
        'generators': {},
        'storage': {}
    }
    
    # Extract and store results for each year
    for year in years:
        # Map to the actual year used in the keys
        actual_year = year_map.get(year, year)
        
        # Extract generator results
        for g in network.generators.index:
            if (g, actual_year) in p_gen:
                network.generators_t_by_year[year]['p'][g] = p_gen[(g, actual_year)].value
                network.generators_installed_by_year[year][g] = gen_installed[(g, actual_year)].value
                network.generators_first_install_by_year[year][g] = gen_first_install[(g, actual_year)].value
                
                # Determine if this is a replacement
                is_replacement = False
                if hasattr(network, 'generators_replacement_by_year') and year in network.generators_replacement_by_year:
                    if g in network.generators_replacement_by_year[year]:
                        replacement_val = network.generators_replacement_by_year[year][g]
                        is_replacement = replacement_val is not None and replacement_val > 0.5
                
                # Add to installation history if first installed in this year
                if gen_first_install[(g, actual_year)].value is not None and gen_first_install[(g, actual_year)].value > 0.5:
                    if g not in network.asset_installation_history['generators']:
                        network.asset_installation_history['generators'][g] = []
                    
                    network.asset_installation_history['generators'][g].append({
                        'installation_year': year,
                        'capacity_mw': network.generators.loc[g, 'p_nom'],
                        'capex_per_mw': network.generators.loc[g].get('capex_per_mw', 0),
                        'lifetime_years': network.generators.loc[g].get('lifetime_years', 20),
                        'is_replacement': is_replacement
                    })
            else:
                print(f"WARNING: Generator {g}, year {actual_year} not found in p_gen keys")
                network.generators_t_by_year[year]['p'][g] = 0.0
                network.generators_installed_by_year[year][g] = 0.0
                network.generators_first_install_by_year[year][g] = 0.0
        
        # Storage units
        for s in network.storage_units.index:
            if (s, actual_year) in p_charge:
                network.storage_units_t_by_year[year]['p_charge'][s] = p_charge[(s, actual_year)].value
                network.storage_units_t_by_year[year]['p_discharge'][s] = p_discharge[(s, actual_year)].value
                network.storage_units_t_by_year[year]['state_of_charge'][s] = soc[(s, actual_year)].value
                network.storage_installed_by_year[year][s] = storage_installed[(s, actual_year)].value
                network.storage_first_install_by_year[year][s] = storage_first_install[(s, actual_year)].value
                
                # Store replacement information if available
                if storage_replacement is not None:
                    network.storage_replacement_by_year[year][s] = storage_replacement[(s, actual_year)].value
                
                # Determine if this is a replacement
                is_replacement = False
                if hasattr(network, 'storage_replacement_by_year') and year in network.storage_replacement_by_year:
                    if s in network.storage_replacement_by_year[year]:
                        replacement_val = network.storage_replacement_by_year[year][s]
                        is_replacement = replacement_val is not None and replacement_val > 0.5
                
                # Add to installation history if first installed in this year
                if storage_first_install[(s, actual_year)].value is not None and storage_first_install[(s, actual_year)].value > 0.5:
                    if s not in network.asset_installation_history['storage']:
                        network.asset_installation_history['storage'][s] = []
                    
                    network.asset_installation_history['storage'][s].append({
                        'installation_year': year,
                        'capacity_mw': network.storage_units.loc[s, 'p_nom'],
                        'energy_capacity_mwh': network.storage_units.loc[s, 'max_hours'] * network.storage_units.loc[s, 'p_nom'],
                        'capex_per_mw': network.storage_units.loc[s].get('capex_per_mw', 0),
                        'lifetime_years': network.storage_units.loc[s].get('lifetime_years', 10),
                        'is_replacement': is_replacement
                    })
            else:
                print(f"WARNING: Storage {s}, year {actual_year} not found in p_charge keys")
                network.storage_units_t_by_year[year]['p_charge'][s] = 0.0
                network.storage_units_t_by_year[year]['p_discharge'][s] = 0.0
                network.storage_units_t_by_year[year]['state_of_charge'][s] = 0.0
                network.storage_installed_by_year[year][s] = 0.0
                network.storage_first_install_by_year[year][s] = 0.0
                
                if storage_replacement is not None:
                    network.storage_replacement_by_year[year][s] = 0.0
        
        # Lines
        for l in network.lines.index:
            if (l, actual_year) in p_line:
                network.lines_t_by_year[year]['p'][l] = p_line[(l, actual_year)].value
            else:
                print(f"WARNING: Line {l}, year {actual_year} not found in p_line keys")
                network.lines_t_by_year[year]['p'][l] = 0.0
        
        # Bus voltage angles
        for b in network.buses.index:
            if (b, actual_year) in theta:
                network.buses_t_by_year[year]['v_ang'][b] = theta[(b, actual_year)].value
            else:
                print(f"WARNING: Bus {b}, year {actual_year} not found in theta keys")
                network.buses_t_by_year[year]['v_ang'][b] = 0.0
    
    # Also populate the regular result containers with the last year's results
    # (for backward compatibility)
    last_year = years[-1]
    network.generators_t = {'p': network.generators_t_by_year[last_year]['p'].copy()}
    network.storage_units_t = {
        'p_charge': network.storage_units_t_by_year[last_year]['p_charge'].copy(),
        'p_discharge': network.storage_units_t_by_year[last_year]['p_discharge'].copy(),
        'state_of_charge': network.storage_units_t_by_year[last_year]['state_of_charge'].copy()
    }
    network.lines_t = {'p': network.lines_t_by_year[last_year]['p'].copy()}
    network.buses_t = {'v_ang': network.buses_t_by_year[last_year]['v_ang'].copy()}
    
    network.generators_installed = network.generators_installed_by_year[last_year]
    network.storage_installed = network.storage_installed_by_year[last_year]
    
    # Store objective value
    network.objective_value = solution['value']
    
    # Print a simplified summary of the optimization results
    if len(years) > 1:
        # Count installed assets in final year
        installed_gens = sum(1 for g, val in network.generators_installed_by_year[last_year].items() if val > 0.5)
        installed_storage = sum(1 for s, val in network.storage_installed_by_year[last_year].items() if val > 0.5)
        
        print(f"Multi-year results: Objective: {solution['value']:.2f}, Final year: {installed_gens} generators, {installed_storage} storage units")
    else:
        print(f"Single-year results: Objective: {solution['value']:.2f}")

def create_integrated_dcopf_problem(integrated_network):
    """
    Create an integrated DC Optimal Power Flow problem across all seasons
    
    This function takes an integrated network and creates an optimization
    problem that links all seasonal submodels with common asset decisions.
    
    Args:
        integrated_network: IntegratedNetwork object
        
    Returns:
        Dictionary with variables, constraints, objective and metadata
    """
    print("Creating integrated optimization problem...")
    
    # Extract years and seasons
    years = integrated_network.years
    seasons = list(integrated_network.season_networks.keys())
    
    # Initialize containers for variables and constraints
    season_variables = {season: {} for season in seasons}
    season_constraints = {season: [] for season in seasons}
    
    # Create variables for each season and each year
    print(f"Creating variables for {len(seasons)} seasons × {len(years)} years")
    
    # First, collect all asset indices from the first seasonal network
    # (all networks should have the same components)
    first_network = list(integrated_network.season_networks.values())[0]
    generators = first_network.generators.index
    storage_units = first_network.storage_units.index
    buses = first_network.buses.index
    lines = first_network.lines.index
    
    # Global variables shared across all seasons
    print("Creating global asset decision variables...")
    
    # Generator installation binary variables (global across seasons)
    gen_installed = {(g, y): cp.Variable(boolean=True) 
                    for g in generators for y in years}
    
    # Storage installation binary variables (global across seasons)
    storage_installed = {(s, y): cp.Variable(boolean=True)
                        for s in storage_units for y in years}
    
    # Global first installation variables
    gen_first_install = {}
    storage_first_install = {}
    
    # Replacement variables if multiple years
    gen_replacement = {}
    storage_replacement = {}
    
    # If multi-year, track when assets are first installed vs. replacement
    if len(years) > 1:
        # Generator first installation variables
        gen_first_install = {(g, y): cp.Variable(boolean=True) 
                            for g in generators for y in years}
        
        # Storage first installation variables
        storage_first_install = {(s, y): cp.Variable(boolean=True)
                                for s in storage_units for y in years}
        
        # Replacement variables (when asset is replaced before end of life)
        gen_replacement = {(g, y): cp.Variable(boolean=True)
                          for g in generators for y in years}
        
        storage_replacement = {(s, y): cp.Variable(boolean=True)
                              for s in storage_units for y in years}
    
    # Global constraints for asset continuity between years
    global_constraints = []
    
    # First installation happens when asset goes from not installed to installed
    if len(years) > 1:
        for g in generators:
            # First year is a special case
            global_constraints.append(gen_first_install[(g, years[0])] == gen_installed[(g, years[0])])
            
            # For subsequent years, first installed if it wasn't installed before but is now
            for y_idx in range(1, len(years)):
                global_constraints.append(
                    gen_first_install[(g, years[y_idx])] == (
                        (gen_installed[(g, years[y_idx])] - gen_installed[(g, years[y_idx-1])]) + gen_replacement[(g, years[y_idx])]
                    )
                )
        
        # Same for storage
        for s in storage_units:
            # First year is a special case
            global_constraints.append(storage_first_install[(s, years[0])] == storage_installed[(s, years[0])])
            
            # For subsequent years, first installed if it wasn't installed before but is now
            for y_idx in range(1, len(years)):
                global_constraints.append(
                    storage_first_install[(s, years[y_idx])] == (
                        (storage_installed[(s, years[y_idx])] - storage_installed[(s, years[y_idx-1])]) + storage_replacement[(s, years[y_idx])]
                    )
                )
    else:
        # If only one year, installed is same as first install
        gen_first_install = gen_installed
        storage_first_install = storage_installed
    
    # Season-specific variables
    # These operational variables are specific to each season but constrained by global asset decisions
    for season in seasons:
        print(f"Creating variables for {season} season...")
        network = integrated_network.season_networks[season]
        T = network.T  # Time steps in this season
        
        # Generator dispatch variables
        season_variables[season]['p_gen'] = {(g, y): cp.Variable(T, nonneg=True) 
                                          for g in generators for y in years}
        
        # Storage variables
        season_variables[season]['p_charge'] = {(s, y): cp.Variable(T, nonneg=True)
                                              for s in storage_units for y in years}
        season_variables[season]['p_discharge'] = {(s, y): cp.Variable(T, nonneg=True)
                                                 for s in storage_units for y in years}
        season_variables[season]['soc'] = {(s, y): cp.Variable(T, nonneg=True)
                                         for s in storage_units for y in years}
        
        # Bus voltage angle variables
        season_variables[season]['theta'] = {(b, y): cp.Variable(T) 
                                           for b in buses for y in years}
        
        # Line flow variables
        season_variables[season]['p_line'] = {(l, y): cp.Variable(T)
                                       for l in lines for y in years}
        
        # Set reference bus for each year
        if not network.buses.empty:
            ref_bus = network.buses.index[0]
            for y in years:
                season_constraints[season].append(season_variables[season]['theta'][(ref_bus, y)] == 0)
        
        # Generator capacity constraints linked to global installation decisions
        for y in years:
            for gen_id, gen_data in network.generators.iterrows():
                capacity = gen_data['p_nom']
                
                if hasattr(network, 'gen_p_max_pu') and gen_id in network.gen_p_max_pu:
                    # Time-dependent constraint
                    for t in range(T):
                        max_capacity = capacity * network.gen_p_max_pu[gen_id][t]
                        # Use global gen_installed variable here!
                        season_constraints[season].append(
                            season_variables[season]['p_gen'][(gen_id, y)][t] <= max_capacity * gen_installed[(gen_id, y)]
                        )
                else:
                    # Static capacity constraint
                    # Use global gen_installed variable here!
                    season_constraints[season].append(
                        season_variables[season]['p_gen'][(gen_id, y)] <= capacity * gen_installed[(gen_id, y)]
                    )
        
        # Storage constraints linked to global installation decisions
        for y in years:
            for storage_id, storage_data in network.storage_units.iterrows():
                # Power limits - use global storage_installed variable!
                season_constraints[season].append(
                    season_variables[season]['p_charge'][(storage_id, y)] <= 
                    storage_data['p_nom'] * storage_installed[(storage_id, y)]
                )
                season_constraints[season].append(
                    season_variables[season]['p_discharge'][(storage_id, y)] <= 
                    storage_data['p_nom'] * storage_installed[(storage_id, y)]
                )
                
                # Energy capacity limits - use global storage_installed variable!
                season_constraints[season].append(
                    season_variables[season]['soc'][(storage_id, y)] <= 
                    storage_data['max_hours'] * storage_data['p_nom'] * storage_installed[(storage_id, y)]
                )
                
                # Storage energy balance with initial SoC at 50% (for now)
                soc_init = 0.5 * storage_data['max_hours'] * storage_data['p_nom']
                for t in range(T):
                    if t == 0:
                        season_constraints[season].append(
                            season_variables[season]['soc'][(storage_id, y)][t] == 
                            soc_init * storage_installed[(storage_id, y)] 
                            + storage_data['efficiency_store'] * season_variables[season]['p_charge'][(storage_id, y)][t]
                            - (1 / storage_data['efficiency_dispatch']) * season_variables[season]['p_discharge'][(storage_id, y)][t]
                        )
                    else:
                        season_constraints[season].append(
                            season_variables[season]['soc'][(storage_id, y)][t] == 
                            season_variables[season]['soc'][(storage_id, y)][t-1] 
                            + storage_data['efficiency_store'] * season_variables[season]['p_charge'][(storage_id, y)][t]
                            - (1 / storage_data['efficiency_dispatch']) * season_variables[season]['p_discharge'][(storage_id, y)][t]
                        )
        
        # DC power flow constraints for each line
        for y in years:
            for line_id, line_data in network.lines.iterrows():
                from_bus = line_data['from_bus']
                to_bus = line_data['to_bus']
                susceptance = line_data['susceptance']
                capacity = line_data['s_nom']
                
                # Flow equation based on susceptance and voltage angle difference
                for t in range(T):
                    season_constraints[season].append(
                        season_variables[season]['p_line'][(line_id, y)][t] == 
                        susceptance * (
                            season_variables[season]['theta'][(from_bus, y)][t] - 
                            season_variables[season]['theta'][(to_bus, y)][t]
                        )
                    )
                
                # Transmission line flow limits
                season_constraints[season].append(
                    season_variables[season]['p_line'][(line_id, y)] <= capacity
                )
                season_constraints[season].append(
                    season_variables[season]['p_line'][(line_id, y)] >= -capacity
                )
        
        # Pre-calculate load values per bus for this season
        bus_load = {}
        for bus_id in network.buses.index:
            bus_load[bus_id] = np.zeros(T)
        
        # Map loads to buses
        load_to_bus_map = {}
        for load_id in network.loads.index:
            load_to_bus_map[load_id] = network.loads.at[load_id, 'bus']
        
        # Now map loads to buses using the mapping
        for t in range(T):
            for load_id in network.loads.index:
                if 'p' in network.loads_t and load_id in network.loads_t['p']:
                    load_ts = network.loads_t['p'][load_id]
                    if t < len(load_ts):
                        bus_id = load_to_bus_map.get(load_id)
                        if bus_id in bus_load:
                            bus_load[bus_id][t] += load_ts.iloc[t] if isinstance(load_ts, pd.Series) else load_ts[t]
        
        # Nodal power balance constraints
        for y in years:
            # Get load growth factor for this year
            load_growth_factor = 1.0  # Default no growth
            if hasattr(network, 'year_to_load_factor') and y in network.year_to_load_factor:
                load_growth_factor = network.year_to_load_factor[y]
            
            for t in range(T):
                for bus_id in network.buses.index:
                    # Generator contribution
                    gen_at_bus = [
                        season_variables[season]['p_gen'][(g, y)][t] 
                        for g in generators 
                        if network.generators.loc[g, 'bus'] == bus_id
                    ]
                    gen_sum = sum(gen_at_bus) if gen_at_bus else 0
                    
                    # Storage contribution
                    storage_at_bus = [
                        season_variables[season]['p_discharge'][(s, y)][t] - 
                        season_variables[season]['p_charge'][(s, y)][t] 
                        for s in storage_units 
                        if network.storage_units.loc[s, 'bus'] == bus_id
                    ]
                    storage_net = sum(storage_at_bus) if storage_at_bus else 0
                    
                    # Load at the bus - apply load growth factor
                    load = bus_load[bus_id][t] * load_growth_factor
                    
                    # Line flows into and out of the bus
                    flow_out = sum(
                        season_variables[season]['p_line'][(l, y)][t] 
                        for l in lines 
                        if network.lines.loc[l, 'from_bus'] == bus_id
                    )
                    flow_in = sum(
                        season_variables[season]['p_line'][(l, y)][t] 
                        for l in lines 
                        if network.lines.loc[l, 'to_bus'] == bus_id
                    )
                    
                    # Power balance: generation + storage net + flow in = load + flow out
                    season_constraints[season].append(
                        gen_sum + storage_net + flow_in == load + flow_out
                    )
    
    # Inter-seasonal storage coupling constraints
    print("Creating inter-seasonal storage coupling constraints...")
    interseasonal_constraints = []
    
    # Define the sequence of seasons
    season_sequence = ['winter', 'summer', 'spri_autu']
    
    # For each year and storage unit, link end of one season to start of next
    for y in years:
        for s in storage_units:
            # Winter end → Summer start (within same year)
            if 'winter' in seasons and 'summer' in seasons:
                winter_end_idx = integrated_network.season_networks['winter'].T - 1
                interseasonal_constraints.append(
                    season_variables['winter']['soc'][(s, y)][winter_end_idx] == 
                    season_variables['summer']['soc'][(s, y)][0]
                )
            
            # Summer end → Spring/Autumn start (within same year)
            if 'summer' in seasons and 'spri_autu' in seasons:
                summer_end_idx = integrated_network.season_networks['summer'].T - 1
                interseasonal_constraints.append(
                    season_variables['summer']['soc'][(s, y)][summer_end_idx] == 
                    season_variables['spri_autu']['soc'][(s, y)][0]
                )
            
            # Spring/Autumn end → Winter start (next year, if not last year)
            if 'spri_autu' in seasons and 'winter' in seasons and y < years[-1]:
                spri_autu_end_idx = integrated_network.season_networks['spri_autu'].T - 1
                next_year_idx = years[years.index(y) + 1]
                interseasonal_constraints.append(
                    season_variables['spri_autu']['soc'][(s, y)][spri_autu_end_idx] == 
                    season_variables['winter']['soc'][(s, next_year_idx)][0]
                )
    
    # Objective function calculation - weighted sum of seasonal costs
    print("Creating integrated objective function...")
    
    season_costs = {}
    
    for season in seasons:
        network = integrated_network.season_networks[season]
        T = network.T
        season_weight = integrated_network.season_weights.get(season, 1.0) / 52.0  # Convert to fraction of year
        
        # Initialize operational and capital costs for this season
        operational_costs = []
        capital_costs = []
        
        # 1. Operational costs: generation cost per MWh * generation for this season
        for y in years:
            for gen_id in generators:
                cost_per_mwh = network.generators.loc[gen_id, 'marginal_cost']
                for t in range(T):
                    # Note: season_weight is applied to operational costs 
                    operational_costs.append(
                        season_weight * cost_per_mwh * season_variables[season]['p_gen'][(gen_id, y)][t]
                    )
        
        # 2. Calculate total cost for this season
        # We DON'T include capital costs here, as they are global across seasons
        season_costs[season] = sum(operational_costs)
    
    # Capital costs are global (not season-specific) so add them separately
    capital_costs = []
    
    # Generator CAPEX - simplified annual cost
    for y in years:
        for gen_id in generators:
            gen_data = first_network.generators.loc[gen_id]
            # Only count CAPEX when generator is first installed in a year
            capex = gen_data.get('capex_per_mw', 0) * gen_data['p_nom']
            
            # Get lifetime and calculate simple annual cost
            lifetime = gen_data.get('lifetime_years', 20)
            discount_rate = gen_data.get('discount_rate', 0.05)
            annual_capex = capex * (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)
            
            # Apply CAPEX only at first installation or replacement
            yearly_capex = annual_capex * gen_first_install[(gen_id, y)]
            capital_costs.append(yearly_capex)
    
    # Storage CAPEX - simplified annual cost
    for y in years:
        for storage_id in storage_units:
            storage_data = first_network.storage_units.loc[storage_id]
            
            # Calculate power capacity cost
            p_capex = storage_data.get('capex_per_mw', 0) * storage_data['p_nom']
            
            # Calculate energy capacity cost (estimate)
            e_capex = storage_data.get('capex_per_mwh', 0) * storage_data['max_hours'] * storage_data['p_nom']
            if e_capex == 0 and 'capex_per_mw' in storage_data:
                # If no specific energy cost, use power cost and typical hours
                typical_hours = storage_data['max_hours'] / storage_data['p_nom']
                e_capex = storage_data['capex_per_mw'] * typical_hours * storage_data['max_hours'] * storage_data['p_nom']
            
            # Total CAPEX is sum of power and energy components
            total_capex = p_capex + e_capex
            
            # Get lifetime and calculate simple annual cost
            lifetime = storage_data.get('lifetime_years', 10)
            annual_capex = total_capex / lifetime
            
            # Apply CAPEX only at first installation
            yearly_capex = annual_capex * storage_first_install[(storage_id, y)]
            capital_costs.append(yearly_capex)
    
    # Total objective: Sum of all season costs plus capital costs
    total_cost = sum(season_costs.values()) + sum(capital_costs)
    objective = cp.Minimize(total_cost)
    
    # Combine all constraints
    all_constraints = []
    all_constraints.extend(global_constraints)
    all_constraints.extend(interseasonal_constraints)
    for season in seasons:
        all_constraints.extend(season_constraints[season])
    
    # Create the integrated problem structure
    print(f"Problem created with {len(all_constraints)} constraints")
    
    return {
        'variables': {
            # Global variables
            'gen_installed': gen_installed,
            'storage_installed': storage_installed,
            'gen_first_install': gen_first_install,
            'storage_first_install': storage_first_install,
            'gen_replacement': gen_replacement,
            'storage_replacement': storage_replacement,
            
            # Season-specific variables
            'season_variables': season_variables
        },
        'constraints': all_constraints,
        'objective': objective,
        'integrated_model': True,
        'seasons': seasons,
        'years': years,
        'season_costs': season_costs
    }

def extract_integrated_results(integrated_network, problem, solution):
    """
    Extract results from the integrated optimization problem
    
    Args:
        integrated_network: IntegratedNetwork object
        problem: Dictionary with problem variables and constraints
        solution: Dictionary with solution status and value
        
    Returns:
        Dictionary with the extracted results
    """
    # Initialize the results dictionary
    results = {
        'value': solution['value'],
        'status': solution['status'],
        'generators_installed': {year: {} for year in integrated_network.years},
        'storage_installed': {year: {} for year in integrated_network.years},
        'generators_first_install': {year: {} for year in integrated_network.years},
        'storage_first_install': {year: {} for year in integrated_network.years},
        'generators_replacement': {year: {} for year in integrated_network.years},
        'storage_replacement': {year: {} for year in integrated_network.years},
        'generators_p': {season: {year: {} for year in integrated_network.years} for season in integrated_network.seasons},
        'storage_p_charge': {season: {year: {} for year in integrated_network.years} for season in integrated_network.seasons},
        'storage_p_discharge': {season: {year: {} for year in integrated_network.years} for season in integrated_network.seasons},
        'storage_soc': {season: {year: {} for year in integrated_network.years} for season in integrated_network.seasons},
        'season_costs': {season: 0.0 for season in integrated_network.seasons},
        'operational_costs': {season: 0.0 for season in integrated_network.seasons},
        'capital_costs': 0.0
    }
    
    # Get global variables
    gen_installed = problem['variables'].get('gen_installed', {})
    storage_installed = problem['variables'].get('storage_installed', {})
    gen_first_install = problem['variables'].get('gen_first_install', {})
    storage_first_install = problem['variables'].get('storage_first_install', {})
    gen_replacement = problem['variables'].get('gen_replacement', {})
    storage_replacement = problem['variables'].get('storage_replacement', {})
    
    # Extract installed assets
    for year in integrated_network.years:
        # Generator installation
        for g in gen_installed:
            if (g, year) in gen_installed:
                results['generators_installed'][year][g] = gen_installed[(g, year)].value
            
        # Storage installation
        for s in storage_installed:
            if (s, year) in storage_installed:
                results['storage_installed'][year][s] = storage_installed[(s, year)].value
        
        # First installation
        if gen_first_install:
            for g in gen_first_install:
                if (g, year) in gen_first_install:
                    results['generators_first_install'][year][g] = gen_first_install[(g, year)].value
        
        if storage_first_install:
            for s in storage_first_install:
                if (s, year) in storage_first_install:
                    results['storage_first_install'][year][s] = storage_first_install[(s, year)].value
        
        # Replacement
        if gen_replacement:
            for g in gen_replacement:
                if (g, year) in gen_replacement:
                    results['generators_replacement'][year][g] = gen_replacement[(g, year)].value
        
        if storage_replacement:
            for s in storage_replacement:
                if (s, year) in storage_replacement:
                    results['storage_replacement'][year][s] = storage_replacement[(s, year)].value
    
    # Get seasonal variables
    gen_p = problem['variables'].get('gen_p', {})
    storage_p_charge = problem['variables'].get('storage_p_charge', {})
    storage_p_discharge = problem['variables'].get('storage_p_discharge', {})
    storage_soc = problem['variables'].get('storage_soc', {})
    
    # Extract seasonal results
    for season in integrated_network.seasons:
        # Get seasonal operational cost
        season_cost = 0.0
        
        for year in integrated_network.years:
            # Generator output
            for g in gen_p:
                if (g, season, year) in gen_p:
                    results['generators_p'][season][year][g] = gen_p[(g, season, year)].value
                    
                    # Calculate marginal cost contribution to seasonal cost
                    network = integrated_network.get_season_network(season)
                    if g in network.generators.index:
                        marginal_cost = network.generators.loc[g, 'marginal_cost']
                        gen_output = sum(gen_p[(g, season, year)].value)
                        season_cost += marginal_cost * gen_output
            
            # Storage operations
            for s in storage_p_charge:
                if (s, season, year) in storage_p_charge:
                    results['storage_p_charge'][season][year][s] = storage_p_charge[(s, season, year)].value
                    results['storage_p_discharge'][season][year][s] = storage_p_discharge[(s, season, year)].value
                    results['storage_soc'][season][year][s] = storage_soc[(s, season, year)].value
        
        # Store seasonal cost
        results['season_costs'][season] = season_cost
        results['operational_costs'][season] = season_cost
    
    # Calculate capital costs (simplified - just add up all asset costs)
    capital_cost = 0.0
    
    # Extract capital costs from the objective function
    # This is a simplified approach; in a real model you would need to calculate based on CAPEX and lifetimes
    
    # Set the results in the integrated network
    integrated_network.set_optimization_results(results)
    
    # Return the results
    return results

def optimize_integrated_network(integrated_network, solver_options=None):
    """
    Run the integrated optimization across all seasons
    
    This optimizes all seasonal networks together with shared asset decisions.
    The integrated approach ensures that a single coherent set of assets serves
    all seasons and accounts for seasonal interactions, particularly for storage.
    
    Args:
        integrated_network: IntegratedNetwork object
        solver_options: Dictionary with solver options
        
    Returns:
        True if optimization successful, False otherwise
    """
    # Set default solver options if None
    solver_options = solver_options or {}
    
    print(f"Running integrated optimization across {len(integrated_network.seasons)} seasons...")
    print(f"Planning horizon: {len(integrated_network.years)} years")
    print(f"Solver options: {solver_options}")
    
    # Create the integrated problem
    problem = create_integrated_dcopf_problem(integrated_network)
    
    # Solve the problem
    solution = solve_with_cplex(problem)
    
    if not solution['success']:
        print("Integrated optimization failed!")
        return False
    
    # Extract results
    extract_integrated_results(integrated_network, problem, solution)
    
    print(f"Integrated optimization completed successfully. Objective value: {solution['value']:.2f}")
    return True

def optimize_seasonal_network(integrated_network, solver_options=None):
    """
    Run optimization for each season separately, but keep consistent asset decisions
    
    Args:
        integrated_network: IntegratedNetwork object
        solver_options: Dictionary with solver options
        
    Returns:
        True if all optimizations succeeded, False otherwise
    """
    # Set default solver options if None
    solver_options = solver_options or {}
    
    print(f"Running seasonal optimization across {len(integrated_network.seasons)} seasons...")
    
    # Dictionary to store results for each season
    season_results = {}
    season_costs = {}
    
    # Run optimization for each season
    for season, network in integrated_network.season_networks.items():
        print(f"\nOptimizing {season}...")
        
        # Create the DCOPF problem for this season
        problem = create_dcopf_problem(network)
        
        # Solve the problem
        solution = solve_with_cplex(problem)
        
        if not solution['success']:
            print(f"Optimization failed for {season}!")
            return False
        
        # Extract results
        extract_results(network, problem, solution)
        
        # Store the results for this season
        season_results[season] = {
            'objective_value': network.objective_value,
            'generators_p': {g: network.generators_t['p'][g] for g in network.generators.index},
            'storage_p_charge': {s: network.storage_units_t['p_charge'][s] for s in network.storage_units.index} if hasattr(network, 'storage_units_t') else {},
            'storage_p_discharge': {s: network.storage_units_t['p_discharge'][s] for s in network.storage_units.index} if hasattr(network, 'storage_units_t') else {},
            'storage_soc': {s: network.storage_units_t['state_of_charge'][s] for s in network.storage_units.index} if hasattr(network, 'storage_units_t') else {}
        }
        
        # Store the cost for this season
        season_costs[season] = network.objective_value
        
        print(f"Optimization completed for {season}. Objective value: {network.objective_value:.2f}")
    
    # Store the seasonal costs in the integrated network
    integrated_network.seasons_total_cost = season_costs
    
    # Calculate annual cost
    annual_cost = 0
    for season, cost in season_costs.items():
        weeks = integrated_network.season_weights.get(season, 0)
        annual_cost += weeks * cost
    
    integrated_network.annual_cost = annual_cost
    
    print(f"\nSeasonal optimization completed successfully.")
    print(f"Annual cost: {annual_cost:.2f}")
    
    return True 