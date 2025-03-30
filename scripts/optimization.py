"""
Optimization module for DC Optimal Power Flow
This module handles the mathematical optimization aspects of the model
"""
import cvxpy as cp
import pandas as pd
import numpy as np

def create_dcopf_problem(network):
    """
    Create the DC Optimal Power Flow problem
    
    This function takes a network and creates the optimization problem
    with appropriate variables, constraints, and objective function.
    
    Args:
        network: Network object with component data
        
    Returns:
        Dictionary with variables, constraints, and objective
    """
    print("\nCreating DCOPF problem...")
    
    # Initialize constraints list
    constraints = []
    
    # Ensure we have at least one year for planning
    if not hasattr(network, 'years') or len(network.years) == 0:
        # Default to a single year if not specified
        network.years = [0]  # Use 0 as default year index
        network.year_weights = {0: 1.0}
        network.discount_rate = 0.05
        print("No planning years specified, defaulting to a single year")
    
    years = network.years
    print(f"Setting up planning problem with {len(years)} years...")
    
    # Initialize variables with year dimension
    p_gen = {(g, y): cp.Variable(network.T, nonneg=True) 
            for g in network.generators.index for y in years}
    p_charge = {(s, y): cp.Variable(network.T, nonneg=True)
               for s in network.storage_units.index for y in years}
    p_discharge = {(s, y): cp.Variable(network.T, nonneg=True)
                  for s in network.storage_units.index for y in years}
    soc = {(s, y): cp.Variable(network.T, nonneg=True)
          for s in network.storage_units.index for y in years}
    
    # Installation decision variables
    gen_installed = {(g, y): cp.Variable(boolean=True) 
                     for g in network.generators.index for y in years}
    storage_installed = {(s, y): cp.Variable(boolean=True)
                        for s in network.storage_units.index for y in years}
    
    # If we have more than one year, add asset lifetime and reinstallation logic
    if len(years) > 1:
        # Track if each generator is installed for the first time in each year
        gen_first_install = {(g, y): cp.Variable(boolean=True)
                            for g in network.generators.index for y in years}
        # Track if each storage unit is installed for the first time in each year
        storage_first_install = {(s, y): cp.Variable(boolean=True)
                                for s in network.storage_units.index for y in years}
        
        # Track generator installations specifically for replacements
        gen_replacement = {(g, y): cp.Variable(boolean=True)
                         for g in network.generators.index for y in years}
        storage_replacement = {(s, y): cp.Variable(boolean=True)
                             for s in network.storage_units.index for y in years}
        
        # FIXED: Setup lifetime tracking for each asset to ensure they remain installed for their full lifetime
        for g in network.generators.index:
            # Get generator lifetime
            lifetime = network.generators.loc[g].get('lifetime_years', 25)
            
            for y_idx, y in enumerate(years):
                # Find the decommissioning year index, if within planning horizon
                decomm_y_idx = y_idx + lifetime
                if decomm_y_idx < len(years):
                    decomm_y = years[decomm_y_idx]
                    
                    # For each installation year, create a constraint ensuring 
                    # that the asset is either replaced or decommissioned by its end of life
                    constraints += [
                        # If asset is installed in year y, by decomm_y it must either be replaced or decommissioned
                        gen_replacement[(g, decomm_y)] >= gen_first_install[(g, y)]
                    ]
                
                # ADDED: For each year after installation but before lifetime expires, enforce that the asset remains installed
                # This prevents unnecessary reinstallations
                for future_idx in range(y_idx + 1, min(y_idx + lifetime, len(years))):
                    future_y = years[future_idx]
                    constraints += [
                        # If installed in year y, it must remain installed in future years until replacement or lifetime ends
                        gen_installed[(g, future_y)] >= gen_first_install[(g, y)] - sum(gen_replacement[(g, years[i])] for i in range(y_idx + 1, future_idx + 1) if i < len(years))
                    ]
        
        # Same for storage
        for s in network.storage_units.index:
            # Get storage lifetime
            lifetime = network.storage_units.loc[s].get('lifetime_years', 15)
            
            for y_idx, y in enumerate(years):
                # Find the decommissioning year index, if within planning horizon
                decomm_y_idx = y_idx + lifetime
                if decomm_y_idx < len(years):
                    decomm_y = years[decomm_y_idx]
                    
                    # Create constraint for decommissioning or replacement
                    constraints += [
                        # If asset is installed in year y, by decomm_y it must either be replaced or decommissioned
                        storage_replacement[(s, decomm_y)] >= storage_first_install[(s, y)]
                    ]
                
                # ADDED: For each year after installation but before lifetime expires, enforce that the asset remains installed
                for future_idx in range(y_idx + 1, min(y_idx + lifetime, len(years))):
                    future_y = years[future_idx]
                    constraints += [
                        # If installed in year y, it must remain installed in future years until replacement or lifetime ends
                        storage_installed[(s, future_y)] >= storage_first_install[(s, y)] - sum(storage_replacement[(s, years[i])] for i in range(y_idx + 1, future_idx + 1) if i < len(years))
                    ]
        
        # MODIFIED: Asset continuity - if an asset is marked for replacement, it remains active
        for g in network.generators.index:
            for y_idx in range(1, len(years)):
                # CHANGED: Asset continuity now ensures asset remains installed until it needs replacement
                # But we don't force continuity for assets that were never installed
                constraints += [
                    # Asset installed status carries forward unless it needs replacement
                    gen_installed[(g, years[y_idx])] >= (gen_installed[(g, years[y_idx-1])] - gen_replacement[(g, years[y_idx])])
                ]
        
        for s in network.storage_units.index:
            for y_idx in range(1, len(years)):
                constraints += [
                    # Asset installed status carries forward unless it needs replacement
                    storage_installed[(s, years[y_idx])] >= (storage_installed[(s, years[y_idx-1])] - storage_replacement[(s, years[y_idx])])
                ]
        
        # MODIFIED: First installation happens when asset goes from not installed to installed
        for g in network.generators.index:
            # First year is a special case
            constraints += [gen_first_install[(g, years[0])] == gen_installed[(g, years[0])]]
            
            # For subsequent years, it's first installed if it wasn't installed before but is now,
            # OR if it was marked for replacement
            for y_idx in range(1, len(years)):
                constraints += [
                    gen_first_install[(g, years[y_idx])] == (
                        (gen_installed[(g, years[y_idx])] - gen_installed[(g, years[y_idx-1])]) + gen_replacement[(g, years[y_idx])]
                    )
                ]
        
        # Same for storage
        for s in network.storage_units.index:
            # First year is a special case
            constraints += [storage_first_install[(s, years[0])] == storage_installed[(s, years[0])]]
            
            # For subsequent years, first installed if it wasn't installed before but is now,
            # OR if it was marked for replacement
            for y_idx in range(1, len(years)):
                constraints += [
                    storage_first_install[(s, years[y_idx])] == (
                        (storage_installed[(s, years[y_idx])] - storage_installed[(s, years[y_idx-1])]) + storage_replacement[(s, years[y_idx])]
                    )
                ]
            
        # NEW: Add a budget constraint to limit installations per year
        # Force diversification of installations over time
        max_gen_installations_per_year = max(1, len(network.generators) // 2)  # At most half the generators in one year
        max_storage_installations_per_year = max(1, len(network.storage_units) // 2)  # At most half the storage in one year
            
        for y in years:
            # Limit generator installations per year
            constraints += [
                sum(gen_first_install[(g, y)] for g in network.generators.index) <= max_gen_installations_per_year
            ]
            
            # Limit storage installations per year
            constraints += [
                sum(storage_first_install[(s, y)] for s in network.storage_units.index) <= max_storage_installations_per_year
            ]
    else:
        # If only one year, installed is same as first install
        gen_first_install = gen_installed
        storage_first_install = storage_installed
    
    # Phase angle variables for buses
    # These are specific to each year and time period
    theta = {(b, y): cp.Variable(network.T) 
            for b in network.buses.index for y in years}
    
    # Line flow variables
    f = {(l, y): cp.Variable(network.T)
        for l in network.lines.index for y in years}
    
    # Set reference buses for each year
    if not network.buses.empty:
        ref_bus = network.buses.index[0]
        for y in years:
            constraints += [theta[(ref_bus, y)] == 0]
    
    print(f"Setting up generator constraints for {len(network.generators)} generators...")
    # Generator capacity constraints
    for y in years:
        for gen_id, gen_data in network.generators.iterrows():
            capacity = gen_data['capacity_mw']
            
            if hasattr(network, 'gen_p_max_pu') and gen_id in network.gen_p_max_pu:
                # Time-dependent constraint
                for t in range(network.T):
                    max_capacity = capacity * network.gen_p_max_pu[gen_id][t]
                    constraints += [p_gen[(gen_id, y)][t] <= max_capacity * gen_installed[(gen_id, y)]]
            else:
                # Static capacity constraint
                constraints += [p_gen[(gen_id, y)] <= capacity * gen_installed[(gen_id, y)]]
    
    print(f"Setting up storage constraints for {len(network.storage_units)} storage units...")
    # Storage constraints
    for y in years:
        for storage_id, storage_data in network.storage_units.iterrows():
            # Power limits
            constraints += [p_charge[(storage_id, y)] <= storage_data['p_mw'] * storage_installed[(storage_id, y)]]
            constraints += [p_discharge[(storage_id, y)] <= storage_data['p_mw'] * storage_installed[(storage_id, y)]]
            
            # Energy capacity limits
            constraints += [soc[(storage_id, y)] <= storage_data['energy_mwh'] * storage_installed[(storage_id, y)]]
            
            # Storage energy balance with initial SoC at 50%
            soc_init = 0.5 * storage_data['energy_mwh']
            for t in range(network.T):
                if t == 0:
                    constraints += [
                        soc[(storage_id, y)][t] == soc_init * storage_installed[(storage_id, y)] 
                        + storage_data['efficiency_store'] * p_charge[(storage_id, y)][t]
                        - (1 / storage_data['efficiency_dispatch']) * p_discharge[(storage_id, y)][t]
                    ]
                else:
                    constraints += [
                        soc[(storage_id, y)][t] == soc[(storage_id, y)][t-1] 
                        + storage_data['efficiency_store'] * p_charge[(storage_id, y)][t]
                        - (1 / storage_data['efficiency_dispatch']) * p_discharge[(storage_id, y)][t]
                    ]
            
            # Final SoC equals initial SoC
            constraints += [soc[(storage_id, y)][network.T-1] == soc_init * storage_installed[(storage_id, y)]]
    
    print(f"Setting up line flow constraints for {len(network.lines)} lines...")
    # DC power flow constraints for each line
    for y in years:
        for line_id, line_data in network.lines.iterrows():
            from_bus = line_data['bus_from']
            to_bus = line_data['bus_to']
            susceptance = line_data['susceptance']
            capacity = line_data['capacity_mw']
            
            # Flow equation based on susceptance and voltage angle difference
            for t in range(network.T):
                constraints += [
                    f[(line_id, y)][t] == susceptance * (theta[(from_bus, y)][t] - theta[(to_bus, y)][t])
                ]
            
            # Transmission line flow limits
            constraints += [f[(line_id, y)] <= capacity, f[(line_id, y)] >= -capacity]
    
    # Pre-calculate load values per bus
    print("Pre-calculating loads per bus...")
    bus_load = {}
    for bus_id in network.buses.index:
        bus_load[bus_id] = np.zeros(network.T)
    
    # Map loads to buses
    # First create a mapping from load index to bus ID
    load_to_bus_map = {}
    for load_id in network.loads.index:
        load_to_bus_map[load_id] = network.loads.at[load_id, 'bus_id']
    
    # Print the load to bus mapping
    print(f"Load to bus mapping: {load_to_bus_map}")
    
    # Now map loads to buses using the mapping
    for t in range(network.T):
        for load_id in network.loads_t.columns:
            if t < len(network.loads_t):
                bus_id = load_to_bus_map.get(load_id)
                if bus_id in bus_load:
                    bus_load[bus_id][t] += network.loads_t.iloc[t][load_id]
    
    # Debug total system load
    total_load = np.zeros(network.T)
    for bus_id in network.buses.index:
        total_load += bus_load[bus_id]
    
    print(f"Total system load: min={total_load.min():.2f}, max={total_load.max():.2f}, avg={total_load.mean():.2f}")
    
    print(f"Setting up power balance constraints for {len(network.buses)} buses...")
    # Nodal power balance constraints
    for y in years:
        # Get load growth factor for this year
        load_growth_factor = 1.0  # Default no growth
        if hasattr(network, 'year_to_load_factor') and y in network.year_to_load_factor:
            load_growth_factor = network.year_to_load_factor[y]
            print(f"Year {y}: Applying load growth factor {load_growth_factor}")
        
        for t in range(network.T):
            for bus_id in network.buses.index:
                # Generator contribution
                gen_at_bus = [p_gen[(g, y)][t] for g in network.generators.index 
                            if network.generators.loc[g, 'bus_id'] == bus_id]
                gen_sum = sum(gen_at_bus) if gen_at_bus else 0
                
                # Storage contribution
                storage_at_bus = [p_discharge[(s, y)][t] - p_charge[(s, y)][t] for s in network.storage_units.index 
                                if network.storage_units.loc[s, 'bus_id'] == bus_id]
                storage_net = sum(storage_at_bus) if storage_at_bus else 0
                
                # Load at the bus - apply load growth factor
                load = bus_load[bus_id][t] * load_growth_factor
                
                # Line flows into and out of the bus
                flow_out = sum(f[(l, y)][t] for l, data in network.lines.iterrows() 
                            if data['bus_from'] == bus_id)
                flow_in = sum(f[(l, y)][t] for l, data in network.lines.iterrows() 
                            if data['bus_to'] == bus_id)
                
                # Power balance: generation + storage net + flow in = load + flow out
                constraints += [gen_sum + storage_net + flow_in == load + flow_out]
    
    # Print debug info for first and last hour
    for t in [0, network.T-1]:
        print(f"Hour {t} loads: {[(bus_id, bus_load[bus_id][t]) for bus_id in sorted(network.buses.index)]}")
    
    # Objective function: minimize total cost (operation + capital)
    
    # Apply discount rate to costs
    default_discount_rate = getattr(network, 'discount_rate', 0.05)  # Default 5% if not specified
    
    # 1. Operational costs: generation cost per MWh * generation
    operational_costs = []
    for y_idx, y in enumerate(years):
        for gen_id in network.generators.index:
            # Use generator-specific discount rate if available, otherwise use default
            gen_discount_rate = network.generators.loc[gen_id].get('discount_rate', default_discount_rate)
            discount_factor = 1 / ((1 + gen_discount_rate) ** y_idx)
            
            cost_per_mwh = network.generators.loc[gen_id, 'cost_mwh']
            for t in range(network.T):
                operational_costs.append(discount_factor * cost_per_mwh * p_gen[(gen_id, y)][t])
    
    # 2. Capital costs: CAPEX / lifetime * binary installation variable
    capital_costs = []
    
    # Generator CAPEX
    for y_idx, y in enumerate(years):
        for gen_id, gen_data in network.generators.iterrows():
            # Use generator-specific discount rate if available, otherwise use default
            gen_discount_rate = gen_data.get('discount_rate', default_discount_rate)
            discount_factor = 1 / ((1 + gen_discount_rate) ** y_idx)
            
            # Only count CAPEX when generator is first installed in a year
            capex = gen_data.get('capex_per_mw', 0) * gen_data['capacity_mw']
            
            # FIXED: Only apply annual CAPEX amortization if the asset is newly installed or replaced
            # Get correct lifetime
            lifetime = gen_data.get('lifetime_years', 25)
            annual_capex = capex / lifetime  # Annualized capex
            
            # Apply different learning curves based on technology type
            tech_type = gen_data.get('type', 'unknown')
            learning_factor = 1.0  # Default no learning
            
            # Apply different learning curves based on technology type
            if tech_type == 'wind':
                learning_factor = max(0.7, 1.0 - (0.03 * y_idx))  # Wind costs decrease by 3% per year
            elif tech_type == 'solar':
                learning_factor = max(0.6, 1.0 - (0.04 * y_idx))  # Solar costs decrease by 4% per year
            elif tech_type == 'storage':
                learning_factor = max(0.5, 1.0 - (0.05 * y_idx))  # Storage costs decrease by 5% per year
            
            # Cost increases for thermal generators (carbon taxes or fuel costs)
            elif tech_type == 'thermal':
                learning_factor = min(1.5, 1.0 + (0.02 * y_idx))  # Thermal costs increase by 2% per year
            
            # FIXED: Apply CAPEX only at first installation or replacement, not every year the asset is installed
            if 'gen_first_install' in locals():
                yearly_capex = discount_factor * annual_capex * learning_factor * gen_first_install[(gen_id, y)]
                capital_costs.append(yearly_capex)
            else:
                yearly_capex = discount_factor * annual_capex * learning_factor * gen_installed[(gen_id, y)]
                capital_costs.append(yearly_capex)
    
    # Storage CAPEX
    for y_idx, y in enumerate(years):
        for storage_id, storage_data in network.storage_units.iterrows():
            # Use storage-specific discount rate if available, otherwise use default
            storage_discount_rate = storage_data.get('discount_rate', default_discount_rate)
            discount_factor = 1 / ((1 + storage_discount_rate) ** y_idx)
            
            # Only count CAPEX when storage is first installed in a year
            capex = storage_data.get('capex_per_mw', 0) * storage_data['p_mw']
            
            # FIXED: Use correct lifetime and only apply CAPEX when newly installed or replaced
            lifetime = storage_data.get('lifetime_years', 15)
            annual_capex = capex / lifetime
            
            # Apply learning factor for storage technology
            learning_factor = max(0.5, 1.0 - (0.05 * y_idx))  # Storage costs decrease by 5% per year
            
            # FIXED: Apply CAPEX only at first installation or replacement, not every year the asset is installed
            if 'storage_first_install' in locals():
                yearly_capex = discount_factor * annual_capex * learning_factor * storage_first_install[(storage_id, y)]
                capital_costs.append(yearly_capex)
            else:
                yearly_capex = discount_factor * annual_capex * learning_factor * storage_installed[(storage_id, y)]
                capital_costs.append(yearly_capex)
    
    # Total objective: sum of operational and capital costs
    objective = cp.Minimize(sum(operational_costs) + sum(capital_costs))
    
    # Return all problem components
    return {
        'variables': {
            'p_gen': p_gen,
            'p_charge': p_charge,
            'p_discharge': p_discharge,
            'soc': soc,
            'f': f,
            'theta': theta,
            'gen_installed': gen_installed,
            'storage_installed': storage_installed,
            'gen_first_install': gen_first_install,
            'storage_first_install': storage_first_install,
            'gen_replacement': gen_replacement if len(years) > 1 else None,
            'storage_replacement': storage_replacement if len(years) > 1 else None
        },
        'constraints': constraints,
        'objective': objective,
        'multi_year': True,
        'years': years
    }

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
    
    try:
        # Always use CPLEX - no fallbacks
        print("Solving with CPLEX...")
        prob.solve(solver=cp.CPLEX, verbose=True)
        print(f"Problem status: {prob.status}")
        print(f"Objective value: {prob.value}")
    except Exception as e:
        print(f"CPLEX optimization failed: {e}")
        return {'status': 'failed', 'success': False, 'value': None}
        
    # Check if solution is optimal
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Problem status: {prob.status}")
        return {'status': prob.status, 'success': False, 'value': None}
    
    # Return the solution status and value
    return {
        'status': prob.status,
        'success': True,
        'value': prob.value
    }

def extract_results(network, problem, solution):
    """
    Extract and format the results from the solved problem
    
    Args:
        network: Network object to store results in
        problem: Dictionary with problem variables
        solution: Dictionary with solution status and value
        
    Returns:
        Nothing, updates the network object directly
    """
    if not solution['success']:
        return
    
    # Extract variables
    p_gen = problem['variables']['p_gen']
    p_charge = problem['variables']['p_charge']
    p_discharge = problem['variables']['p_discharge']
    soc = problem['variables']['soc']
    f = problem['variables']['f']
    theta = problem['variables']['theta']
    gen_installed = problem['variables']['gen_installed']
    storage_installed = problem['variables']['storage_installed']
    
    # Get optional variables
    gen_first_install = problem['variables'].get('gen_first_install', gen_installed)
    storage_first_install = problem['variables'].get('storage_first_install', storage_installed)
    
    # Get replacement variables if they exist in the problem
    gen_replacement = problem['variables'].get('gen_replacement', None)
    storage_replacement = problem['variables'].get('storage_replacement', None)
    
    # Get years from the problem
    years = problem['years']
    print(f"\nExtracting results for {len(years)} years...")
    
    # Initialize dictionaries to store results by year
    network.generators_t_by_year = {year: {} for year in years}
    network.storage_units_t_by_year = {year: {} for year in years}
    network.lines_t_by_year = {year: {} for year in years}
    network.buses_t_by_year = {year: {} for year in years}
    
    network.generators_installed_by_year = {year: {} for year in years}
    network.storage_installed_by_year = {year: {} for year in years}
    
    # Track first installation years 
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
    
    # Extract results for each year
    for year in years:
        # Initialize DataFrames for this year
        network.generators_t_by_year[year]['p'] = pd.DataFrame(index=range(network.T))
        network.storage_units_t_by_year[year]['p_charge'] = pd.DataFrame(index=range(network.T))
        network.storage_units_t_by_year[year]['p_discharge'] = pd.DataFrame(index=range(network.T))
        network.storage_units_t_by_year[year]['state_of_charge'] = pd.DataFrame(index=range(network.T))
        network.lines_t_by_year[year]['p'] = pd.DataFrame(index=range(network.T))
        network.buses_t_by_year[year]['v_ang'] = pd.DataFrame(index=range(network.T))
        
        # Populate results for this year
        # Generators
        for g in network.generators.index:
            network.generators_t_by_year[year]['p'][g] = p_gen[(g, year)].value
            network.generators_installed_by_year[year][g] = gen_installed[(g, year)].value
            network.generators_first_install_by_year[year][g] = gen_first_install[(g, year)].value
            
            # Store replacement information if available
            if gen_replacement is not None:
                network.generators_replacement_by_year[year][g] = gen_replacement[(g, year)].value
            
            # Determine if this is a replacement
            is_replacement = False
            if hasattr(network, 'generators_replacement_by_year') and year in network.generators_replacement_by_year:
                if g in network.generators_replacement_by_year[year]:
                    replacement_val = network.generators_replacement_by_year[year][g]
                    is_replacement = replacement_val is not None and replacement_val > 0.5
            
            # Add to installation history if first installed in this year
            if gen_first_install[(g, year)].value > 0.5:
                if g not in network.asset_installation_history['generators']:
                    network.asset_installation_history['generators'][g] = []
                
                network.asset_installation_history['generators'][g].append({
                    'installation_year': year,
                    'capacity_mw': network.generators.loc[g, 'capacity_mw'],
                    'capex_per_mw': network.generators.loc[g].get('capex_per_mw', 0),
                    'lifetime_years': network.generators.loc[g].get('lifetime_years', 25),
                    'is_replacement': is_replacement
                })
        
        # Storage units
        for s in network.storage_units.index:
            network.storage_units_t_by_year[year]['p_charge'][s] = p_charge[(s, year)].value
            network.storage_units_t_by_year[year]['p_discharge'][s] = p_discharge[(s, year)].value
            network.storage_units_t_by_year[year]['state_of_charge'][s] = soc[(s, year)].value
            network.storage_installed_by_year[year][s] = storage_installed[(s, year)].value
            network.storage_first_install_by_year[year][s] = storage_first_install[(s, year)].value
            
            # Store replacement information if available
            if storage_replacement is not None:
                network.storage_replacement_by_year[year][s] = storage_replacement[(s, year)].value
            
            # Determine if this is a replacement
            is_replacement = False
            if hasattr(network, 'storage_replacement_by_year') and year in network.storage_replacement_by_year:
                if s in network.storage_replacement_by_year[year]:
                    replacement_val = network.storage_replacement_by_year[year][s]
                    is_replacement = replacement_val is not None and replacement_val > 0.5
            
            # Add to installation history if first installed in this year
            if storage_first_install[(s, year)].value > 0.5:
                if s not in network.asset_installation_history['storage']:
                    network.asset_installation_history['storage'][s] = []
                
                network.asset_installation_history['storage'][s].append({
                    'installation_year': year,
                    'capacity_mw': network.storage_units.loc[s, 'p_mw'],
                    'energy_capacity_mwh': network.storage_units.loc[s, 'energy_mwh'],
                    'capex_per_mw': network.storage_units.loc[s].get('capex_per_mw', 0),
                    'lifetime_years': network.storage_units.loc[s].get('lifetime_years', 15),
                    'is_replacement': is_replacement
                })
        
        # Lines
        for l in network.lines.index:
            network.lines_t_by_year[year]['p'][l] = f[(l, year)].value
        
        # Bus voltage angles
        for b in network.buses.index:
            network.buses_t_by_year[year]['v_ang'][b] = theta[(b, year)].value
    
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
    
    # Print summary of generation and installation decisions
    print("\nSUMMARY OF OPTIMIZATION RESULTS")
    
    # Results summary for each year
    total_discounted_cost = 0
    
    # ADDED: Track asset lifetimes and replacement history
    lifetime_expirations = {
        'generators': {},
        'storage': {}
    }
    
    # Initialize lifetime tracking
    for g in network.generators.index:
        lifetime = network.generators.loc[g].get('lifetime_years', 25)
        lifetime_expirations['generators'][g] = lifetime
    
    for s in network.storage_units.index:
        lifetime = network.storage_units.loc[s].get('lifetime_years', 15)
        lifetime_expirations['storage'][s] = lifetime
    
    for year_idx, year in enumerate(years):
        # Apply discount factor to calculations
        default_discount_rate = getattr(network, 'discount_rate', 0.05)  # Default 5% if not specified
        
        total_gen = 0
        operational_cost = 0
        capex_cost = 0
        
        print(f"\n----- YEAR {year} -----")
        
        # Calculate generator costs and dispatch
        print("\nGenerator installation and dispatch decisions:")
        for g in network.generators.index:
            gen_sum = network.generators_t_by_year[year]['p'][g].sum()
            total_gen += gen_sum
            gen_cost = network.generators.loc[g, 'cost_mwh']
            op_cost = gen_cost * gen_sum
            operational_cost += op_cost
            
            # Get installation status
            installed = network.generators_installed_by_year[year][g]
            is_first_install = network.generators_first_install_by_year[year][g] > 0.5
            
            # Determine if this is a replacement
            is_replacement = False
            if hasattr(network, 'generators_replacement_by_year') and year in network.generators_replacement_by_year:
                if g in network.generators_replacement_by_year[year]:
                    replacement_val = network.generators_replacement_by_year[year][g]
                    is_replacement = replacement_val is not None and replacement_val > 0.5
            
            # Calculate CAPEX if generator is first installed or replaced in this year
            capex = 0
            if is_first_install:
                capacity = network.generators.loc[g, 'capacity_mw']
                capex_per_mw = network.generators.loc[g].get('capex_per_mw', 0)
                lifetime = network.generators.loc[g].get('lifetime_years', 25)
                capex = (capex_per_mw * capacity) / lifetime
                capex_cost += capex
            
            # Use asset-specific discount rate if available
            gen_discount_rate = network.generators.loc[g].get('discount_rate', default_discount_rate)
            discount_factor = 1 / ((1 + gen_discount_rate) ** year_idx)
            discounted_op_cost = op_cost * discount_factor
            discounted_capex = capex * discount_factor
            
            # IMPROVED: Display clearer installation status information
            installation_status = "Not installed"
            if installed > 0.5:
                if is_first_install:
                    installation_status = "Newly installed" if not is_replacement else "Replaced"
                else:
                    installation_status = "Active"
            
            print(f"Generator {g}: Status = {installation_status}, Total dispatch = {gen_sum:.2f} MWh, "
                  f"Operational cost = {op_cost:.2f}, Annual CAPEX = {capex:.2f}, "
                  f"Discount rate = {gen_discount_rate:.1%}, Discounted cost = {(discounted_op_cost + discounted_capex):.2f}")
        
        # Calculate storage costs
        print("\nStorage installation decisions:")
        for s in network.storage_units.index:
            # Get installation status
            installed = network.storage_installed_by_year[year][s]
            is_first_install = network.storage_first_install_by_year[year][s] > 0.5
            
            # Determine if this is a replacement
            is_replacement = False
            if hasattr(network, 'storage_replacement_by_year') and year in network.storage_replacement_by_year:
                if s in network.storage_replacement_by_year[year]:
                    replacement_val = network.storage_replacement_by_year[year][s]
                    is_replacement = replacement_val is not None and replacement_val > 0.5
            
            # Calculate CAPEX if storage is first installed or replaced in this year
            capex = 0
            if is_first_install:
                capacity = network.storage_units.loc[s, 'p_mw']
                capex_per_mw = network.storage_units.loc[s].get('capex_per_mw', 0)
                lifetime = network.storage_units.loc[s].get('lifetime_years', 15)
                capex = (capex_per_mw * capacity) / lifetime
                capex_cost += capex
            
            # Use asset-specific discount rate if available
            storage_discount_rate = network.storage_units.loc[s].get('discount_rate', default_discount_rate)
            discount_factor = 1 / ((1 + storage_discount_rate) ** year_idx)
            discounted_capex = capex * discount_factor
            
            charge_sum = network.storage_units_t_by_year[year]['p_charge'][s].sum()
            discharge_sum = network.storage_units_t_by_year[year]['p_discharge'][s].sum()
            
            # IMPROVED: Display clearer installation status information
            installation_status = "Not installed"
            if installed > 0.5:
                if is_first_install:
                    installation_status = "Newly installed" if not is_replacement else "Replaced"
                else:
                    installation_status = "Active"
            
            print(f"Storage {s}: Status = {installation_status}, "
                  f"Total charging = {charge_sum:.2f} MWh, Total discharging = {discharge_sum:.2f} MWh, "
                  f"Annual CAPEX = {capex:.2f}, Discount rate = {storage_discount_rate:.1%}, "
                  f"Discounted CAPEX = {discounted_capex:.2f}")
        
        # ADDED: Track and report assets approaching end of lifetime
        if year_idx > 0:
            print("\nAssets approaching end of lifetime:")
            approaching_eol = False
            for g in network.generators.index:
                if network.generators_installed_by_year[year][g] > 0.5:
                    # Look ahead to see if this asset will need replacement in the next 2 years
                    lifetime = network.generators.loc[g].get('lifetime_years', 25)
                    # Find the most recent installation year
                    installation_years = [y for y in range(year_idx+1) if 
                                          y < len(years) and 
                                          network.generators_first_install_by_year[years[y]][g] > 0.5]
                    if installation_years:
                        most_recent = years[max(installation_years)]
                        years_active = year_idx - max(installation_years) + 1
                        years_remaining = lifetime - years_active
                        if 0 < years_remaining <= 2:
                            approaching_eol = True
                            print(f"Generator {g}: Installed in Year {most_recent}, "
                                  f"Active for {years_active} years, "
                                  f"Lifetime: {lifetime} years, "
                                  f"Years remaining: {years_remaining}")
            
            for s in network.storage_units.index:
                if network.storage_installed_by_year[year][s] > 0.5:
                    # Look ahead to see if this asset will need replacement in the next 2 years
                    lifetime = network.storage_units.loc[s].get('lifetime_years', 15)
                    # Find the most recent installation year
                    installation_years = [y for y in range(year_idx+1) if 
                                          y < len(years) and 
                                          network.storage_first_install_by_year[years[y]][s] > 0.5]
                    if installation_years:
                        most_recent = years[max(installation_years)]
                        years_active = year_idx - max(installation_years) + 1
                        years_remaining = lifetime - years_active
                        if 0 < years_remaining <= 2:
                            approaching_eol = True
                            print(f"Storage {s}: Installed in Year {most_recent}, "
                                  f"Active for {years_active} years, "
                                  f"Lifetime: {lifetime} years, "
                                  f"Years remaining: {years_remaining}")
            
            if not approaching_eol:
                print("None")
        
        # Calculate load for this year
        total_load = 0
        load_growth_factor = 1.0
        if hasattr(network, 'year_to_load_factor') and year in network.year_to_load_factor:
            load_growth_factor = network.year_to_load_factor[year]
            
        for load_id in network.loads.index:
            if load_id in network.loads_t.columns:
                load_sum = network.loads_t[load_id].sum() * load_growth_factor
                total_load += load_sum
        
        # Calculate weighted year costs using default discount rate for totals
        discount_factor = 1 / ((1 + default_discount_rate) ** year_idx)
        year_total_cost = operational_cost + capex_cost
        discounted_cost = year_total_cost * discount_factor
        total_discounted_cost += discounted_cost
        
        print(f"\nYear {year} Summary:")
        print(f"Total generation: {total_gen:.2f} MWh")
        print(f"Total load: {total_load:.2f} MWh")
        print(f"Total operational cost: {operational_cost:.2f}")
        print(f"Total annual CAPEX: {capex_cost:.2f}")
        print(f"Total cost for year {year}: {year_total_cost:.2f}")
        print(f"System discount rate: {default_discount_rate:.1%}")
        print(f"Discounted cost (using system rate): {discounted_cost:.2f}")
        
        if abs(total_gen - total_load) > 0.01:
            print(f"WARNING: Generation-load mismatch! Difference: {total_gen - total_load:.2f} MWh")
    
    # Print overall summary for the entire planning horizon
    print("\n----- OVERALL PLANNING HORIZON SUMMARY -----")
    print(f"Total discounted cost across planning horizon: {total_discounted_cost:.2f}")
    
    # Store the total discounted cost
    network.total_discounted_cost = total_discounted_cost 