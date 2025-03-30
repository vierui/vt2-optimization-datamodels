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
    
    # Initialize optimization variables
    p_gen = {}
    p_charge = {}
    p_discharge = {}
    soc = {}
    f = {}
    
    # Phase angle variables for buses (voltage angles in DCOPF)
    theta = {bus_id: cp.Variable(network.T) for bus_id in network.buses.index}
    
    # Create variables for generators
    for gen_id in network.generators.index:
        p_gen[gen_id] = cp.Variable(network.T, nonneg=True)
        
    # Create variables for storage units
    for storage_id in network.storage_units.index:
        p_charge[storage_id] = cp.Variable(network.T, nonneg=True)
        p_discharge[storage_id] = cp.Variable(network.T, nonneg=True)
        soc[storage_id] = cp.Variable(network.T, nonneg=True)
        
    # Initialize constraints list
    constraints = []
    
    # Choose a reference bus and fix its voltage angle to zero
    if not network.buses.empty:
        ref_bus = network.buses.index[0]
        constraints += [theta[ref_bus] == 0]
    
    print(f"Setting up generator constraints for {len(network.generators)} generators...")
    # Generator capacity constraints
    for gen_id, gen_data in network.generators.iterrows():
        # Get generator capacity
        capacity = gen_data['capacity_mw']
        
        # Check if we have time-dependent availability (p_max_pu)
        if hasattr(network, 'gen_p_max_pu') and gen_id in network.gen_p_max_pu:
            # Time-dependent constraint
            for t in range(network.T):
                max_capacity = capacity * network.gen_p_max_pu[gen_id][t]
                constraints += [p_gen[gen_id][t] <= max_capacity]
        else:
            # Static capacity constraint
            constraints += [p_gen[gen_id] <= capacity]
    
    print(f"Setting up storage constraints for {len(network.storage_units)} storage units...")
    # Storage constraints
    for storage_id, storage_data in network.storage_units.iterrows():
        # Power limits
        constraints += [p_charge[storage_id] <= storage_data['p_mw']]
        constraints += [p_discharge[storage_id] <= storage_data['p_mw']]
        
        # Energy capacity limits
        constraints += [soc[storage_id] <= storage_data['energy_mwh']]
        
        # Storage energy balance with initial SoC at 50%
        soc_init = 0.5 * storage_data['energy_mwh']
        for t in range(network.T):
            if t == 0:
                constraints += [
                    soc[storage_id][t] == soc_init 
                    + storage_data['efficiency_store'] * p_charge[storage_id][t]
                    - (1 / storage_data['efficiency_dispatch']) * p_discharge[storage_id][t]
                ]
            else:
                constraints += [
                    soc[storage_id][t] == soc[storage_id][t-1] 
                    + storage_data['efficiency_store'] * p_charge[storage_id][t]
                    - (1 / storage_data['efficiency_dispatch']) * p_discharge[storage_id][t]
                ]
        
        # Add constraint to ensure final SoC equals initial SoC
        constraints += [soc[storage_id][network.T-1] == soc_init]
    
    print(f"Setting up line flow constraints for {len(network.lines)} lines...")
    # DC power flow constraints for each line
    for line_id, line_data in network.lines.iterrows():
        from_bus = line_data['bus_from']
        to_bus = line_data['bus_to']
        susceptance = line_data['susceptance']
        capacity = line_data['capacity_mw']
        
        # Create flow variables
        f[line_id] = cp.Variable(network.T)
        
        # Flow equation based on susceptance and voltage angle difference
        for t in range(network.T):
            constraints += [
                f[line_id][t] == susceptance * (theta[from_bus][t] - theta[to_bus][t])
            ]
        
        # Transmission line flow limits
        constraints += [f[line_id] <= capacity, f[line_id] >= -capacity]
    
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
    for t in range(network.T):
        for bus_id in network.buses.index:
            # Generator contribution
            gen_at_bus = [p_gen[g][t] for g in network.generators.index 
                           if network.generators.loc[g, 'bus_id'] == bus_id]
            gen_sum = sum(gen_at_bus) if gen_at_bus else 0
            
            # Storage contribution
            storage_at_bus = [p_discharge[s][t] - p_charge[s][t] for s in network.storage_units.index 
                               if network.storage_units.loc[s, 'bus_id'] == bus_id]
            storage_net = sum(storage_at_bus) if storage_at_bus else 0
            
            # Load at the bus - use pre-calculated value
            load = bus_load[bus_id][t]
            
            # Line flows into and out of the bus
            flow_out = sum(f[l][t] for l, data in network.lines.iterrows() 
                           if data['bus_from'] == bus_id)
            flow_in = sum(f[l][t] for l, data in network.lines.iterrows() 
                          if data['bus_to'] == bus_id)
            
            # Power balance: generation + storage net + flow in = load + flow out
            constraints += [gen_sum + storage_net + flow_in == load + flow_out]
        
        # Print debug info for first and last hour
        if t == 0 or t == network.T-1:
            print(f"Hour {t} loads: {[(bus_id, bus_load[bus_id][t]) for bus_id in sorted(network.buses.index)]}")
    
    # Objective function: minimize total generation cost
    # Make sure to correctly apply the cost to each time period's generation
    objective_terms = []
    for gen_id in network.generators.index:
        # Get cost per MWh
        cost_per_mwh = network.generators.loc[gen_id, 'cost_mwh']
        
        # Add cost for each hour's generation
        for t in range(network.T):
            objective_terms.append(cost_per_mwh * p_gen[gen_id][t])
    
    objective = cp.Minimize(sum(objective_terms))
    
    # Return all problem components
    return {
        'variables': {
            'p_gen': p_gen,
            'p_charge': p_charge,
            'p_discharge': p_discharge,
            'soc': soc,
            'f': f,
            'theta': theta
        },
        'constraints': constraints,
        'objective': objective
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
    
    # Initialize result dataframes with proper indexes
    network.generators_t['p'] = pd.DataFrame(index=range(network.T))
    network.storage_units_t['p_charge'] = pd.DataFrame(index=range(network.T))
    network.storage_units_t['p_discharge'] = pd.DataFrame(index=range(network.T))
    network.storage_units_t['state_of_charge'] = pd.DataFrame(index=range(network.T))
    network.lines_t['p'] = pd.DataFrame(index=range(network.T))
    network.buses_t['v_ang'] = pd.DataFrame(index=range(network.T))
    
    # Populate results
    # Generators
    for g in network.generators.index:
        network.generators_t['p'][g] = p_gen[g].value
    
    # Storage units
    for s in network.storage_units.index:
        network.storage_units_t['p_charge'][s] = p_charge[s].value
        network.storage_units_t['p_discharge'][s] = p_discharge[s].value
        network.storage_units_t['state_of_charge'][s] = soc[s].value
    
    # Lines
    for l in network.lines.index:
        network.lines_t['p'][l] = f[l].value
    
    # Bus voltage angles
    for b in network.buses.index:
        network.buses_t['v_ang'][b] = theta[b].value
    
    # Store objective value
    network.objective_value = solution['value']
    
    # Print summary of generation
    total_gen = 0
    for g in network.generators.index:
        gen_sum = network.generators_t['p'][g].sum()
        total_gen += gen_sum
        gen_cost = network.generators.loc[g, 'cost_mwh']
        print(f"Generator {g}: Total dispatch = {gen_sum:.2f} MWh, Cost = {gen_cost * gen_sum:.2f}")
    
    print(f"Total generation: {total_gen:.2f} MWh")
    
    # Calculate total load
    total_load = 0
    for load_id in network.loads.index:
        if load_id in network.loads_t.columns:
            load_sum = network.loads_t[load_id].sum() 
            total_load += load_sum
            print(f"Load {load_id}: {load_sum:.2f} MWh")
    
    print(f"Total load: {total_load:.2f} MWh")
    
    if abs(total_gen - total_load) > 0.01:
        print(f"WARNING: Generation-load mismatch! Difference: {total_gen - total_load:.2f} MWh") 