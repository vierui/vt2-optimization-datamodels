"""
Optimization module for DC Optimal Power Flow
This module handles the mathematical optimization aspects of the model
"""
import cvxpy as cp
import pandas as pd

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
    # Initialize optimization variables
    p_gen = {}
    p_charge = {}
    p_discharge = {}
    soc = {}
    f = {}
    
    # Phase angle variables for buses (voltage angles in DCOPF)
    theta = {bus: cp.Variable(network.T) for bus in network.buses.index}
    
    # Create variables for generators
    for gen_name in network.generators.index:
        p_gen[gen_name] = cp.Variable(network.T, nonneg=True)
        
    # Create variables for storage units
    for storage_name in network.storage_units.index:
        p_charge[storage_name] = cp.Variable(network.T, nonneg=True)
        p_discharge[storage_name] = cp.Variable(network.T, nonneg=True)
        soc[storage_name] = cp.Variable(network.T, nonneg=True)
        
    # Initialize constraints list
    constraints = []
    
    # Choose a reference bus and fix its voltage angle to zero
    if not network.buses.empty:
        ref_bus = network.buses.index[0]
        constraints += [theta[ref_bus] == 0]
    
    # Generator capacity constraints
    for gen_name, gen_data in network.generators.iterrows():
        constraints += [p_gen[gen_name] <= gen_data['capacity']]
        
    # Storage constraints
    for storage_name, storage_data in network.storage_units.iterrows():
        # Power limits
        constraints += [p_charge[storage_name] <= storage_data['power']]
        constraints += [p_discharge[storage_name] <= storage_data['power']]
        
        # Energy capacity limits
        constraints += [soc[storage_name] <= storage_data['energy']]
        
        # Storage energy balance with initial SoC at 50%
        soc_init = 0.5 * storage_data['energy']
        for t in range(network.T):
            if t == 0:
                constraints += [
                    soc[storage_name][t] == soc_init 
                    + storage_data['charge_efficiency'] * p_charge[storage_name][t]
                    - (1 / storage_data['discharge_efficiency']) * p_discharge[storage_name][t]
                ]
            else:
                constraints += [
                    soc[storage_name][t] == soc[storage_name][t-1] 
                    + storage_data['charge_efficiency'] * p_charge[storage_name][t]
                    - (1 / storage_data['discharge_efficiency']) * p_discharge[storage_name][t]
                ]
    
    # DC power flow constraints for each line
    for line_name, line_data in network.lines.iterrows():
        from_bus = line_data['from']
        to_bus = line_data['to']
        susceptance = line_data['susceptance']
        capacity = line_data['capacity']
        
        # Create flow variables
        f[line_name] = cp.Variable(network.T)
        
        # Flow equation based on susceptance and voltage angle difference
        for t in range(network.T):
            constraints += [
                f[line_name][t] == susceptance * (theta[from_bus][t] - theta[to_bus][t])
            ]
        
        # Transmission line flow limits
        constraints += [f[line_name] <= capacity, f[line_name] >= -capacity]
                
    # Nodal power balance constraints
    for t in range(network.T):
        for bus_name in network.buses.index:
            # Generator contribution
            gen_at_bus = [p_gen[g][t] for g in network.generators.index 
                           if network.generators.loc[g, 'bus'] == bus_name]
            gen_sum = sum(gen_at_bus) if gen_at_bus else 0
            
            # Storage contribution
            storage_at_bus = [p_discharge[s][t] - p_charge[s][t] for s in network.storage_units.index 
                               if network.storage_units.loc[s, 'bus'] == bus_name]
            storage_net = sum(storage_at_bus) if storage_at_bus else 0
            
            # Load at the bus
            load = network.loads_t.loc[t, bus_name] if bus_name in network.loads_t.columns else 0
            
            # Line flows into and out of the bus
            flow_out = sum(f[l][t] for l, data in network.lines.iterrows() 
                           if data['from'] == bus_name)
            flow_in = sum(f[l][t] for l, data in network.lines.iterrows() 
                          if data['to'] == bus_name)
            
            # Power balance: generation + storage net + flow in = load + flow out
            constraints += [gen_sum + storage_net + flow_in == load + flow_out]
        
    # Objective function: minimize total generation cost
    objective = cp.Minimize(
        sum(network.generators.loc[g, 'cost'] * cp.sum(p_gen[g]) for g in network.generators.index)
    )
    
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
        prob.solve(solver=cp.CPLEX)
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