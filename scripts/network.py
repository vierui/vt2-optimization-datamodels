import pandas as pd
import numpy as np
import cvxpy as cp
from components import Bus, Generator, Load, Storage, Line
from optimization import create_dcopf_problem, solve_with_cplex, extract_results

class Network:
    """
    Network class that holds all power system components and handles optimization
    Similar in concept to PyPSA's Network class
    """
    def __init__(self):
        """Initialize an empty network"""
        # Time settings
        self.snapshots = None
        self.T = 0
        
        # Static component DataFrames - columns defined explicitly
        self.buses = pd.DataFrame(columns=['name'])
        self.generators = pd.DataFrame(columns=['bus', 'capacity', 'cost'])
        self.loads = pd.DataFrame(columns=['bus'])
        self.storage_units = pd.DataFrame(columns=['bus', 'power', 'energy', 'charge_efficiency', 'discharge_efficiency'])
        self.lines = pd.DataFrame(columns=['from', 'to', 'susceptance', 'capacity'])
        
        # Time-series component DataFrames
        self.loads_t = pd.DataFrame()
        
        # Results storage
        self.generators_t = {}
        self.storage_units_t = {}
        self.lines_t = {}
        self.buses_t = {}
                
    def set_snapshots(self, T):
        """Set the time snapshots for the model"""
        self.T = T
        self.snapshots = pd.RangeIndex(T)
        
    def add_bus(self, name):
        """Add a bus to the network"""
        if name not in self.buses.index:
            self.buses.loc[name] = {'name': name}
        return self
        
    def add_generator(self, name, bus, capacity, cost):
        """Add a generator to the network"""
        if name not in self.generators.index:
            self.generators.loc[name] = {
                'bus': bus,
                'capacity': capacity,
                'cost': cost
            }
        return self
        
    def add_load(self, bus, p_set):
        """Add a load to the network"""
        if bus not in self.loads.index:
            self.loads.loc[bus] = {'bus': bus}
            
        # Set or update the load time series
        if self.T > 0:
            if isinstance(p_set, (int, float)):
                # Make sure loads_t has an index
                if self.loads_t.empty:
                    self.loads_t = pd.DataFrame(index=range(self.T))
                self.loads_t[bus] = [p_set] * self.T
            else:
                # Make sure loads_t has an index
                if self.loads_t.empty:
                    self.loads_t = pd.DataFrame(index=range(self.T))
                self.loads_t[bus] = p_set
        return self
        
    def add_storage(self, name, bus, power, energy, charge_eff=0.95, discharge_eff=0.95):
        """Add a storage unit to the network"""
        if name not in self.storage_units.index:
            self.storage_units.loc[name] = {
                'bus': bus,
                'power': power,
                'energy': energy,
                'charge_efficiency': charge_eff,
                'discharge_efficiency': discharge_eff
            }
        return self
        
    def add_line(self, name, bus_from, bus_to, susceptance, capacity):
        """Add a transmission line to the network"""
        if name not in self.lines.index:
            self.lines.loc[name] = {
                'from': bus_from,
                'to': bus_to,
                'susceptance': susceptance,
                'capacity': capacity
            }
        return self
    
    def dcopf(self):
        """
        DC Optimal Power Flow - solves the optimization problem with DC power flow
        
        This method uses the external optimization module to create and solve
        the DC Optimal Power Flow problem with CPLEX.
            
        Returns:
            bool: True if optimization was successful, False otherwise
        """
        if self.T == 0:
            raise ValueError("No time snapshots defined. Use set_snapshots() first.")
                
        # 1. Create the optimization problem
        problem = create_dcopf_problem(self)
        
        # 2. Solve the problem with CPLEX
        solution = solve_with_cplex(problem)
        
        # 3. Extract and format the results
        if solution['success']:
            extract_results(self, problem, solution)
            return True
        else:
            return False
    
    # Keep lopf as an alias for dcopf for backward compatibility
    def lopf(self):
        """Alias for dcopf"""
        return self.dcopf()
        
    def summary(self):
        """Print a summary of the optimization results"""
        if not hasattr(self, 'objective_value'):
            print("No optimization results available. Run dcopf() first.")
            return
            
        print("Optimal objective value (total cost):", self.objective_value)
        
        print("\nGenerator dispatch:")
        print(self.generators_t['p'])
        
        if self.storage_units_t:
            print("\nStorage charging power:")
            print(self.storage_units_t['p_charge'])
            
            print("\nStorage discharging power:")
            print(self.storage_units_t['p_discharge'])
            
            print("\nStorage state of charge:")
            print(self.storage_units_t['state_of_charge'])
            
        print("\nLine flows:")
        print(self.lines_t['p'])
        
        print("\nBus voltage angles:")
        print(self.buses_t['v_ang']) 