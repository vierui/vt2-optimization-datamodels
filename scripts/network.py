import pandas as pd
import numpy as np
import cvxpy as cp
from loader import load_grid_data
from optimization import create_dcopf_problem, solve_with_cplex, extract_results

class Network:
    """
    Network class that holds all power system components and handles optimization
    Similar to PyPSA's Network class with numeric IDs for components
    """
    def __init__(self, data_dir=None):
        """
        Initialize the network, optionally loading data from CSV files
        
        Args:
            data_dir: Directory containing grid component CSV files
        """
        # Time settings
        self.snapshots = None
        self.T = 0
        
        # Static component DataFrames - now using numeric IDs
        self.buses = pd.DataFrame(columns=['name', 'v_nom'])
        self.generators = pd.DataFrame(columns=['name', 'bus_id', 'capacity_mw', 'cost_mwh'])
        self.loads = pd.DataFrame(columns=['name', 'bus_id', 'p_mw'])
        self.storage_units = pd.DataFrame(columns=['name', 'bus_id', 'p_mw', 'energy_mwh', 
                                                 'efficiency_store', 'efficiency_dispatch'])
        self.lines = pd.DataFrame(columns=['name', 'bus_from', 'bus_to', 'susceptance', 'capacity_mw'])
        
        # Time-series component DataFrames
        self.loads_t = pd.DataFrame()
        
        # Results storage
        self.generators_t = {}
        self.storage_units_t = {}
        self.lines_t = {}
        self.buses_t = {}
        
        # Load data if directory provided
        if data_dir:
            self.import_from_csv(data_dir)
            
    def import_from_csv(self, data_dir="data/grid"):
        """
        Import network data from CSV files
        
        Args:
            data_dir: Directory containing grid component CSV files
        """
        data = load_grid_data(data_dir)
        
        # Set component DataFrames
        self.buses = data['buses']
        self.generators = data['generators']
        self.loads = data['loads']
        self.storage_units = data['storage_units']
        self.lines = data['lines']
        self.loads_t = data['loads_t']
        
        # Set time horizon
        self.T = data['T']
        self.snapshots = pd.RangeIndex(self.T)
                
    def set_snapshots(self, T):
        """Set the time snapshots for the model"""
        self.T = T
        self.snapshots = pd.RangeIndex(T)
        
    def add_bus(self, id, name, v_nom=1.0):
        """Add a bus to the network with numeric ID"""
        if id not in self.buses.index:
            self.buses.loc[id] = {'name': name, 'v_nom': v_nom}
        return self
        
    def add_generator(self, id, name, bus, capacity, cost, gen_type='thermal'):
        """Add a generator to the network with numeric ID"""
        if id not in self.generators.index:
            self.generators.loc[id] = {
                'name': name,
                'bus_id': bus,
                'capacity_mw': capacity,
                'cost_mwh': cost
            }
        return self
        
    def add_load(self, id, name, bus, p_mw):
        """Add a load to the network with numeric ID"""
        if id not in self.loads.index:
            self.loads.loc[id] = {
                'name': name,
                'bus_id': bus,
                'p_mw': p_mw
            }
            
        # Set or update the load time series
        if self.T > 0:
            # Make sure loads_t has an index
            if self.loads_t.empty:
                self.loads_t = pd.DataFrame(index=range(self.T))
            self.loads_t[id] = [p_mw] * self.T
                
        return self
        
    def add_storage(self, id, name, bus, p_mw, energy_mwh, charge_eff=0.95, discharge_eff=0.95):
        """Add a storage unit to the network with numeric ID"""
        if id not in self.storage_units.index:
            self.storage_units.loc[id] = {
                'name': name,
                'bus_id': bus,
                'p_mw': p_mw,
                'energy_mwh': energy_mwh,
                'efficiency_store': charge_eff,
                'efficiency_dispatch': discharge_eff
            }
        return self
        
    def add_line(self, id, name, bus_from, bus_to, susceptance, capacity):
        """Add a transmission line to the network with numeric ID"""
        if id not in self.lines.index:
            self.lines.loc[id] = {
                'name': name,
                'bus_from': bus_from,
                'bus_to': bus_to,
                'susceptance': susceptance,
                'capacity_mw': capacity
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
        
        print("\nGenerator dispatch (MW):")
        print(self.generators_t['p'])
        
        if self.storage_units_t:
            print("\nStorage charging power (MW):")
            print(self.storage_units_t['p_charge'])
            
            print("\nStorage discharging power (MW):")
            print(self.storage_units_t['p_discharge'])
            
            print("\nStorage state of charge (MWh):")
            print(self.storage_units_t['state_of_charge'])
            
        print("\nLine flows (MW):")
        print(self.lines_t['p'])
        
        print("\nBus voltage angles (rad):")
        print(self.buses_t['v_ang']) 