import pandas as pd
import numpy as np
import cvxpy as cp
import pickle
from loader import load_grid_data, load_day_profiles
from optimization import create_dcopf_problem, solve_with_cplex, extract_results

class Network:
    """
    Network class that holds all power system components and handles optimization
    Similar to PyPSA's Network class with numeric IDs for components
    """
    def __init__(self, data_dir=None, use_day_profiles=False, day=10):
        """
        Initialize the network, optionally loading data from CSV files
        
        Args:
            data_dir: Directory containing grid component CSV files
            use_day_profiles: Whether to use time-dependent profiles from processed data
            day: Day of the year (1-365) to use for time profiles
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
        self.generators_t = {}  # Will store results after optimization
        self.storage_units_t = {}
        self.lines_t = {}
        self.buses_t = {}
        
        # Day profiles
        self.use_day_profiles = use_day_profiles
        self.day = day
        self.day_profiles = None
        
        # Load data if directory provided
        if data_dir:
            self.import_from_csv(data_dir)
            
        # Load day profiles if requested
        if use_day_profiles:
            self.load_day_profiles(day)
            
    def load_day_profiles(self, day=10):
        """
        Load time-dependent profiles for the specified day
        
        Args:
            day: Day of the year (1-365)
        """
        self.day_profiles = load_day_profiles(day)
        
        # Update time settings
        self.T = self.day_profiles['T']
        self.snapshots = pd.RangeIndex(self.T)
        
        # Create time-dependent availability factors for generators
        self.gen_p_max_pu = pd.DataFrame(index=self.snapshots)
        
        # Assign wind/solar profiles to generators based on their type (if specified in the CSV)
        for gen_id, gen_data in self.generators.iterrows():
            gen_type = gen_data.get('type', 'thermal')
            
            if gen_type == 'wind':
                # Wind generators use wind profile
                self.gen_p_max_pu[gen_id] = self.day_profiles['wind'] / 100.0
            elif gen_type == 'solar':
                # Solar generators use solar profile
                self.gen_p_max_pu[gen_id] = self.day_profiles['solar'] / 100.0
            else:
                # Thermal generators are always available
                self.gen_p_max_pu[gen_id] = [1.0] * self.T
        
        # Update load time series with day profile
        # Scale the nominal load values by the profile
        load_factors = self.day_profiles['load'] / 100.0  # Normalize to percentage
        self.loads_t = pd.DataFrame(index=self.snapshots)
        
        for load_id, load_data in self.loads.iterrows():
            nom_load = load_data['p_mw']
            self.loads_t[load_id] = nom_load * load_factors
            
        print(f"Loaded time-dependent profiles for day {day}")
        
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
        
        # Set time horizon if not using day profiles
        if not self.use_day_profiles:
            self.loads_t = data['loads_t']
            self.T = data['T']
            self.snapshots = pd.RangeIndex(self.T)
                
    def set_snapshots(self, T):
        """
        Set the time snapshots for the optimization
        
        This initializes the time dimension of the network and creates
        the necessary time series data structures (gen_p_max_pu and loads_t).
        
        Args:
            T: Number of time steps
            
        Returns:
            None
        """
        self.T = T
        self.snapshots = pd.RangeIndex(T)
        
        # Initialize generator availability
        self.gen_p_max_pu = pd.DataFrame(index=self.snapshots)
        if not self.generators.empty:
            for gen_id in self.generators.index:
                self.gen_p_max_pu[gen_id] = [1.0] * T
        
        # Initialize loads
        if self.loads_t.empty and not self.loads.empty:
            self.loads_t = pd.DataFrame(index=self.snapshots)
            for load_id in self.loads.index:
                load_p = self.loads.loc[load_id, 'p_mw']
                self.loads_t[load_id] = [load_p] * T
        
    def add_bus(self, id, name, v_nom=1.0):
        """Add a bus to the network with numeric ID"""
        if id not in self.buses.index:
            self.buses.loc[id] = {'name': name, 'v_nom': v_nom}
        return self
        
    def add_generator(self, id, name, bus, capacity, cost, gen_type='thermal', capex_per_mw=0, lifetime_years=25):
        """Add a generator to the network with numeric ID"""
        if id not in self.generators.index:
            self.generators.loc[id] = {
                'name': name,
                'bus_id': bus,
                'capacity_mw': capacity,
                'cost_mwh': cost,
                'type': gen_type,
                'capex_per_mw': capex_per_mw,
                'lifetime_years': lifetime_years
            }
            
            # Initialize availability factor if time series exists
            if hasattr(self, 'gen_p_max_pu') and self.T > 0:
                if gen_type == 'wind' and self.use_day_profiles:
                    self.gen_p_max_pu[id] = self.day_profiles['wind'] / 100.0
                elif gen_type == 'solar' and self.use_day_profiles:
                    self.gen_p_max_pu[id] = self.day_profiles['solar'] / 100.0
                else:
                    self.gen_p_max_pu[id] = [1.0] * self.T
                    
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
                
            if self.use_day_profiles:
                # Scale the load by the day profile
                load_factors = self.day_profiles['load'] / 100.0
                self.loads_t[id] = p_mw * load_factors
            else:
                # Use constant load
                self.loads_t[id] = [p_mw] * self.T
                
        return self
        
    def add_storage(self, id, name, bus, p_mw, energy_mwh, charge_eff=0.95, discharge_eff=0.95, capex_per_mw=0, lifetime_years=15):
        """Add a storage unit to the network with numeric ID"""
        if id not in self.storage_units.index:
            self.storage_units.loc[id] = {
                'name': name,
                'bus_id': bus,
                'p_mw': p_mw,
                'energy_mwh': energy_mwh,
                'efficiency_store': charge_eff,
                'efficiency_dispatch': discharge_eff,
                'capex_per_mw': capex_per_mw,
                'lifetime_years': lifetime_years
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
        Run DC optimal power flow and extract results
        
        Returns:
            True if optimization successful, False otherwise
        """
        # Ensure we have time settings
        if self.T <= 0:
            print("Error: Time horizon not set. Call set_snapshots() first.")
            return False
        
        # Ensure we have all required data
        if self.buses.empty:
            print("Error: No buses defined.")
            return False
        
        if self.generators.empty:
            print("Error: No generators defined.")
            return False
        
        if self.loads.empty or self.loads_t.empty:
            print("Error: No loads defined or no load time series.")
            return False
        
        if self.lines.empty:
            print("Error: No lines defined.")
            return False
        
        # Ensure gen_p_max_pu is properly initialized
        if self.gen_p_max_pu.empty:
            self.gen_p_max_pu = pd.DataFrame(index=self.snapshots)
            for gen_id in self.generators.index:
                self.gen_p_max_pu[gen_id] = [1.0] * self.T
        
        # Create and solve the problem
        print("Solving with CPLEX...")
        problem = create_dcopf_problem(self)
        solution = solve_with_cplex(problem)
        
        if solution['success']:
            # Extract results
            extract_results(self, problem, solution)
            return True
        else:
            print("Optimization failed!")
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
        
    def save_to_pickle(self, filepath):
        """
        Save the network object to a pickle file
        
        Args:
            filepath: Path to save the pickle file
            
        Returns:
            True if successful, False otherwise
        """
        import pickle
        import os
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the network to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            return True
        except Exception as e:
            print(f"Error saving network to pickle: {e}")
            return False
    
    @staticmethod
    def load_from_pickle(filepath):
        """
        Load a network object from a pickle file
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            Network object or None if loading fails
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            print(f"Error: Pickle file not found: {filepath}")
            return None
            
        try:
            with open(filepath, 'rb') as f:
                network = pickle.load(f)
            
            return network
        except Exception as e:
            print(f"Error loading network from pickle: {e}")
            return None 