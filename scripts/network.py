#!/usr/bin/env python3
"""
Network module for power grid optimization

Contains the IntegratedNetwork class that integrates multiple 
seasonal subnetworks into a unified optimization model.
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import field
from loader import load_grid_data, load_day_profiles
from optimization import create_dcopf_problem, solve_with_cplex, extract_results
from components import Bus, Branch, Generator, Storage, Load

class IntegratedNetwork:
    """
    Integrated network model that combines multiple seasonal grid models 
    with shared asset investment decisions.
    
    This class maintains seasonal subnetworks while ensuring consistency in
    asset investment decisions across all seasons.
    """
    def __init__(self, seasons=None, years=None, discount_rate=0.05, season_weights=None):
        """
        Initialize the integrated network
        
        Args:
            seasons: List of season names
            years: List of planning years
            discount_rate: Discount rate for multi-year planning
            season_weights: Dictionary mapping season names to number of weeks
        """
        # Initialize with empty seasons list if None
        self.seasons = seasons or []
        self.years = years or []
        self.discount_rate = discount_rate
        self.season_networks = {}
        
        # Season weights for calculating annual costs
        self.season_weights = season_weights or {
            'winter': 13,     # Winter represents 13 weeks
            'summer': 13,     # Summer represents 13 weeks
            'spri_autu': 26   # Spring/Autumn represents 26 weeks
        }
        
        # Initialize asset installation trackers
        self.asset_installation = {
            'generators': {year: {} for year in self.years},
            'storage': {year: {} for year in self.years}
        }
        
        # Track replacement and first installation decisions by year
        self.asset_installation_history = {
            'generators': {},
            'storage': {}
        }
        
    def add_season_network(self, season, network):
        """
        Add a seasonal network to the integrated model
        
        Args:
            season: Season name
            network: Network object for this season
            
        Returns:
            True if successful
        """
        if season not in self.seasons:
            self.seasons.append(season)
            
        self.season_networks[season] = network
        
        # Add generators to installation trackers
        for gen_id in network.generators.index:
            for year in self.years:
                if gen_id not in self.asset_installation['generators'][year]:
                    self.asset_installation['generators'][year][gen_id] = 0
                    
                if gen_id not in self.asset_installation_history['generators']:
                    self.asset_installation_history['generators'][gen_id] = []
                    
        # Add storage units to installation trackers
        for storage_id in network.storage_units.index:
            for year in self.years:
                if storage_id not in self.asset_installation['storage'][year]:
                    self.asset_installation['storage'][year][storage_id] = 0
                    
                if storage_id not in self.asset_installation_history['storage']:
                    self.asset_installation_history['storage'][storage_id] = []
                    
        return True
        
    def get_season_network(self, season):
        """
        Get the network for a specific season
        
        Args:
            season: Season name
            
        Returns:
            Network object for the specified season
        """
        return self.season_networks.get(season)
        
    def set_optimization_results(self, results):
        """
        Set optimization results from the integrated model solution
        
        Args:
            results: Results dictionary from the solver
            
        Returns:
            True if successful
        """
        # Store the integrated results
        self.integrated_results = results
        
        # Set asset installation decisions
        for year in self.years:
            for gen_id, installed in results['generators_installed'][year].items():
                self.asset_installation['generators'][year][gen_id] = installed
                
            for storage_id, installed in results['storage_installed'][year].items():
                self.asset_installation['storage'][year][storage_id] = installed
                
        # Store first installation info
        if 'generators_first_install' in results:
            for year in self.years:
                for gen_id, is_first in results['generators_first_install'][year].items():
                    if is_first > 0.5:  # Binary variable threshold
                        installation_info = {
                            'installation_year': year,
                            'generator_id': gen_id,
                            'is_replacement': False
                        }
                        
                        # Check if it's a replacement
                        if 'generators_replacement' in results and gen_id in results['generators_replacement'][year]:
                            if results['generators_replacement'][year][gen_id] > 0.5:
                                installation_info['is_replacement'] = True
                                
                        self.asset_installation_history['generators'][gen_id].append(installation_info)
        
        # Store first installation info for storage
        if 'storage_first_install' in results:
            for year in self.years:
                for storage_id, is_first in results['storage_first_install'][year].items():
                    if is_first > 0.5:  # Binary variable threshold
                        installation_info = {
                            'installation_year': year,
                            'storage_id': storage_id,
                            'is_replacement': False
                        }
                        
                        # Check if it's a replacement
                        if 'storage_replacement' in results and storage_id in results['storage_replacement'][year]:
                            if results['storage_replacement'][year][storage_id] > 0.5:
                                installation_info['is_replacement'] = True
                                
                        self.asset_installation_history['storage'][storage_id].append(installation_info)
        
        # Set up seasonal results
        for season, network in self.season_networks.items():
            # Set generator outputs by year
            if 'generators_p' in results and season in results['generators_p']:
                network.generators_t_by_year = {}
                
                for year in self.years:
                    network.generators_t_by_year[year] = {'p': {}}
                    
                    for gen_id in network.generators.index:
                        # Check if this generator has output for this year/season
                        if gen_id in results['generators_p'][season][year]:
                            p_values = results['generators_p'][season][year][gen_id]
                            network.generators_t_by_year[year]['p'][gen_id] = pd.Series(p_values)
                            
            # Set storage outputs by year
            if 'storage_p_charge' in results and season in results['storage_p_charge']:
                network.storage_units_t_by_year = {}
                
                for year in self.years:
                    network.storage_units_t_by_year[year] = {
                        'p_charge': {},
                        'p_discharge': {},
                        'soc': {}
                    }
                    
                    for storage_id in network.storage_units.index:
                        # Check if this storage has output for this year/season
                        if storage_id in results['storage_p_charge'][season][year]:
                            charge_values = results['storage_p_charge'][season][year][storage_id]
                            discharge_values = results['storage_p_discharge'][season][year][storage_id]
                            soc_values = results['storage_soc'][season][year][storage_id]
                            
                            network.storage_units_t_by_year[year]['p_charge'][storage_id] = pd.Series(charge_values)
                            network.storage_units_t_by_year[year]['p_discharge'][storage_id] = pd.Series(discharge_values)
                            network.storage_units_t_by_year[year]['soc'][storage_id] = pd.Series(soc_values)
            
            # Set generators installed status
            network.generators_installed_by_year = {}
            for year in self.years:
                network.generators_installed_by_year[year] = results['generators_installed'][year]
                
            # Set storage installed status
            network.storage_installed_by_year = {}
            for year in self.years:
                network.storage_installed_by_year[year] = results['storage_installed'][year]
        
        # Store seasonal costs if available
        if 'season_costs' in results:
            self.seasons_total_cost = results['season_costs']
            
        # Store operational and capital costs if available
        if 'operational_costs' in results:
            self.operational_costs = results['operational_costs']
            
        if 'capital_costs' in results:
            self.capital_costs = results['capital_costs']
            
        return True
        
    def save_to_pickle(self, filename):
        """
        Save the network to a pickle file
        
        Args:
            filename: Path to save the file
            
        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
                
            print(f"Network saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving network: {e}")
            return False
            
    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load a network from a pickle file
        
        Args:
            filename: Path to the pickle file
            
        Returns:
            Network object or None if unsuccessful
        """
        try:
            with open(filename, 'rb') as f:
                network = pickle.load(f)
                
            print(f"Network loaded from {filename}")
            return network
            
        except Exception as e:
            print(f"Error loading network: {e}")
            return None
                
    def get_common_generators(self):
        """
        Get a list of generators common to all seasonal networks
        
        Returns:
            List of generator IDs
        """
        if not self.season_networks:
            return []
            
        # Get generators from the first season
        first_season = list(self.season_networks.keys())[0]
        common_generators = set(self.season_networks[first_season].generators.index)
        
        # Intersect with generators from other seasons
        for season, network in self.season_networks.items():
            if season != first_season:
                common_generators &= set(network.generators.index)
                
        return list(common_generators)
        
    def get_common_storage_units(self):
        """
        Get a list of storage units common to all seasonal networks
        
        Returns:
            List of storage unit IDs
        """
        if not self.season_networks:
            return []
            
        # Get storage units from the first season
        first_season = list(self.season_networks.keys())[0]
        common_storage = set(self.season_networks[first_season].storage_units.index)
        
        # Intersect with storage units from other seasons
        for season, network in self.season_networks.items():
            if season != first_season:
                common_storage &= set(network.storage_units.index)
                
        return list(common_storage)
        
    def get_annual_cost(self):
        """
        Calculate the annual cost based on seasonal costs and weights
        
        Returns:
            Total annual cost
        """
        if not hasattr(self, 'seasons_total_cost'):
            return None
            
        annual_cost = 0
        for season, cost in self.seasons_total_cost.items():
            weeks = self.season_weights.get(season, 0)
            annual_cost += weeks * cost
            
        return annual_cost

class Network:
    """
    Power grid network representation for a single season
    
    This class is used as a subnetwork within the IntegratedNetwork
    """
    def __init__(self, name=""):
        self.name = name
        self.buses = pd.DataFrame()
        self.branches = pd.DataFrame()
        self.lines = self.branches  # Add this line to make lines an alias for branches
        self.generators = pd.DataFrame()
        self.storage_units = pd.DataFrame()
        self.loads = pd.DataFrame()
        
        # Time series data
        self.snapshots = pd.DatetimeIndex([])
        self.loads_t = {'p': {}, 'q': {}}
        self.generators_t = {'p': {}, 'q': {}}
        self.storage_units_t = {'p_charge': {}, 'p_discharge': {}, 'soc': {}}
        
        # Optimization trackers for multi-year planning
        self.generators_t_by_year = {}
        self.storage_units_t_by_year = {}
        self.loads_t_by_year = {}
        self.generators_installed_by_year = {}
        self.storage_installed_by_year = {}
    
    def create_snapshots(self, start_time, periods, freq):
        """
        Create snapshots for time-series data
        
        Args:
            start_time: Start time for snapshots
            periods: Number of periods
            freq: Frequency (e.g., 'H' for hourly)
            
        Returns:
            DatetimeIndex of snapshots
        """
        self.snapshots = pd.date_range(start=start_time, periods=periods, freq=freq)
        self.T = len(self.snapshots)
        return self.snapshots
        
    def add_load_time_series(self, load_id, p_values, q_values=None):
        """
        Add time series data for a load
        
        Args:
            load_id: Load ID
            p_values: Active power values
            q_values: Reactive power values (optional)
            
        Returns:
            True if successful
        """
        # Check if load exists
        if load_id not in self.loads.index:
            print(f"Error: Load {load_id} does not exist")
            return False
            
        # Check length of values matches snapshots
        if len(p_values) != len(self.snapshots):
            print(f"Error: Length of p_values ({len(p_values)}) does not match snapshots ({len(self.snapshots)})")
            return False
            
        # Add active power values
        self.loads_t['p'][load_id] = pd.Series(p_values, index=self.snapshots)
        
        # Add reactive power values if provided
        if q_values is not None:
            if len(q_values) != len(self.snapshots):
                print(f"Error: Length of q_values ({len(q_values)}) does not match snapshots ({len(self.snapshots)})")
                return False
                
            self.loads_t['q'][load_id] = pd.Series(q_values, index=self.snapshots)
            
        return True
            
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
        
    def add_generator(self, id, name, bus, capacity, cost, gen_type, capex_per_mw, lifetime_years, discount_rate=None):
        """
        Add a generator to the network with numeric ID
        
        All parameters must be explicitly provided except discount_rate
        """
        if id not in self.generators.index:
            self.generators.loc[id] = {
                'name': name,
                'bus_id': bus,
                'capacity_mw': capacity,
                'cost_mwh': cost,
                'type': gen_type,
                'capex_per_mw': capex_per_mw,
                'lifetime_years': lifetime_years,
                'discount_rate': discount_rate  # Asset-specific discount rate
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
        
    def add_storage(self, id, name, bus, p_mw, energy_mwh, charge_eff, discharge_eff, capex_per_mw, lifetime_years, discount_rate=None):
        """
        Add a storage unit to the network with numeric ID
        
        All parameters must be explicitly provided except discount_rate
        """
        if id not in self.storage_units.index:
            self.storage_units.loc[id] = {
                'name': name,
                'bus_id': bus,
                'p_mw': p_mw,
                'energy_mwh': energy_mwh,
                'efficiency_store': charge_eff,
                'efficiency_dispatch': discharge_eff,
                'capex_per_mw': capex_per_mw,
                'lifetime_years': lifetime_years,
                'discount_rate': discount_rate  # Asset-specific discount rate
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

    def set_planning_horizon(self, years, discount_rate=0.05):
        """
        Set up multi-year planning horizon
        
        Args:
            years: List of years in the planning horizon, e.g., [2023, 2024, 2025]
            discount_rate: Discount rate for future costs (default: 5%)
            
        Returns:
            Self for method chaining
        """
        self.years = years
        self.year_weights = {year: 1.0 for year in years}  # Default equal weights
        self.discount_rate = discount_rate
        print(f"Set up planning horizon with {len(years)} years, discount rate: {discount_rate:.1%}")
        return self
        
    def set_year_weights(self, weights):
        """
        Set weights for each year in the planning horizon
        
        Args:
            weights: Dictionary mapping years to weights
            
        Returns:
            Self for method chaining
        """
        if not self.years:
            print("Warning: Planning horizon not set. Call set_planning_horizon() first.")
            return self
            
        for year, weight in weights.items():
            if year in self.years:
                self.year_weights[year] = weight
                
        print(f"Set year weights: {self.year_weights}")
        return self 