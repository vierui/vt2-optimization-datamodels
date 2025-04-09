#!/usr/bin/env python3
"""
Component classes for power grid network modeling.

This module defines the core component classes used in the power grid optimization model:
- Bus: Represents a node in the power network
- Generator: Represents a power generation unit
- Load: Represents a power demand point
- Storage: Represents an energy storage unit
- Branch: Represents a transmission line connecting two buses
"""

import pandas as pd
import numpy as np

class Bus:
    """Bus component representing a node in the power network"""
    def __init__(self, index, name, v_nom=1.0):
        """
        Initialize a bus component
        
        Args:
            index: Unique identifier for the bus
            name: Descriptive name for the bus
            v_nom: Nominal voltage in per unit (default: 1.0)
        """
        self.index = index
        self.name = name
        self.v_nom = v_nom
        
    @classmethod
    def create_dataframe(cls, buses=None):
        """
        Create a DataFrame for buses
        
        Args:
            buses: List of bus identifiers or dictionary with bus data
            
        Returns:
            pandas DataFrame with bus data
        """
        if isinstance(buses, list):
            return pd.DataFrame(index=buses, columns=['name', 'v_nom'])
        elif isinstance(buses, dict):
            data = {bus_id: {'name': data.get('name', f'Bus {bus_id}'),
                            'v_nom': data.get('v_nom', 1.0)}
                   for bus_id, data in buses.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['name', 'v_nom'])

class Generator:
    """Generator component representing a power generation unit"""
    def __init__(self, index, name, bus, p_nom=0, marginal_cost=0, 
                type='thermal', capex_per_mw=0, lifetime_years=30):
        """
        Initialize a generator component
        
        Args:
            index: Unique identifier for the generator
            name: Descriptive name for the generator
            bus: Bus ID where the generator is connected
            p_nom: Nominal power capacity in MW
            marginal_cost: Marginal cost of generation in currency/MWh
            type: Generator type (e.g., 'solar', 'wind', 'thermal')
            capex_per_mw: Capital expenditure per MW of capacity
            lifetime_years: Expected lifetime of the generator in years
        """
        self.index = index
        self.name = name
        self.bus = bus
        self.p_nom = p_nom
        self.marginal_cost = marginal_cost
        self.type = type
        self.capex_per_mw = capex_per_mw
        self.lifetime_years = lifetime_years
        
    @classmethod
    def create_dataframe(cls, generators=None):
        """
        Create a DataFrame for generators
        
        Args:
            generators: Dictionary with generator data
            
        Returns:
            pandas DataFrame with generator data
        """
        if isinstance(generators, dict):
            data = {gen_id: {
                'name': data.get('name', f'Generator {gen_id}'),
                'bus': data['bus'],
                'p_nom': data.get('p_nom', 0),
                'marginal_cost': data.get('marginal_cost', 0),
                'type': data.get('type', 'thermal'),
                'capex_per_mw': data.get('capex_per_mw', 0),
                'lifetime_years': data.get('lifetime_years', 30)
            } for gen_id, data in generators.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['name', 'bus', 'p_nom', 'marginal_cost',
                                       'type', 'capex_per_mw', 'lifetime_years'])

class Load:
    """Load component representing a power demand point"""
    def __init__(self, index, name, bus, p_set=0):
        """
        Initialize a load component
        
        Args:
            index: Unique identifier for the load
            name: Descriptive name for the load
            bus: Bus ID where the load is connected
            p_set: Power demand in MW
        """
        self.index = index
        self.name = name
        self.bus = bus
        self.p_set = p_set
        
    @classmethod
    def create_dataframe(cls, loads=None):
        """
        Create a DataFrame for loads
        
        Args:
            loads: Dictionary with load data
            
        Returns:
            pandas DataFrame with load data
        """
        if isinstance(loads, dict):
            data = {load_id: {
                'name': data.get('name', f'Load {load_id}'),
                'bus': data['bus'],
                'p_set': data.get('p_set', 0)
            } for load_id, data in loads.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['name', 'bus', 'p_set'])
        
    @classmethod
    def create_time_series(cls, loads, periods):
        """
        Create a time series DataFrame for loads
        
        Args:
            loads: Dictionary with load time series data
            periods: Number of time periods
            
        Returns:
            pandas DataFrame with load time series data
        """
        time_index = pd.RangeIndex(periods)
        loads_t = pd.DataFrame(index=time_index)
        
        if isinstance(loads, dict):
            for load_id, values in loads.items():
                if isinstance(values, (list, np.ndarray)) and len(values) == periods:
                    loads_t[load_id] = values
                else:
                    # Use constant value
                    p_set = values.get('p_set', 0) if isinstance(values, dict) else 0
                    loads_t[load_id] = [p_set] * periods
        
        return loads_t

class Storage:
    """Storage component representing an energy storage unit"""
    def __init__(self, index, name, bus, p_nom=0, 
                efficiency_store=0.9, efficiency_dispatch=0.9,
                max_hours=6, capex_per_mw=0, lifetime_years=15):
        """
        Initialize a storage component
        
        Args:
            index: Unique identifier for the storage
            name: Descriptive name for the storage
            bus: Bus ID where the storage is connected
            p_nom: Nominal power capacity in MW
            efficiency_store: Efficiency for charging (between 0 and 1)
            efficiency_dispatch: Efficiency for discharging (between 0 and 1)
            max_hours: Maximum energy storage in terms of hours at full output
            capex_per_mw: Capital expenditure per MW of capacity
            lifetime_years: Expected lifetime of the storage in years
        """
        self.index = index
        self.name = name
        self.bus = bus
        self.p_nom = p_nom
        self.efficiency_store = efficiency_store
        self.efficiency_dispatch = efficiency_dispatch
        self.max_hours = max_hours
        self.capex_per_mw = capex_per_mw
        self.lifetime_years = lifetime_years
        
    @classmethod
    def create_dataframe(cls, storages=None):
        """
        Create a DataFrame for storage units
        
        Args:
            storages: Dictionary with storage data
            
        Returns:
            pandas DataFrame with storage data
        """
        if isinstance(storages, dict):
            data = {storage_id: {
                'name': data.get('name', f'Storage {storage_id}'),
                'bus': data['bus'],
                'p_nom': data.get('p_nom', 0),
                'efficiency_store': data.get('efficiency_store', 0.9),
                'efficiency_dispatch': data.get('efficiency_dispatch', 0.9),
                'max_hours': data.get('max_hours', 6),
                'capex_per_mw': data.get('capex_per_mw', 0),
                'lifetime_years': data.get('lifetime_years', 15)
            } for storage_id, data in storages.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['name', 'bus', 'p_nom', 
                                        'efficiency_store', 'efficiency_dispatch',
                                        'max_hours', 'capex_per_mw', 'lifetime_years'])

class Branch:
    """Branch component representing a transmission line in the network"""
    def __init__(self, index, name, from_bus, to_bus, x=0.0001, s_nom=0):
        """
        Initialize a branch component
        
        Args:
            index: Unique identifier for the branch
            name: Descriptive name for the branch
            from_bus: Bus ID where the branch starts
            to_bus: Bus ID where the branch ends
            x: Reactance of the branch (in per unit)
            s_nom: Nominal apparent power capacity in MVA
        """
        self.index = index
        self.name = name
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.x = x
        self.s_nom = s_nom
        
    @classmethod
    def create_dataframe(cls, branches=None):
        """
        Create a DataFrame for branches
        
        Args:
            branches: Dictionary with branch data
            
        Returns:
            pandas DataFrame with branch data
        """
        if isinstance(branches, dict):
            data = {branch_id: {
                'name': data.get('name', f'Branch {branch_id}'),
                'from_bus': data['from_bus'],
                'to_bus': data['to_bus'],
                'x': data.get('x', 0.0001),
                's_nom': data.get('s_nom', 0)
            } for branch_id, data in branches.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['name', 'from_bus', 'to_bus', 'x', 's_nom']) 