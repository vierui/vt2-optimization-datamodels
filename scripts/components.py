import pandas as pd
import numpy as np

# Component classes for the network

class Bus:
    """Bus component for the network"""
    def __init__(self, name):
        self.name = name
        
    @classmethod
    def create_dataframe(cls, buses):
        """Create a DataFrame for buses"""
        if isinstance(buses, list):
            # Initialize with at least one column 'name' to avoid 'no defined columns' error
            return pd.DataFrame(index=buses, columns=['name'])
        else:
            return pd.DataFrame(columns=['name'])


class Generator:
    """Generator component for the network"""
    def __init__(self, name, bus, capacity, cost):
        self.name = name
        self.bus = bus
        self.capacity = capacity
        self.cost = cost
        
    @classmethod
    def create_dataframe(cls, gens):
        """Create a DataFrame for generators"""
        if isinstance(gens, dict):
            data = {name: {'bus': data['bus'], 
                           'capacity': data['capacity'], 
                           'cost': data['cost']} 
                   for name, data in gens.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['bus', 'capacity', 'cost'])


class Load:
    """Load component for the network"""
    def __init__(self, bus, p_set):
        self.bus = bus
        self.p_set = p_set
        
    @classmethod
    def create_dataframe(cls, loads, T):
        """Create a DataFrame for loads"""
        if isinstance(loads, dict):
            data = {}
            for bus, load_values in loads.items():
                data[bus] = {'bus': bus}
            df = pd.DataFrame.from_dict(data, orient='index')
            
            # Create time-series DataFrame
            time_index = pd.RangeIndex(T)
            loads_t = pd.DataFrame(index=time_index)
            
            for bus, load_values in loads.items():
                loads_t[bus] = load_values
                
            return df, loads_t
        else:
            return pd.DataFrame(columns=['bus']), pd.DataFrame()


class Storage:
    """Storage component for the network"""
    def __init__(self, name, bus, power, energy, charge_eff, discharge_eff):
        self.name = name
        self.bus = bus
        self.power = power
        self.energy = energy
        self.charge_efficiency = charge_eff
        self.discharge_efficiency = discharge_eff
        
    @classmethod
    def create_dataframe(cls, storages):
        """Create a DataFrame for storage units"""
        if isinstance(storages, dict):
            data = {name: {'bus': data['bus'], 
                           'power': data['power'], 
                           'energy': data['energy'],
                           'charge_efficiency': data['charge_efficiency'],
                           'discharge_efficiency': data['discharge_efficiency']} 
                   for name, data in storages.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['bus', 'power', 'energy', 
                                        'charge_efficiency', 'discharge_efficiency'])


class Line:
    """Line component for the network"""
    def __init__(self, name, bus_from, bus_to, susceptance, capacity):
        self.name = name
        self.bus_from = bus_from
        self.bus_to = bus_to
        self.susceptance = susceptance
        self.capacity = capacity
        
    @classmethod
    def create_dataframe(cls, lines):
        """Create a DataFrame for transmission lines"""
        if isinstance(lines, dict):
            data = {name: {'from': data['from'], 
                           'to': data['to'], 
                           'susceptance': data['susceptance'],
                           'capacity': data['capacity']} 
                   for name, data in lines.items()}
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            return pd.DataFrame(columns=['from', 'to', 'susceptance', 'capacity']) 