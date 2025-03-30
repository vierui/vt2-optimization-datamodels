#!/usr/bin/env python3
"""
Script to check grid connections and topology to diagnose optimization issues
"""
import os
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.network import Network

def load_network(season="winter"):
    """
    Load network from pickle file
    
    Args:
        season: Season to load
        
    Returns:
        Network object or None if error
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results/annual")
    
    # Load the network
    network_file = os.path.join(results_dir, f"{season}_network.pkl")
    return Network.load_from_pickle(network_file)

def load_grid_data():
    """Load grid data directly from CSV files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    grid_dir = os.path.join(project_root, "data/grid")
    
    # Load components
    buses = pd.read_csv(os.path.join(grid_dir, "buses.csv"))
    generators = pd.read_csv(os.path.join(grid_dir, "generators.csv"))
    loads = pd.read_csv(os.path.join(grid_dir, "loads.csv"))
    lines = pd.read_csv(os.path.join(grid_dir, "lines.csv"))
    
    return {
        'buses': buses,
        'generators': generators,
        'loads': loads,
        'lines': lines
    }

def analyze_grid_connections():
    """
    Analyze the grid connections to identify potential issues
    """
    # Load grid data
    grid_data = load_grid_data()
    
    # Get components
    buses = grid_data['buses']
    generators = grid_data['generators']
    loads = grid_data['loads']
    lines = grid_data['lines']
    
    print("\n=== GRID CONNECTION ANALYSIS ===")
    
    # Create a graph of the network
    G = nx.Graph()
    
    # Add buses to the graph
    for _, bus in buses.iterrows():
        G.add_node(bus['id'], type='bus', name=bus['name'])
    
    # Add lines to the graph
    for _, line in lines.iterrows():
        G.add_edge(line['bus_from'], line['bus_to'], 
                  capacity=line['capacity_mw'], 
                  susceptance=line['susceptance'])
    
    # Check connectivity
    if not nx.is_connected(G):
        print("WARNING: The grid is not connected! Found separate components.")
        components = list(nx.connected_components(G))
        for i, component in enumerate(components):
            print(f"Component {i+1}: Buses {component}")
    else:
        print("Grid is fully connected.")
    
    # Check if generators and loads are on the same bus, or connected buses
    print("\n=== GENERATOR AND LOAD CONNECTIVITY ===")
    
    # Map buses to loads and generators
    bus_to_gen = {}
    bus_to_load = {}
    
    for _, gen in generators.iterrows():
        bus_id = gen['bus_id']
        if bus_id not in bus_to_gen:
            bus_to_gen[bus_id] = []
        bus_to_gen[bus_id].append(gen['id'])
    
    for _, load in loads.iterrows():
        bus_id = load['bus_id']
        if bus_id not in bus_to_load:
            bus_to_load[bus_id] = []
        bus_to_load[bus_id].append(load['id'])
    
    # Check if loads and generators are connected
    load_buses = set(bus_to_load.keys())
    gen_buses = set(bus_to_gen.keys())
    
    # Print bus presence in graph
    print("\nChecking buses in graph:")
    for bus_id in sorted(set(load_buses) | set(gen_buses)):
        if bus_id in G:
            print(f"Bus {bus_id} is in the graph.")
        else:
            print(f"WARNING: Bus {bus_id} is NOT in the graph!")
    
    # Print load buses and connected generators
    print("\nLoad buses and connected generators:")
    for bus_id in sorted(load_buses):
        print(f"Bus {bus_id} has loads: {bus_to_load.get(bus_id, [])}")
        
        # Check if this bus has generators
        if bus_id in gen_buses:
            print(f"  - Direct generators: {bus_to_gen[bus_id]}")
        else:
            # Check if this bus is in the graph
            if bus_id not in G:
                print(f"  - WARNING: Bus {bus_id} is not in the grid graph!")
                continue
                
            # Check if this bus can reach any generator buses
            connected_to_gen = False
            for gen_bus in gen_buses:
                # Make sure gen_bus is in the graph
                if gen_bus not in G:
                    print(f"  - WARNING: Generator bus {gen_bus} is not in the grid graph!")
                    continue
                
                # Check path existence
                if nx.has_path(G, bus_id, gen_bus):
                    path = nx.shortest_path(G, bus_id, gen_bus)
                    print(f"  - Can reach generators on bus {gen_bus} via path: {path}")
                    connected_to_gen = True
            
            if not connected_to_gen:
                print("  - WARNING: This load bus cannot reach any generators!")
    
    # Print list of all buses and their connections
    print("\nBus connection summary:")
    for bus_id in sorted(buses['id']):
        if bus_id not in G:
            print(f"Bus {bus_id}: NOT CONNECTED")
            continue
            
        neighbors = list(G.neighbors(bus_id))
        print(f"Bus {bus_id}: Connected to {neighbors}")
        
        # Print what's at this bus
        if bus_id in bus_to_gen:
            print(f"  - Has generators: {bus_to_gen[bus_id]}")
        if bus_id in bus_to_load:
            print(f"  - Has loads: {bus_to_load[bus_id]}")
    
    return grid_data

def check_missing_components(grid_data, network):
    """
    Check for any components missing in the network vs grid data
    """
    print("\n=== COMPONENT COMPARISON ===")
    
    # Check buses
    grid_buses = set(grid_data['buses']['id'])
    network_buses = set(network.buses.index)
    if grid_buses != network_buses:
        print("Buses mismatch!")
        missing_buses = grid_buses - network_buses
        if missing_buses:
            print(f"  Missing buses in network: {missing_buses}")
        extra_buses = network_buses - grid_buses
        if extra_buses:
            print(f"  Extra buses in network: {extra_buses}")
    else:
        print("All buses correctly added to the network.")
    
    # Check generators
    grid_gens = set(grid_data['generators']['id'])
    network_gens = set(network.generators.index)
    if grid_gens != network_gens:
        print("Generators mismatch!")
        missing_gens = grid_gens - network_gens
        if missing_gens:
            print(f"  Missing generators in network: {missing_gens}")
        extra_gens = network_gens - grid_gens
        if extra_gens:
            print(f"  Extra generators in network: {extra_gens}")
    else:
        print("All generators correctly added to the network.")
    
    # Check loads
    grid_loads = set(grid_data['loads']['id'])
    network_loads = set(network.loads.index)
    if grid_loads != network_loads:
        print("Loads mismatch!")
        missing_loads = grid_loads - network_loads
        if missing_loads:
            print(f"  Missing loads in network: {missing_loads}")
        extra_loads = network_loads - grid_loads
        if extra_loads:
            print(f"  Extra loads in network: {extra_loads}")
    else:
        print("All loads correctly added to the network.")
    
    # Check lines
    grid_lines = set(grid_data['lines']['id'])
    network_lines = set(network.lines.index)
    if grid_lines != network_lines:
        print("Lines mismatch!")
        missing_lines = grid_lines - network_lines
        if missing_lines:
            print(f"  Missing lines in network: {missing_lines}")
        extra_lines = network_lines - grid_lines
        if extra_lines:
            print(f"  Extra lines in network: {extra_lines}")
    else:
        print("All lines correctly added to the network.")

def main():
    """Main function"""
    # Analyze grid connections
    grid_data = analyze_grid_connections()
    
    # Load a network from results for comparison
    network = load_network()
    
    if network is not None:
        # Check for missing components
        check_missing_components(grid_data, network)
    
if __name__ == "__main__":
    main() 