#!/usr/bin/env python3
"""
Main script for running the power grid optimization using the Network class
with numeric component IDs and standardized data structure
"""
from network import Network
import os

def build_example_network_programmatically():
    """Build an example two-bus network with numeric IDs for all components"""
    # Create an empty network
    net = Network()
    
    # Set time horizon (24 hours)
    net.set_snapshots(24)
    
    # Add buses (numeric IDs)
    net.add_bus(1, "Bus 1")
    net.add_bus(2, "Bus 2")
    
    # Add generators (IDs in 1000s range)
    net.add_generator(1001, "Generator 1", 1, 100, 50)  # 100 MW capacity, 50 €/MWh
    net.add_generator(1002, "Generator 2", 2, 80, 60)   # 80 MW capacity, 60 €/MWh
    
    # Add loads (IDs in 2000s range)
    net.add_load(2001, "Load 1", 1, 30)  # 30 MW load at Bus 1
    net.add_load(2002, "Load 2", 2, 40)  # 40 MW load at Bus 2
    
    # Add storage (IDs in 3000s range)
    net.add_storage(3001, "Storage 1", 1, 50, 100, 0.95, 0.95)  # 50 MW power, 100 MWh capacity
    
    # Add transmission line (IDs in 4000s range)
    net.add_line(4001, "Line 1-2", 1, 2, 1.0, 70)  # 70 MW capacity
    
    return net

def build_example_network_from_csv():
    """Build an example network by loading data from CSV files"""
    # Get script directory and go up to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create a network with data from CSV files
    data_path = os.path.join(project_root, "data", "grid")
    print(f"Looking for data in: {data_path}")
    net = Network(data_path)
    
    return net

def main():
    """
    Main function to run the optimization
    """
    # Build the network (either programmatically or from CSV)
    print("Building network...")
    
    # Try loading from CSV first
    try:
        net = build_example_network_from_csv()
        data_source = "CSV files"
    except Exception as e:
        print(f"Could not load from CSV: {e}")
        print("Building network programmatically instead...")
        net = build_example_network_programmatically()
        data_source = "programmatic definition"
    
    # Print network structure
    print(f"Network structure created from {data_source}:")
    print(f"- Buses: {net.buses.index.tolist()}")
    print(f"- Generators: {net.generators.index.tolist()}")
    print(f"- Loads: {net.loads.index.tolist()}")
    print(f"- Storage units: {net.storage_units.index.tolist()}")
    print(f"- Lines: {net.lines.index.tolist()}")
    
    # Run DC Optimal Power Flow with CPLEX
    print("\nSolving DC optimal power flow problem with CPLEX...")
    success = net.dcopf()
    
    if success:
        # Print results summary
        net.summary()
        return True
    else:
        print("Optimization failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 