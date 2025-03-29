#!/usr/bin/env python3
"""
Main script for running the power grid optimization using the Network class
"""
from network import Network

def build_example_network():
    """Build an example two-bus network with generation, storage, and transmission"""
    # Create an empty network
    net = Network()
    
    # Set time horizon (24 hours)
    net.set_snapshots(24)
    
    # Add buses
    net.add_bus('Bus1')
    net.add_bus('Bus2')
    
    # Add generators
    net.add_generator('Gen1', 'Bus1', 100, 50)  # 100 MW capacity, 50 €/MWh
    net.add_generator('Gen2', 'Bus2', 80, 60)   # 80 MW capacity, 60 €/MWh
    
    # Add loads
    net.add_load('Bus1', 30)  # 30 MW load at Bus1
    net.add_load('Bus2', 40)  # 40 MW load at Bus2
    
    # Add storage
    net.add_storage('Storage1', 'Bus1', 50, 100, 0.95, 0.95)  # 50 MW power, 100 MWh capacity
    
    # Add transmission line
    net.add_line('Line1', 'Bus1', 'Bus2', 1.0, 70)  # 70 MW capacity
    
    return net

def main():
    """
    Main function to run the optimization
    """
    # Build the network
    net = build_example_network()
    
    # Print network structure
    print("Network structure created:")
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