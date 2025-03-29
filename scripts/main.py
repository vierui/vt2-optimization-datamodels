#!/usr/bin/env python3
"""
Main script for running the power grid optimization using the Network class
with numeric component IDs and standardized data structure
"""
from network import Network
import os
import argparse

def build_example_network_programmatically(use_day_profiles=False, day=10):
    """
    Build an example two-bus network with numeric IDs for all components
    
    Args:
        use_day_profiles: Whether to use time-dependent profiles
        day: Day of the year to use for profiles
    """
    # Create an empty network with or without day profiles
    net = Network(use_day_profiles=use_day_profiles, day=day)
    
    # Set time horizon (24 hours)
    if not use_day_profiles:
        net.set_snapshots(24)
    
    # Add buses (numeric IDs)
    net.add_bus(1, "Bus 1")
    net.add_bus(2, "Bus 2")
    
    # Add generators (IDs in 1000s range)
    # Add thermal, wind, and solar generators to test different profiles
    net.add_generator(1001, "Thermal Generator", 1, 100, 50, gen_type='thermal')  # 100 MW capacity, 50 €/MWh
    net.add_generator(1002, "Wind Generator", 2, 80, 30, gen_type='wind')  # 80 MW capacity, 30 €/MWh
    net.add_generator(1003, "Solar Generator", 1, 70, 40, gen_type='solar')  # 70 MW capacity, 40 €/MWh
    
    # Add loads (IDs in 2000s range)
    net.add_load(2001, "Load 1", 1, 30)  # 30 MW load at Bus 1
    net.add_load(2002, "Load 2", 2, 40)  # 40 MW load at Bus 2
    
    # Add storage (IDs in 3000s range)
    net.add_storage(3001, "Storage 1", 1, 50, 100, 0.95, 0.95)  # 50 MW power, 100 MWh capacity
    
    # Add transmission line (IDs in 4000s range)
    net.add_line(4001, "Line 1-2", 1, 2, 1.0, 70)  # 70 MW capacity
    
    return net

def build_example_network_from_csv(use_day_profiles=False, day=10):
    """
    Build an example network by loading data from CSV files
    
    Args:
        use_day_profiles: Whether to use time-dependent profiles
        day: Day of the year to use for profiles
    """
    # Get script directory and go up to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create a network with data from CSV files
    data_path = os.path.join(project_root, "data", "grid")
    print(f"Looking for data in: {data_path}")
    net = Network(data_path, use_day_profiles=use_day_profiles, day=day)
    
    return net

def main():
    """
    Main function to run the optimization
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DC Optimal Power Flow with time-dependent profiles")
    parser.add_argument("--use-day-profiles", action="store_true", help="Use time-dependent profiles")
    parser.add_argument("--day", type=int, default=10, help="Day of the year to use (1-365)")
    args = parser.parse_args()
    
    # Build the network (either programmatically or from CSV)
    print("Building network...")
    
    # Try loading from CSV first
    try:
        net = build_example_network_from_csv(use_day_profiles=args.use_day_profiles, day=args.day)
        data_source = "CSV files"
    except Exception as e:
        print(f"Could not load from CSV: {e}")
        print("Building network programmatically instead...")
        net = build_example_network_programmatically(use_day_profiles=args.use_day_profiles, day=args.day)
        data_source = "programmatic definition"
    
    # Print network structure
    print(f"Network structure created from {data_source}:")
    print(f"- Buses: {net.buses.index.tolist()}")
    print(f"- Generators: {net.generators.index.tolist()}")
    print(f"- Loads: {net.loads.index.tolist()}")
    print(f"- Storage units: {net.storage_units.index.tolist()}")
    print(f"- Lines: {net.lines.index.tolist()}")
    
    if args.use_day_profiles:
        print(f"\nUsing time-dependent profiles for day {args.day}")
    
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