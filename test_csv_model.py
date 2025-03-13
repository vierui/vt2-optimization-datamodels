#!/usr/bin/env python3

"""
Test script to verify that the DCOPF model runs correctly with CSV-based data.
"""

from scripts.marginal_price_analysis import create_simple_power_system, run_dcopf_analysis

def test_csv_data():
    """Test that the model correctly reads bus and branch data from CSV files."""
    print("Testing CSV-based power system model...")
    
    # Test data reading
    gen_time_series, branch, bus, demand_time_series = create_simple_power_system()
    
    # Verify bus data
    print("\nBus Data Preview:")
    print(f"Number of buses: {len(bus)}")
    print(bus.head())
    
    # Verify branch data
    print("\nBranch Data Preview:")
    print(f"Number of branches: {len(branch)}")
    print(branch.head())
    
    # Verify generation data
    print("\nGeneration Data Preview:")
    print(f"Number of generators: {len(gen_time_series['id'].unique())}")
    print(gen_time_series.head())
    
    # Verify demand data
    print("\nDemand Data Preview:")
    print(f"Number of load points: {len(demand_time_series)}")
    print(demand_time_series)
    
    # Verify DCOPF solution
    print("\nRunning DCOPF to verify model...")
    try:
        results = run_dcopf_analysis()
        if results:
            print("DCOPF solved successfully!")
            print(f"Total system cost: ${results['cost']:.2f}")
            
            # Print marginal prices
            prices = results.get('marginal_prices')
            if prices is not None:
                print("\nMarginal Prices ($/MWh):")
                for _, row in prices.iterrows():
                    print(f"Bus {int(row['bus'])}: ${row['price']:.2f}/MWh")
            else:
                print("No marginal prices found in results.")
        else:
            print("DCOPF failed to solve.")
    except Exception as e:
        print(f"Error running DCOPF: {e}")

if __name__ == "__main__":
    test_csv_data() 