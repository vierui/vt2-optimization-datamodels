#!/usr/bin/env python3

"""
network.py

Implementation of a PyPSA-like Network class to simplify power systems modeling.
This class provides a container for power system data and interfaces with the DCOPF solver.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add CPLEX Python API to the Python path (if not already done)
cplex_python_path = "/Applications/CPLEX_Studio2211/python"
if os.path.exists(cplex_python_path) and cplex_python_path not in sys.path:
    sys.path.append(cplex_python_path)

class Network:
    def __init__(self, name="PowerSystem"):
        """
        Initialize a Network object to store power system data.
        
        Args:
            name: Name of the network (optional)
        """
        self.name = name
        
        # Core data containers
        self.buses = pd.DataFrame()
        self.lines = pd.DataFrame()  # Same as branch in original code
        self.generators = pd.DataFrame()
        self.loads = pd.DataFrame()
        
        # Investment parameters
        self.investment_periods = 1  # Number of investment periods
        self.planning_horizon = 10   # Default 10-year planning horizon
        self.asset_lifetimes = {}    # Dict mapping asset IDs to lifetimes in years
        self.asset_capex = {}        # Dict mapping asset IDs to capital costs
        self.existing_investment = {} # Dict mapping asset IDs to boolean (True if already installed)
        
        # Snapshot management
        self.snapshots = pd.Index([])
        self.snapshot_weightings = pd.Series()
        
        # Results storage
        self.results = {}
    
    def set_snapshots(self, snapshots):
        """
        Set the time periods (snapshots) for the network.
        
        Args:
            snapshots: List, pandas DateTimeIndex, or RangeIndex of time steps
        """
        self.snapshots = pd.Index(snapshots)
        # Default weighting is 1.0 for each snapshot
        self.snapshot_weightings = pd.Series(data=1.0, index=self.snapshots)
    
    def set_snapshot_weightings(self, weights):
        """
        Set custom weightings for snapshots.
        
        Args:
            weights: Series or dict with index matching snapshots
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        if not set(weights.index).issubset(set(self.snapshots)):
            raise ValueError("Weightings index doesn't match snapshot index")
        
        # Update weightings, keeping original index order
        for idx in self.snapshots:
            if idx in weights.index:
                self.snapshot_weightings[idx] = weights[idx]
    
    def add_buses(self, bus_data):
        """
        Add or update bus data.
        
        Args:
            bus_data: DataFrame with columns including ['bus_i', 'type', etc.]
        """
        self.buses = bus_data
    
    def add_lines(self, line_data):
        """
        Add or update line data (branches).
        
        Args:
            line_data: DataFrame with columns including ['fbus', 'tbus', 'sus', 'ratea', etc.]
        """
        self.lines = line_data
    
    def add_generators(self, gen_data):
        """
        Add or update generator data.
        
        Args:
            gen_data: DataFrame with columns including ['id', 'bus', 'pmin', 'pmax', 'gencost', etc.]
        """
        self.generators = gen_data
        
        # Extract investment parameters if available
        for _, row in gen_data.iterrows():
            if 'lifetime' in row and 'capex' in row and 'id' in row:
                asset_id = row['id']
                self.asset_lifetimes[asset_id] = row['lifetime']
                self.asset_capex[asset_id] = row['capex']
                
                # Track if asset is already installed
                if 'investment_required' in row:
                    self.existing_investment[asset_id] = (row['investment_required'] == 0)
    
    def add_loads(self, load_data):
        """
        Add or update load (demand) data.
        
        Args:
            load_data: DataFrame with columns including ['time', 'bus', 'pd']
        """
        self.loads = load_data
    
    def get_snapshot_weightings_dict(self):
        """
        Get snapshot weightings as a dictionary.
        
        Returns:
            Dict mapping snapshot to weighting value
        """
        return self.snapshot_weightings.to_dict()
    
    def _build_generator_time_series(self):
        """
        Convert generator data to format expected by dcopf.
        
        Returns:
            DataFrame in the format required by dcopf function
        """
        if self.generators.empty:
            return pd.DataFrame()
        
        # Check if generators already have time series structure
        if 'time' in self.generators.columns:
            # Filter for snapshots we care about
            return self.generators[self.generators['time'].isin(self.snapshots)]
        
        # Otherwise, build time series
        rows = []
        for _, gen in self.generators.iterrows():
            gen_id = gen['id']
            
            # Check for required fields
            required_fields = ['bus', 'pmin', 'pmax', 'gencost']
            for field in required_fields:
                if field not in gen:
                    raise ValueError(f"Generator {gen_id} missing required field {field}")
            
            for snap in self.snapshots:
                # Create row for this generator at this snapshot
                row = {
                    'id': gen_id,
                    'time': snap,
                    'bus': gen['bus'],
                    'pmin': gen['pmin'],
                    'pmax': gen['pmax'],
                    'gencost': gen['gencost']
                }
                
                # Add storage parameters if they exist
                # Note: In original code, emax > 0 indicates storage unit
                if 'emax' in gen and gen['emax'] > 0:
                    row['emax'] = gen['emax']
                    row['einitial'] = gen.get('einitial', 0)
                    row['eta'] = gen.get('eta', 0.9)  # Default efficiency
                else:
                    row['emax'] = 0
                    row['einitial'] = 0
                    row['eta'] = 0
                
                # Add investment parameters
                row['investment_required'] = gen.get('investment_required', 0)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _build_demand_time_series(self):
        """
        Convert load data to format expected by dcopf.
        
        Returns:
            DataFrame in the format required by dcopf function
        """
        # Check if loads already have the required format
        required_cols = ['time', 'bus', 'pd']
        if all(col in self.loads.columns for col in required_cols):
            # Filter for snapshots we care about
            return self.loads[self.loads['time'].isin(self.snapshots)]
        
        # Otherwise, build the format
        rows = []
        for snap in self.snapshots:
            for _, load in self.loads.iterrows():
                if 'time' in load and load['time'] != snap:
                    continue
                
                rows.append({
                    'time': snap,
                    'bus': load['bus'],
                    'pd': load['pd']
                })
        
        return pd.DataFrame(rows)
    
    def solve_dc(self, investment=False, multi_period=False, **kwargs):
        """
        Run the DCOPF solver with or without investment decisions.
        
        Args:
            investment: Boolean, whether to include investment decisions
            multi_period: Boolean, whether to use multi-period planning
            **kwargs: Additional arguments to pass to the solver
            
        Returns:
            Dictionary with optimization results
        """
        # Import solver - only import here to avoid circular imports
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        
        # Dynamically import - this assumes optimization.py is in same directory
        try:
            from optimization import dcopf, investment_dcopf, investment_dcopf_planning
        except ImportError:
            raise ImportError("Cannot import optimization functions. Check that optimization.py is in the scripts directory.")
        
        # Prepare generator and demand data
        gen_time_df = self._build_generator_time_series()
        demand_time_df = self._build_demand_time_series()
        
        # Get snapshot weightings
        snapshot_weightings = self.get_snapshot_weightings_dict()
        
        print(f"[Network.solve_dc] Running {'investment' if investment else 'operation'} model...")
        print(f"[Network.solve_dc] Network has {len(self.buses)} buses, {len(self.lines)} lines")
        print(f"[Network.solve_dc] Using {len(self.snapshots)} snapshots with weightings")
        
        # Call appropriate solver
        if investment:
            if multi_period:
                print(f"[Network.solve_dc] Running multi-period investment with planning horizon {self.planning_horizon} years")
                result = investment_dcopf_planning(
                    gen_time_series=gen_time_df,
                    branch=self.lines,
                    bus=self.buses,
                    demand_time_series=demand_time_df,
                    planning_horizon=self.planning_horizon,
                    asset_lifetimes=self.asset_lifetimes,
                    asset_capex=self.asset_capex,
                    start_year=kwargs.get('start_year', 2023),
                    delta_t=kwargs.get('delta_t', 1),
                    mip_gap=kwargs.get('mip_gap', 0.01),
                    mip_time_limit=kwargs.get('mip_time_limit', 1800)
                )
            else:
                print(f"[Network.solve_dc] Running single-period investment with planning horizon {self.planning_horizon} years")
                result = investment_dcopf(
                    gen_time_series=gen_time_df,
                    branch=self.lines,
                    bus=self.buses,
                    demand_time_series=demand_time_df,
                    planning_horizon=self.planning_horizon,
                    asset_lifetimes=self.asset_lifetimes,
                    asset_capex=self.asset_capex,
                    existing_investment=self.existing_investment,
                    delta_t=kwargs.get('delta_t', 1)
                )
        else:
            print(f"[Network.solve_dc] Running operation-only model")
            result = dcopf(
                gen_time_series=gen_time_df,
                branch=self.lines,
                bus=self.buses,
                demand_time_series=demand_time_df,
                include_investment=False,
                delta_t=kwargs.get('delta_t', 1)
            )
        
        # Store results
        self.results = result
        
        # If we have generation results and weights, calculate weighted totals
        if result and 'generation' in result:
            self._calculate_weighted_results()
        
        return result
    
    def _calculate_weighted_results(self):
        """
        Calculate weighted annual results based on snapshot weightings.
        Adds 'annual_generation' to results.
        """
        if 'generation' not in self.results:
            return
        
        # Calculate weighted generation
        gen_df = self.results['generation']
        annual_gen = {}
        
        # Group by generator ID
        for gen_id, group in gen_df.groupby('id'):
            total_gen = 0
            for _, row in group.iterrows():
                time = row['time']
                weight = self.snapshot_weightings.get(time, 1.0)
                total_gen += row['gen'] * weight
            
            annual_gen[gen_id] = total_gen
        
        self.results['annual_generation'] = annual_gen
        
        # Calculate weighted costs
        if 'cost' in self.results:
            # Operational cost is already weighted in the model
            self.results['weighted_cost'] = self.results['cost']
    
    def summary(self):
        """
        Print a summary of the network.
        """
        print(f"\n===== Network Summary: {self.name} =====")
        print(f"Buses: {len(self.buses)}")
        print(f"Lines: {len(self.lines)}")
        print(f"Generators: {len(self.generators)}")
        print(f"Loads: {len(self.loads)}")
        print(f"Snapshots: {len(self.snapshots)}")
        
        # Show snapshot weightings summary
        if not self.snapshot_weightings.empty:
            unique_weights = self.snapshot_weightings.value_counts()
            print("Snapshot weightings:")
            for weight, count in unique_weights.items():
                print(f"  Weight {weight}: {count} snapshots")
        
        # Show results summary if available
        if self.results:
            print("\n----- Results Summary -----")
            if 'cost' in self.results:
                print(f"Total cost: ${self.results['cost']:,.2f}")
            
            if 'investment_cost' in self.results:
                print(f"Investment cost: ${self.results['investment_cost']:,.2f}")
                print(f"Operational cost: ${self.results['operational_cost']:,.2f}")
            
            if 'investment_decisions' in self.results:
                print("\nInvestment decisions:")
                for asset_id, decision in self.results['investment_decisions'].items():
                    print(f"  Asset {asset_id}: {'Selected' if decision == 1 else 'Not selected'}")
            
            if 'annual_generation' in self.results:
                print("\nAnnual generation (MWh):")
                for asset_id, gen in self.results['annual_generation'].items():
                    if gen > 0:  # Only show positive generation
                        print(f"  Asset {asset_id}: {gen:,.2f} MWh")
    
    @classmethod
    def from_test_system(cls, time_periods=None, data_mapping=None):
        """
        Create a Network from a test system.
        
        Args:
            time_periods: Optional list of time periods
            data_mapping: Dictionary mapping of timestamps to load, solar, wind data
            
        Returns:
            Network instance
        """
        try:
            from scripts.investment import create_test_system
        except ImportError:
            try:
                from investment import create_test_system
            except ImportError:
                raise ImportError("Could not import create_test_system from either scripts.investment or investment")
        
        # Create test system
        gen_time_series, branch, bus, demand_time_series = create_test_system(
            time_periods=time_periods,
            data_mapping=data_mapping
        )
        
        # Create network from test system data
        net = cls(name="TestPowerSystem")
        
        # Add buses
        net.add_buses(bus)
        
        # Add generators
        net.add_generators(gen_time_series)
        
        # Add lines
        net.add_lines(branch)
        
        # Add load
        net.add_loads(demand_time_series)
        
        # Set the snapshots
        net.set_snapshots(time_periods)
        
        # Extract investment parameters
        for asset_id in gen_time_series['id'].unique():
            asset_data = gen_time_series[gen_time_series['id'] == asset_id].iloc[0]
            net.asset_lifetimes[asset_id] = asset_data.get('lifetime', 20)
            net.asset_capex[asset_id] = asset_data.get('capex', 0)
            net.existing_investment[asset_id] = (asset_data.get('investment_required', 0) == 0)
        
        return net 