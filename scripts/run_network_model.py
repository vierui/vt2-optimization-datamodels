#!/usr/bin/env python3

"""
run_network_model.py

Demonstration script for using the PyPSA-like Network class to run
power system analysis with representative weeks and investment decisions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Get the absolute path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Import the Network class
from scripts.network import Network

def create_representative_weeks(year=2023):
    """
    Create 3 representative weeks using real data from CSV files:
    - Winter (Week 2): 13x weighting
    - Summer (Week 31): 13x weighting
    - Spring/Autumn (Week 43): 26x weighting
    
    Args:
        year: Year to create weeks for
        
    Returns:
        Tuple of (all_snapshots, weights_dict)
    """
    print("Loading real data from CSV files...")
    
    # Define paths to CSV files
    load_csv = os.path.join(project_root, 'data', 'processed', 'load-2023.csv')
    solar_csv = os.path.join(project_root, 'data', 'processed', 'solar-2023.csv')
    wind_csv = os.path.join(project_root, 'data', 'processed', 'wind-2023.csv')
    
    # Check if all files exist
    if not os.path.exists(load_csv):
        raise FileNotFoundError(f"Load data file not found: {load_csv}")
    if not os.path.exists(solar_csv):
        raise FileNotFoundError(f"Solar data file not found: {solar_csv}")
    if not os.path.exists(wind_csv):
        raise FileNotFoundError(f"Wind data file not found: {wind_csv}")
    
    # Load the CSV files
    try:
        load_df = pd.read_csv(load_csv, parse_dates=['time'], index_col='time')
        solar_df = pd.read_csv(solar_csv, parse_dates=['time'], index_col='time')
        wind_df = pd.read_csv(wind_csv, parse_dates=['time'], index_col='time')
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")
    
    # Validate the data
    if load_df.empty or solar_df.empty or wind_df.empty:
        raise ValueError("One or more of the loaded dataframes is empty")
    
    # Make sure all dataframes have the same index
    if not load_df.index.equals(solar_df.index) or not load_df.index.equals(wind_df.index):
        raise ValueError("The time indices of the load, solar, and wind data do not match")
    
    print(f"Successfully loaded {len(load_df)} data points from each CSV file")
    
    # Define the date ranges for each representative week
    winter_start = datetime(year, 1, 9)  # Week 2
    winter_end = winter_start + timedelta(days=7) - timedelta(seconds=1)
    
    summer_start = datetime(year, 7, 31)  # Week 31
    summer_end = summer_start + timedelta(days=7) - timedelta(seconds=1)
    
    spring_autumn_start = datetime(year, 10, 23)  # Week 43
    spring_autumn_end = spring_autumn_start + timedelta(days=7) - timedelta(seconds=1)
    
    # Extract the data for each week
    try:
        winter_load = load_df.loc[winter_start:winter_end].copy()
        winter_solar = solar_df.loc[winter_start:winter_end].copy()
        winter_wind = wind_df.loc[winter_start:winter_end].copy()
        
        summer_load = load_df.loc[summer_start:summer_end].copy()
        summer_solar = solar_df.loc[summer_start:summer_end].copy()
        summer_wind = wind_df.loc[summer_start:summer_end].copy()
        
        spring_autumn_load = load_df.loc[spring_autumn_start:spring_autumn_end].copy()
        spring_autumn_solar = solar_df.loc[spring_autumn_start:spring_autumn_end].copy()
        spring_autumn_wind = wind_df.loc[spring_autumn_start:spring_autumn_end].copy()
    except KeyError as e:
        raise ValueError(f"Failed to extract representative weeks: {str(e)}")
    
    # Verify that we have exactly 168 hours (7 days) for each week
    if len(winter_load) != 168:
        raise ValueError(f"Winter load data has {len(winter_load)} rows instead of 168")
    if len(summer_load) != 168:
        raise ValueError(f"Summer load data has {len(summer_load)} rows instead of 168")
    if len(spring_autumn_load) != 168:
        raise ValueError(f"Spring/Autumn load data has {len(spring_autumn_load)} rows instead of 168")
    
    # Define representative weeks' timestamps
    winter_hours = list(winter_load.index)
    summer_hours = list(summer_load.index)
    spring_autumn_hours = list(spring_autumn_load.index)
    
    # Create a mapping of timestamps to load, solar, and wind data for create_test_system
    # This global mapping will be used by create_test_system
    global real_data_mapping
    real_data_mapping = {}
    
    # Add winter data
    for timestamp in winter_hours:
        real_data_mapping[timestamp] = {
            'load': winter_load.loc[timestamp].iloc[0],
            'solar': winter_solar.loc[timestamp].iloc[0],
            'wind': winter_wind.loc[timestamp].iloc[0],
            'season': 'winter'
        }
    
    # Add summer data
    for timestamp in summer_hours:
        real_data_mapping[timestamp] = {
            'load': summer_load.loc[timestamp].iloc[0],
            'solar': summer_solar.loc[timestamp].iloc[0],
            'wind': summer_wind.loc[timestamp].iloc[0],
            'season': 'summer'
        }
    
    # Add spring/autumn data
    for timestamp in spring_autumn_hours:
        real_data_mapping[timestamp] = {
            'load': spring_autumn_load.loc[timestamp].iloc[0],
            'solar': spring_autumn_solar.loc[timestamp].iloc[0],
            'wind': spring_autumn_wind.loc[timestamp].iloc[0],
            'season': 'spring_autumn'
        }
    
    # Print some sample data for verification
    for season, hours in [('Winter', winter_hours[:3]), ('Summer', summer_hours[:3]), ('Spring/Autumn', spring_autumn_hours[:3])]:
        print(f"\nSample data for {season}:")
        for h in hours:
            print(f"  {h}: Load={real_data_mapping[h]['load']:.2f}, Solar={real_data_mapping[h]['solar']:.2f}, Wind={real_data_mapping[h]['wind']:.2f}")
    
    # Combine all hours
    all_snapshots = winter_hours + summer_hours + spring_autumn_hours
    
    # Create weights dictionary
    weights_dict = {}
    for h in winter_hours:
        weights_dict[h] = 13.0  # Winter represents 13 weeks
    for h in summer_hours:
        weights_dict[h] = 13.0  # Summer represents 13 weeks
    for h in spring_autumn_hours:
        weights_dict[h] = 26.0  # Spring/Autumn represents 26 weeks
    
    print(f"Created {len(all_snapshots)} snapshots representing 52 weeks")
    print(f"- Winter week (Week 2): {len(winter_hours)} hours with weight 13")
    print(f"- Summer week (Week 31): {len(summer_hours)} hours with weight 13")
    print(f"- Spring/Autumn week (Week 43): {len(spring_autumn_hours)} hours with weight 26")
    
    return all_snapshots, weights_dict

def create_network_from_test_system(snapshots, weights_dict):
    """
    Create a Network instance using the test system and setup snapshots and weights.
    
    Args:
        snapshots: List of snapshots to use
        weights_dict: Dict mapping snapshots to weights
        
    Returns:
        Network instance
    """
    # Create network from test system
    print("Creating network from test system with real data...")
    global real_data_mapping
    
    if not real_data_mapping:
        raise ValueError("No real data mapping available. Please run create_representative_weeks first.")
    
    # Pass the real_data_mapping to from_test_system
    net = Network.from_test_system(
        time_periods=snapshots,
        data_mapping=real_data_mapping
    )
    
    # Set snapshot weights
    net.set_snapshot_weightings(weights_dict)
    
    return net

def plot_investment_decisions(network, results_dir):
    """
    Plot the investment decisions.
    
    Args:
        network: Network instance with results
        results_dir: Directory to save results
    """
    if 'investment_decisions' not in network.results:
        print("No investment decisions found in results.")
        return
    
    decisions = network.results['investment_decisions']
    
    # Create directory for plots if it doesn't exist
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    # Create bar chart
    assets = list(decisions.keys())
    values = list(decisions.values())
    
    # Color mapping: 1=green (selected), 0=red (not selected)
    colors = ['green' if v == 1 else 'red' for v in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(assets, values, color=colors)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 'Selected' if height == 1 else 'Not Selected',
                 ha='center', va='bottom')
    
    plt.ylim(0, 1.5)  # Set y-axis limit
    plt.xticks(assets, [f"Asset {asset}" for asset in assets])
    plt.ylabel('Decision (1=Selected, 0=Not Selected)')
    plt.title('Investment Decisions by Asset')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'plots', 'investment_decisions.png'))
    plt.close()

def plot_generation(network, results_dir):
    """
    Plot generation by asset for each representative period.
    
    Args:
        network: Network instance with results
        results_dir: Directory to save results
    """
    if 'generation' not in network.results:
        print("No generation data found in results.")
        return
    
    # Create directory for plots if it doesn't exist
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    gen_df = network.results['generation']
    
    # Plot generation for each time period separately
    # Get all unique timestamps
    time_groups = sorted(gen_df['time'].unique())
    
    # Define date ranges for each representative week 
    # Use the same logic as in create_representative_weeks
    year = time_groups[0].year if time_groups else 2023
    
    # Winter week (Week 2)
    winter_start = datetime(year, 1, 9)
    winter_end = winter_start + timedelta(days=7)
    
    # Summer week (Week 31)
    summer_start = datetime(year, 7, 31)
    summer_end = summer_start + timedelta(days=7)
    
    # Spring/Autumn week (Week 43)
    autumn_start = datetime(year, 10, 23)
    autumn_end = autumn_start + timedelta(days=7)
    
    # Identify timestamps for each week based on date ranges, not just month
    winter_times = [t for t in time_groups if winter_start <= t < winter_end]
    summer_times = [t for t in time_groups if summer_start <= t < summer_end]
    autumn_times = [t for t in time_groups if autumn_start <= t < autumn_end]
    
    print(f"Found {len(winter_times)} timestamps for Winter week")
    print(f"Found {len(summer_times)} timestamps for Summer week")
    print(f"Found {len(autumn_times)} timestamps for Spring/Autumn week")
    
    # Plot for each representative week
    for period_name, time_filter in [
        ('Winter', winter_times),
        ('Summer', summer_times),
        ('Spring_Autumn', autumn_times)
    ]:
        if not time_filter:
            print(f"Warning: No timestamps found for {period_name} week")
            continue
            
        # Filter data for this period
        period_df = gen_df[gen_df['time'].isin(time_filter)]
        
        # Sort by time
        period_df = period_df.sort_values('time')
        
        # Pivot for easier plotting
        pivot_df = period_df.pivot(index='time', columns='id', values='gen')
        
        # Identify storage units (both positive and negative values)
        storage_ids = []
        for col in pivot_df.columns:
            if (pivot_df[col] < 0).any():
                storage_ids.append(col)
        
        # Create plot
        plt.figure(figsize=(16, 8))
        
        # 1. Plot positive generation (including storage discharge)
        gen_plot_df = pivot_df.copy()
        
        # Remove negative values for storage units
        for sid in storage_ids:
            gen_plot_df[sid] = gen_plot_df[sid].clip(lower=0)
        
        # Plot stacked area with customized colors and labels
        colors = ['darkgreen', '#F9CB9C', '#6FA8DC', '#8E7CC3', '#F6B26B', '#76A5AF', '#CC0000']
        labels = ['Nuclear', 'Solar 1', 'Wind 1', 'Wind 2', 'Solar 2', 'Storage 1', 'Storage 2']
        
        ax = gen_plot_df.plot(
            kind='area', 
            stacked=True, 
            alpha=0.7, 
            color=colors[:len(gen_plot_df.columns)],
            figsize=(16, 8)
        )
        
        # Manually set legend labels
        handles, _ = ax.get_legend_handles_labels()
        asset_labels = [f"{labels[i-1]} (ID: {i})" for i in gen_plot_df.columns]
        ax.legend(handles, asset_labels, loc='upper left', fontsize=10)
        
        # 2. Plot storage charging as negative areas (if any storage units)
        if storage_ids:
            charge_df = pd.DataFrame(index=pivot_df.index)
            
            for sid in storage_ids:
                charge_df[f"Storage {sid}"] = -pivot_df[sid].clip(upper=0)
            
            if not charge_df.empty and (charge_df > 0).any().any():
                charge_colors = ['#76A5AF', '#CC0000']  # Same colors as storage units but with transparency
                charge_df.plot(kind='area', stacked=True, ax=ax, alpha=0.4, colormap='Blues')
        
        # Add annotations for the period
        title_parts = []
        
        # Add period name
        title_parts.append(f"Generation by Asset - {period_name} Week")
        
        # Add date range
        if period_name == "Winter":
            title_parts.append(f"January {winter_start.day}-{winter_end.day-1}")
        elif period_name == "Summer":
            title_parts.append(f"July {summer_start.day}-August {summer_end.day-1}")
        elif period_name == "Spring_Autumn":
            title_parts.append(f"October {autumn_start.day}-{autumn_end.day-1}")
        
        plt.title(" | ".join(title_parts), fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Improve x-axis formatting
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # Fewer date labels
        plt.gcf().autofmt_xdate()  # Auto-format date labels
        
        # Annotate day/night cycles for better visualization
        if len(time_filter) >= 24:
            min_time = min(time_filter)
            for day in range(7):
                day_start = min_time + timedelta(days=day)
                night_start = day_start + timedelta(hours=18)  # Assume 6pm is start of night
                plt.axvline(x=day_start, color='black', linestyle='--', alpha=0.2)
                plt.text(day_start, plt.ylim()[1]*0.95, f"Day {day+1}", horizontalalignment='left', verticalalignment='top', fontsize=8)
        
        # Save the plot with high resolution
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', f'generation_{period_name}.png'), dpi=300)
        plt.close()
    
    # Plot annual generation summary (if available)
    if 'annual_generation' in network.results:
        annual_gen = network.results['annual_generation']
        
        # Separate positive and negative generation
        pos_gen = {k: v for k, v in annual_gen.items() if v > 0}
        neg_gen = {k: -v for k, v in annual_gen.items() if v < 0}
        
        # Plot positive generation (including discharge)
        if pos_gen:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(pos_gen.keys(), pos_gen.values(), color=colors[:len(pos_gen)], alpha=0.7)
            plt.title('Annual Generation by Asset')
            plt.xticks([k for k in pos_gen.keys()], [f"{labels[i-1]} (ID: {i})" for i in pos_gen.keys()])
            plt.ylabel('Generation (MWh)')
            plt.grid(True, alpha=0.3)
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(pos_gen.values()),
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', 'annual_generation.png'), dpi=300)
            plt.close()
        
        # Plot negative generation (charging)
        if neg_gen:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(neg_gen.keys(), neg_gen.values(), color=['#76A5AF', '#CC0000'][:len(neg_gen)], alpha=0.7)
            plt.title('Annual Storage Charging by Asset')
            plt.xticks([k for k in neg_gen.keys()], [f"Storage {k}" for k in neg_gen.keys()])
            plt.ylabel('Charging (MWh)')
            plt.grid(True, alpha=0.3)
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(neg_gen.values()),
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', 'annual_charging.png'), dpi=300)
            plt.close()
            
    # Create a comparative plot showing all three periods
    try:
        # Prepare data for each period
        period_data = {}
        for period_name, time_filter in [
            ('Winter', winter_times),
            ('Summer', summer_times),
            ('Spring/Autumn', autumn_times)
        ]:
            if not time_filter:
                continue
                
            # Filter data for this period
            period_df = gen_df[gen_df['time'].isin(time_filter)]
            
            # Calculate average generation by asset for this period
            avg_gen = period_df.groupby('id')['gen'].mean()
            period_data[period_name] = avg_gen
        
        if len(period_data) > 1:
            # Create a DataFrame with all periods
            comp_df = pd.DataFrame(period_data)
            
            # Plot comparative bar chart
            plt.figure(figsize=(12, 7))
            comp_df.plot(kind='bar', figsize=(12, 7))
            plt.title('Average Generation by Asset - Seasonal Comparison')
            plt.xlabel('Asset ID')
            plt.ylabel('Average Generation (MW)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Season')
            
            # Save the comparative plot
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', 'generation_seasonal_comparison.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Could not create comparative seasonal plot: {str(e)}")

def plot_marginal_prices(network, results_dir):
    """
    Plot marginal prices.
    
    Args:
        network: Network instance with results
        results_dir: Directory to save results
    """
    if 'marginal_prices' not in network.results:
        print("No marginal price data found in results.")
        return
    
    # Create directory for plots if it doesn't exist
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    price_df = network.results['marginal_prices']
    
    # Check if required columns exist, otherwise add them
    if price_df.empty:
        print("Marginal price dataframe is empty, skipping plots.")
        return
    
    if 'bus' not in price_df.columns:
        # Try to find alternative column names for bus
        if 'bus_id' in price_df.columns:
            price_df['bus'] = price_df['bus_id']
        else:
            print("No 'bus' column found in marginal prices. Cannot create price plots.")
            return
    
    if 'time' not in price_df.columns:
        # Try to find alternative column names for time
        if 'snapshot' in price_df.columns:
            price_df['time'] = price_df['snapshot']
        else:
            print("No 'time' column found in marginal prices. Cannot create price plots.")
            return
    
    if 'price' not in price_df.columns and 'value' in price_df.columns:
        price_df['price'] = price_df['value']
    
    # Plot prices for each bus separately
    for bus in price_df['bus'].unique():
        bus_prices = price_df[price_df['bus'] == bus]
        
        plt.figure(figsize=(14, 7))
        plt.plot(bus_prices['time'], bus_prices['price'], marker='o', linestyle='-', alpha=0.7)
        plt.title(f'Marginal Prices at Bus {bus}')
        plt.xlabel('Time')
        plt.ylabel('Price ($/MWh)')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        plt.savefig(os.path.join(results_dir, 'plots', f'prices_bus{bus}.png'))
        plt.close()
    
    # Create a summary plot with all buses
    plt.figure(figsize=(14, 7))
    
    # Plot each bus with different color/style
    for bus in sorted(price_df['bus'].unique()):
        bus_prices = price_df[price_df['bus'] == bus]
        plt.plot(bus_prices['time'], bus_prices['price'], label=f'Bus {bus}', alpha=0.7)
    
    plt.title('Marginal Prices by Bus')
    plt.xlabel('Time')
    plt.ylabel('Price ($/MWh)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis for datetime
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'plots', 'prices_all_buses.png'))
    plt.close()

def save_summary_to_markdown(network, results_dir):
    """
    Save a summary of the results to a markdown file.
    
    Args:
        network: Network instance with results
        results_dir: Directory to save results
    """
    # Create summary file
    summary_file = os.path.join(results_dir, 'summary.md')
    
    with open(summary_file, 'w') as f:
        f.write(f"# Power System Analysis Results - {network.name}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Network summary
        f.write("## Network Summary\n\n")
        f.write(f"- Buses: {len(network.buses)}\n")
        f.write(f"- Lines: {len(network.lines)}\n")
        f.write(f"- Generators: {len(network.generators['id'].unique())}\n")
        f.write(f"- Snapshots: {len(network.snapshots)}\n\n")
        
        # Snapshot weightings
        f.write("### Snapshot Weightings\n\n")
        unique_weights = network.snapshot_weightings.value_counts()
        for weight, count in unique_weights.items():
            f.write(f"- Weight {weight}: {count} snapshots\n")
        f.write("\n")
        
        # Cost summary
        if 'cost' in network.results:
            f.write("## Cost Summary\n\n")
            f.write(f"- Total cost: ${network.results['cost']:,.2f}\n")
            
            if 'investment_cost' in network.results:
                f.write(f"- Investment cost: ${network.results['investment_cost']:,.2f}\n")
                f.write(f"- Operational cost: ${network.results['operational_cost']:,.2f}\n")
            f.write("\n")
        
        # Investment decisions
        if 'investment_decisions' in network.results:
            f.write("## Investment Decisions\n\n")
            f.write("| Asset ID | Decision | Annual Cost |\n")
            f.write("|----------|----------|-------------|\n")
            
            for asset_id, decision in network.results['investment_decisions'].items():
                decision_text = "Selected" if decision == 1 else "Not selected"
                capex = network.asset_capex.get(asset_id, 0)
                lifetime = network.asset_lifetimes.get(asset_id, 1)
                annual_cost = capex / lifetime if decision == 1 else 0
                
                f.write(f"| {asset_id} | {decision_text} | ${annual_cost:,.2f} |\n")
            f.write("\n")
        
        # Generation summary
        if 'annual_generation' in network.results:
            f.write("## Annual Generation Summary\n\n")
            f.write("| Asset ID | Annual Generation (MWh) |\n")
            f.write("|----------|-------------------------|\n")
            
            for asset_id, gen in network.results['annual_generation'].items():
                if gen > 0:  # Only show positive generation
                    f.write(f"| {asset_id} | {gen:,.2f} |\n")
            f.write("\n")
            
            # Check for negative generation (charging)
            charging = {k: -v for k, v in network.results['annual_generation'].items() if v < 0}
            if charging:
                f.write("## Annual Storage Charging\n\n")
                f.write("| Storage ID | Annual Charging (MWh) |\n")
                f.write("|------------|------------------------|\n")
                
                for asset_id, charge in charging.items():
                    f.write(f"| {asset_id} | {charge:,.2f} |\n")
                f.write("\n")
        
        # Include links to images
        f.write("## Results Visualizations\n\n")
        
        # Check if investment decisions plot exists
        investment_plot_path = os.path.join(results_dir, 'plots', 'investment_decisions.png')
        if os.path.exists(investment_plot_path):
            f.write("### Investment Decisions\n\n")
            f.write("![Investment Decisions](plots/investment_decisions.png)\n\n")
        
        # Check if generation plots exist
        winter_plot_path = os.path.join(results_dir, 'plots', 'generation_Winter.png')
        summer_plot_path = os.path.join(results_dir, 'plots', 'generation_Summer.png')
        spring_autumn_plot_path = os.path.join(results_dir, 'plots', 'generation_Spring_Autumn.png')
        
        if any([os.path.exists(winter_plot_path), os.path.exists(summer_plot_path), os.path.exists(spring_autumn_plot_path)]):
            f.write("### Generation Profiles\n\n")
            
            if os.path.exists(winter_plot_path):
                f.write("![Winter Generation](plots/generation_Winter.png)\n\n")
            
            if os.path.exists(summer_plot_path):
                f.write("![Summer Generation](plots/generation_Summer.png)\n\n")
            
            if os.path.exists(spring_autumn_plot_path):
                f.write("![Spring/Autumn Generation](plots/generation_Spring_Autumn.png)\n\n")
        
        # Check if price plot exists
        price_plot_path = os.path.join(results_dir, 'plots', 'prices_all_buses.png')
        if os.path.exists(price_plot_path):
            f.write("### Price Profiles\n\n")
            f.write("![Marginal Prices](plots/prices_all_buses.png)\n\n")
    
    print(f"Summary saved to {summary_file}")

def run_investment_model():
    """
    Run the investment model using the Network class and representative weeks.
    """
    print("Starting investment model with representative weeks approach...")
    
    # Create results directory
    results_dir = os.path.join(project_root, 'results', 'network_model')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create representative weeks using real data
    snapshots, weights_dict = create_representative_weeks()
    
    # Create network
    net = create_network_from_test_system(snapshots, weights_dict)
    
    # Print network summary
    net.summary()
    
    # Run investment optimization
    print("\nRunning investment optimization...")
    results = net.solve_dc(investment=True, delta_t=1)
    
    # Check if we have valid results
    if not results:
        print("Error: No valid results returned from solver")
        return
    
    # Print basic results
    print("\n===== Investment Results =====")
    if 'investment_decisions' in results:
        print("Investment decisions:")
        for asset_id, decision in results['investment_decisions'].items():
            print(f"  Asset {asset_id}: {'Selected' if decision == 1 else 'Not selected'}")
    
    if 'cost' in results:
        print(f"\nTotal cost: ${results['cost']:,.2f}")
        if 'investment_cost' in results:
            print(f"Investment cost: ${results['investment_cost']:,.2f}")
            print(f"Operational cost: ${results['operational_cost']:,.2f}")
    
    # Create plots
    print("\nCreating visualization plots...")
    plot_investment_decisions(net, results_dir)
    plot_generation(net, results_dir)
    plot_marginal_prices(net, results_dir)
    
    # Save summary
    print("\nSaving summary report...")
    save_summary_to_markdown(net, results_dir)
    
    print(f"\nResults saved to: {results_dir}")
    return net, results_dir

if __name__ == "__main__":
    run_investment_model() 