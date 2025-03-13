"""
Functions for creating summary and comparison plots.
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

# Set the style for all plots
plt.style.use('seaborn-v0_8-darkgrid')  # Using a valid matplotlib style

# Define a consistent color palette for assets using a softer palette
ASSET_COLORS = {
    'nuclear': '#FF9D5C',  # Orange
    'gas': '#4C9CFF',      # Blue
    'solar': '#4DB870',    # Green
    'wind': '#B07CFF',     # Purple
    'battery1': '#8B4513',  # Brown for storage
    'battery2': '#8B4513'   # Brown for storage
}

def create_annual_summary_plots(scenario_data: dict, results_root: str) -> None:
    """Create annual generation and cost mix plots with sensitivity analysis"""
    scenario_name = scenario_data['base_scenario']
    variant = scenario_data.get('variant', 'nominal')
    
    # Only create plots for nominal variant
    if variant != 'nominal':
        return
        
    # Create figure directory
    figure_dir = os.path.join(results_root, scenario_name, "figure")
    os.makedirs(figure_dir, exist_ok=True)

    # Create a figure with 3 subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Annual Generation (Pie Chart)
    generation_data = {k.replace('gen_', ''): v for k, v in scenario_data.items() 
                      if k.startswith('gen_') and not k.startswith('gen_cost_')}
    if generation_data:
        generation_data = {k: abs(v) for k, v in generation_data.items() 
                         if v and not pd.isna(v)}
        if generation_data:
            total_gen = sum(generation_data.values())
            colors = [ASSET_COLORS.get(asset, sns.color_palette("pastel")[7]) 
                     for asset in generation_data.keys()]
            wedges, texts, autotexts = ax1.pie(
                list(generation_data.values()),
                labels=list(generation_data.keys()),
                colors=colors,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*total_gen):,} MWh)',
                textprops={'fontsize': 8}
            )
            plt.setp(autotexts, size=7, weight="bold")
            plt.setp(texts, size=8)
        else:
            ax1.text(0.5, 0.5, 'No valid generation data', 
                    ha='center', va='center')
    ax1.set_title('Annual Generation Mix')

    # Plot 2: Winter vs Summer Generation (Tornado Chart)
    winter_gen = {}
    summer_gen = {}
    
    # Get all generation and storage keys
    gen_keys = [k for k in scenario_data.keys() 
                if (k.startswith('winter_gen_') or k.startswith('summer_gen_'))]
    
    # Process winter and summer generation
    for key in gen_keys:
        value = scenario_data.get(key, 0)
        if pd.isna(value):
            value = 0
        
        # Extract asset type and handle both generators and storage
        if key.startswith('winter_gen_'):
            asset_type = key.replace('winter_gen_', '')
            winter_gen[asset_type] = float(value)
        elif key.startswith('summer_gen_'):
            asset_type = key.replace('summer_gen_', '')
            summer_gen[asset_type] = float(value)
    
    # Ensure all assets appear in both seasons
    all_assets = sorted(set(winter_gen.keys()) | set(summer_gen.keys()))
    for asset in all_assets:
        winter_gen.setdefault(asset, 0)
        summer_gen.setdefault(asset, 0)
    
    if all_assets:
        # Create data for tornado chart
        winter_values = [-winter_gen.get(asset, 0) for asset in all_assets]
        summer_values = [summer_gen.get(asset, 0) for asset in all_assets]
        
        # Set white background with grey grid
        ax2.set_facecolor('white')
        ax2.grid(True, color='grey', alpha=0.3)
        
        # Find max absolute value for symmetric axis
        max_val = max(abs(min(winter_values)), abs(max(summer_values)))
        
        # Create horizontal bars
        y_pos = np.arange(len(all_assets))
        
        # Plot winter values (left side)
        winter_bars = ax2.barh(y_pos, winter_values, 
                             align='center',
                             color=[sns.set_hls_values(ASSET_COLORS.get(asset, 'gray'), l=0.8) 
                                   for asset in all_assets],
                             label='Winter',
                             alpha=0.8)
        
        # Plot summer values (right side)
        summer_bars = ax2.barh(y_pos, summer_values, 
                             align='center',
                             color=[ASSET_COLORS.get(asset, 'gray') for asset in all_assets],
                             label='Summer',
                             alpha=0.8)
        
        # Customize plot
        ax2.set_xlabel('Generation (MWh/week)')
        ax2.set_title('Winter vs Summer Weekly Generation')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_assets)
        ax2.legend(loc='upper right')
        
        # Set symmetric x-axis
        ax2.set_xlim(-max_val*1.1, max_val*1.1)
        
        # Add zero line
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, v in enumerate(winter_values):
            if abs(v) > 1e-6:
                ax2.text(-v/2, i, f'{abs(v):,.0f}', 
                        ha='center', va='center', fontsize=8)
        for i, v in enumerate(summer_values):
            if abs(v) > 1e-6:
                ax2.text(v/2, i, f'{v:,.0f}', 
                        ha='center', va='center', fontsize=8)
        
        # Add black frame
        for spine in ax2.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)
    else:
        ax2.text(0.5, 0.5, 'No generation data available', 
                ha='center', va='center',
                transform=ax2.transAxes)

    # Plot 3: NPV with Sensitivity Range
    # Set white background with grey grid for NPV plot
    ax3.set_facecolor('white')
    ax3.grid(True, color='grey', alpha=0.3)
    
    # Get nominal NPV values
    npv_data = {
        '10y': scenario_data.get('npv_10y', 0) / 1e6,
        '20y': scenario_data.get('npv_20y', 0) / 1e6,
        '30y': scenario_data.get('npv_30y', 0) / 1e6
    }
    
    # Calculate high and low variants based on load factor sensitivity
    nominal_load = 1.0
    high_load = 1.2
    low_load = 0.8
    
    # Prepare data for plotting
    years = [10, 20, 30]  # x-axis values
    nominal_values = list(npv_data.values())
    
    # Calculate high and low values based on load factor scaling
    high_values = [v * (high_load/nominal_load) for v in nominal_values]
    low_values = [v * (low_load/nominal_load) for v in nominal_values]
    
    # Debug prints
    print(f"\nDebug NPV values for {scenario_name}:")
    print("Nominal (load=1.0):", nominal_values)
    print("High (load=1.2):", high_values)
    print("Low (load=0.8):", low_values)
    
    # Plot shaded area for uncertainty
    ax3.fill_between(years, low_values, high_values, 
                    alpha=0.3, color='grey', 
                    label='Load Sensitivity Range')
    
    # Plot nominal line
    line = ax3.plot(years, nominal_values, 'o-', 
                    color='darkblue', linewidth=2, 
                    label='Nominal NPV',
                    zorder=2)
    
    # Add value labels for nominal points
    for x, y in zip(years, nominal_values):
        ax3.text(x, y, f'{y:,.1f}M', 
                ha='center', va='bottom' if y >= 0 else 'top',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Customize plot
    ax3.set_title('Net Present Value (NPV) with Load Sensitivity')
    ax3.set_xlabel('Years')
    ax3.set_ylabel('NPV (Million CHF)')
    
    # Set x-axis ticks
    ax3.set_xticks(years)
    ax3.set_xticklabels([f'{y}y' for y in years])
    
    # Add zero line
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend
    ax3.legend(loc='lower left')
    
    # Add black frame to NPV plot
    for spine in ax3.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.suptitle(f'Annual Summary - {scenario_name}', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(figure_dir, "annual_summary.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved annual summary plots => {plot_path}")

def create_scenario_comparison_plot(scenario_data: dict, results_root: str) -> None:
    """Create a three-panel comparison plot for a scenario"""
    scenario_name = scenario_data.get('scenario_name', 'Unknown')
    scenario_folder = os.path.join(results_root, scenario_name, "figure")
    os.makedirs(scenario_folder, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Generation Mix Comparison
    winter_gen = scenario_data.get('winter_gen', {})
    summer_gen = scenario_data.get('summer_gen', {})
    autumn_spring_gen = scenario_data.get('autumn_spring_gen', {})
    
    # Get all unique asset types
    all_assets = sorted(set(winter_gen.keys()) | 
                       set(summer_gen.keys()) | 
                       set(autumn_spring_gen.keys()))
    
    if all_assets:
        # Create data for comparison
        seasons = ['Winter', 'Summer', 'Autumn/Spring']
        data = {
            'Winter': winter_gen,
            'Summer': summer_gen,
            'Autumn/Spring': autumn_spring_gen
        }
        
        x = np.arange(len(seasons))
        width = 0.8 / len(all_assets)
        
        for i, asset in enumerate(all_assets):
            values = [data[season].get(asset, 0) for season in seasons]
            position = x + (i - len(all_assets)/2 + 0.5) * width
            ax1.bar(position, values, width, label=asset)
        
        ax1.set_ylabel('Generation (MWh)')
        ax1.set_title('Seasonal Generation Mix')
        ax1.set_xticks(x)
        ax1.set_xticklabels(seasons)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Winter vs Summer Generation (Tornado Chart)
    if all_assets:
        # Create data for tornado chart
        winter_values = [-winter_gen.get(asset, 0) for asset in all_assets]  # Negative for left side
        summer_values = [summer_gen.get(asset, 0) for asset in all_assets]   # Positive for right side
        
        # Create horizontal bars
        y_pos = np.arange(len(all_assets))
        
        # Plot winter values (left side)
        ax2.barh(y_pos, winter_values, 
                align='center', 
                color='lightblue',
                label='Winter', 
                alpha=0.8)
        
        # Plot summer values (right side)
        ax2.barh(y_pos, summer_values, 
                align='center', 
                color='orange',
                label='Summer', 
                alpha=0.8)
        
        # Customize plot
        ax2.set_xlabel('Generation (MWh/week)')
        ax2.set_title('Winter vs Summer Weekly Generation')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_assets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add zero line
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, v in enumerate(winter_values):
            if v != 0:
                ax2.text(v, i, f'{abs(v):,.0f}', 
                        ha='right', va='center', fontsize=8)
        for i, v in enumerate(summer_values):
            if v != 0:
                ax2.text(v, i, f'{v:,.0f}', 
                        ha='left', va='center', fontsize=8)

    # Plot 3: Capacity Factors
    capacity_factors = {k.replace('capacity_factor_', ''): v 
                       for k, v in scenario_data.items() 
                       if k.startswith('capacity_factor_')}
    
    if capacity_factors:
        assets = list(capacity_factors.keys())
        values = [capacity_factors[asset] * 100 for asset in assets]  # Convert to percentage
        
        bars = ax3.bar(assets, values)
        ax3.set_ylabel('Capacity Factor (%)')
        ax3.set_title('Asset Capacity Factors')
        ax3.set_xticklabels(assets, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

    plt.suptitle(f'Scenario Comparison - {scenario_name}', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(scenario_folder, "scenario_comparison.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved scenario comparison plot => {plot_path}") 