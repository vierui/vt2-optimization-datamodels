"""
Functions for creating summary and comparison plots.
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def create_annual_summary_plots(scenario_data: dict, results_root: str) -> None:
    """Create annual generation and cost mix plots"""
    scenario_name = scenario_data['scenario_name']
    
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
            wedges, texts, autotexts = ax1.pie(
                list(generation_data.values()),
                labels=list(generation_data.keys()),
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*total_gen):,} MWh)',
                textprops={'fontsize': 8}
            )
            plt.setp(autotexts, size=7, weight="bold")
            plt.setp(texts, size=8)
        else:
            ax1.text(0.5, 0.5, 'No valid generation data', 
                    ha='center', va='center')
    ax1.set_title('Annual Generation Mix')

    # Plot 2: Winter vs Summer Generation Comparison (Tornado Chart)
    winter_gen = scenario_data.get('winter_gen', {})
    summer_gen = scenario_data.get('summer_gen', {})
    
    print("\nDebug - Seasonal Generation Data:")
    print(f"Winter generation: {winter_gen}")
    print(f"Summer generation: {summer_gen}")
    
    all_assets = sorted(set(winter_gen.keys()) | set(summer_gen.keys()))
    print(f"All assets found: {all_assets}")
    
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
    else:
        ax2.text(0.5, 0.5, 'No seasonal generation data', 
                ha='center', va='center')

    # Plot 3: NPV Comparison
    npv_data = {
        '10y': scenario_data.get('npv_10y', 0),
        '20y': scenario_data.get('npv_20y', 0),
        '30y': scenario_data.get('npv_30y', 0)
    }
    
    npv_data = {k: v for k, v in npv_data.items() if not pd.isna(v)}
    
    if npv_data:
        bars = ax3.bar(npv_data.keys(), npv_data.values(), 
                      color=['red' if v < 0 else 'lightgreen' for v in npv_data.values()])
        ax3.set_title('Net Present Value (NPV) Comparison')
        ax3.set_ylabel('NPV (€)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax3.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'€{height:,.0f}',
                ha='center',
                va=va,
                rotation=0,
                fontsize=8
            )
    else:
        ax3.text(0.5, 0.5, 'No valid NPV data', 
                ha='center', va='center')

    plt.suptitle(f'Annual Summary - {scenario_name}', fontsize=16, y=1.05)
    plt.tight_layout()
    
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
    
    # [Rest of the implementation from multi_scenario.py]
    # Would you like me to include the full implementation of this function as well? 