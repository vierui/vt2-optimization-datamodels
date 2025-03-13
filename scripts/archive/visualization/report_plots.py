import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Dict

def create_scenario_plots(all_scenarios_data: Dict[str, pd.DataFrame]):
    """
    Create plots for all scenarios' storage data in one image:
    - Left subplot: Summer Storage SoC for all scenarios
    - Right subplot: Winter Storage SoC for all scenarios
    """
    # Get project root directory (where data/ is located)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Colors for different storage units
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Up to 10 different colors
    
    for scenario_name, scenario_data in all_scenarios_data.items():
        # Ensure scenario directory exists with correct path
        scenario_dir = os.path.join(project_root, 'data', 'results', f'{scenario_name}', 'figure')
        os.makedirs(scenario_dir, exist_ok=True)
        
        if isinstance(scenario_data.index, pd.Index):
            scenario_data.index = pd.to_datetime(scenario_data.index)
        
        # Group data by storage_id if it exists
        if 'storage_id' in scenario_data.columns:
            storage_units = scenario_data['storage_id'].unique()
        else:
            storage_units = [1]  # Default if no storage_id column
            
        # Define summer and winter periods
        summer_period = scenario_data[scenario_data.index.month.isin([6, 7, 8])]
        winter_period = scenario_data[scenario_data.index.month.isin([12, 1, 2])]
        
        if 'Storage_SoC' in scenario_data.columns:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.patch.set_facecolor('white')
            
            # Get max value for common y-axis scale
            y_max = scenario_data['Storage_SoC'].max()
            
            # Plot each storage unit
            for idx, storage_id in enumerate(storage_units):
                if 'storage_id' in scenario_data.columns:
                    storage_data = scenario_data[scenario_data['storage_id'] == storage_id]
                    summer_storage = summer_period[summer_period['storage_id'] == storage_id]
                    winter_storage = winter_period[winter_period['storage_id'] == storage_id]
                else:
                    storage_data = scenario_data
                    summer_storage = summer_period
                    winter_storage = winter_period
                
                label = f'Storage {storage_id}'
                color = colors[idx]
                
                # Summer plot
                ax1.plot(summer_storage.index, summer_storage['Storage_SoC'], 
                        label=label, color=color, linestyle='-', marker='o', markersize=4)
                
                # Winter plot
                ax2.plot(winter_storage.index, winter_storage['Storage_SoC'], 
                        label=label, color=color, linestyle='-', marker='o', markersize=4)
            
            # Configure plots
            for ax, title in [(ax1, 'Summer'), (ax2, 'Winter')]:
                # Set title and labels
                ax.set_title(f'{title} Storage State of Charge', fontsize=14, pad=20)
                ax.set_xlabel('Time')
                ax.set_ylabel('Energy (MWh)')
                
                # Set grid style
                ax.grid(True, linestyle='--', color='grey', alpha=0.3)
                
                # Set axis limits and style
                ax.set_ylim(0, y_max * 1.1)
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                for spine in ax.spines.values():
                    spine.set_color('black')
                    spine.set_linewidth(1.0)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %b'))
                ax.tick_params(axis='x', rotation=0)
                
                # Add legend if multiple storage units
                if len(storage_units) > 1:
                    ax.legend(frameon=True, edgecolor='black')
            
            plt.tight_layout()
            
            # Save plot in scenario/figure directory
            filename = os.path.join(scenario_dir, 'storage_soc_comparison.png')
            fig.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            print(f"Created storage SoC comparison plot for scenario {scenario_name}: {filename}")