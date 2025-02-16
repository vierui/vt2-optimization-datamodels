from openai import OpenAI
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import os
import matplotlib.pyplot as plt
import re

class ScenarioCritic:
    def __init__(self, api_key: str):
        """Initialize the critic with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        
        self.context_prompt = """
        You are analyzing energy system scenarios with different mixes of generation sources.
        The analysis includes:
        - Annual operational costs
        - Generation per asset type
        - Generation costs per asset type
        - Capacity factors
        - NPVs and annuity
        
        Technologies involved may include:
        - Nuclear
        - Gas
        - Wind 
        - Solar 
        - Battery storage systems
        
        The goal is to evaluate the economic efficiency and technical feasibility of different energy mix scenarios.
        Output in markdown format.
        """

    def generate_critique(self, scenario_data: Dict[str, Any]) -> str:
        """Generate a critique for a single scenario using OpenAI API"""
        
        # Format the generation and cost data
        gen_data = {k: v for k, v in scenario_data.items() if k.startswith('gen_')}
        cost_data = {k: v for k, v in scenario_data.items() if k.startswith('gen_cost_')}
        capacity_factors = {k: v for k, v in scenario_data.items() if k.startswith('capacity_factor_')}
        
        # Create formatted strings for each section
        gen_lines = '\n'.join([f'- {k.replace("gen_", "")}: {v} MW' for k, v in gen_data.items()])
        cost_lines = '\n'.join([f'- {k.replace("gen_cost_", "")}: {v}' for k, v in cost_data.items()])
        cf_lines = '\n'.join([f'- {k.replace("capacity_factor_", "")}: {v}' for k, v in capacity_factors.items()])
        
        scenario_prompt = f"""Scenario Analysis Results:

Scenario Name: {scenario_data.get('scenario_name', 'Unknown')}
Annual Cost: {scenario_data.get('annual_cost', 'N/A')}

Generation per Asset:
{gen_lines}

Generation Costs per Asset:
{cost_lines}

Capacity Factors:
{cf_lines}

Based on these results, provide a brief (200 words max) critical analysis addressing:
1. Economic efficiency of the generation mix
2. System composition strengths/weaknesses
3. Key recommendations for improvement"""

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.context_prompt},
                {"role": "user", "content": scenario_prompt}
            ],
            model="gpt-4o-mini",
            store=True,
        )
        
        return response.choices[0].message.content

    def create_markdown_report(self, scenario_data: Dict[str, Any], critique: str, results_root: str) -> None:
        """Create a markdown report for a single scenario"""
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        scenario_name = scenario_data.get('scenario_name', 'Unknown')
        
        markdown = f"""# Scenario Analysis Report: {scenario_name}
Generated on: {now}

## Scenario Overview
![Scenario Comparison](scenario_comparison.png)

<div style="display: flex; justify-content: space-between;">
<div style="width: 48%;">

## Investment Analysis
- 10-year NPV: {scenario_data.get('npv_10y', 'N/A'):,.2f}
- 20-year NPV: {scenario_data.get('npv_20y', 'N/A'):,.2f}
- 30-year NPV: {scenario_data.get('npv_30y', 'N/A'):,.2f}
- Initial Investment: {scenario_data.get('initial_investment', 'N/A'):,.2f}
- Annual Operating Cost: {scenario_data.get('annual_cost', 'N/A'):,.2f}

</div>
<div style="width: 48%;">

## Generation Statistics

### Generation per Asset
```
{self._format_dict({k: v for k, v in scenario_data.items() if k.startswith('gen_')})}
```

### Generation Costs per Asset
```
{self._format_dict({k: v for k, v in scenario_data.items() if k.startswith('gen_cost_')})}
```

</div>
</div>

## Storage State of Charge
![Storage SOC Comparison](figure/storage_soc_comparison.png)

## Executive Summary
{critique}

---
"""
        # Create scenario folder if it doesn't exist
        scenario_folder = os.path.join(results_root, scenario_name)
        os.makedirs(scenario_folder, exist_ok=True)
        
        # Save markdown report
        report_path = os.path.join(scenario_folder, f"{scenario_name}_analysis.md")
        with open(report_path, 'w') as f:
            f.write(markdown)
        
        print(f"Analysis report saved to '{report_path}'")

    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Helper function to format dictionary data for markdown"""
        return '\n'.join([f"{k.replace('gen_', '').replace('gen_cost_', '').replace('capacity_factor_', '')}: {v}"
                         for k, v in d.items()])

    def _create_seasonal_comparison(self, scenario_name: str, results_root: str) -> None:
        """Create seasonal comparison plot"""
        scenario_folder = os.path.join(results_root, scenario_name)
        figure_folder = os.path.join(scenario_folder, "figure")
        os.makedirs(figure_folder, exist_ok=True)
        
        # Create figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # Plot each season
        for ax, season in zip([ax1, ax2, ax3], ['winter', 'summer', 'autumn_spring']):
            season_image = os.path.join(figure_folder, f'{season}_generation.png')
            if os.path.exists(season_image):
                img = plt.imread(season_image)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'{season.capitalize()} Generation')
        
        # Add overall title
        plt.suptitle(f'Seasonal Generation Comparison - {scenario_name}',
                    fontsize=16, y=1.02)
        
        # Save plot
        plt.savefig(os.path.join(figure_folder, 'seasonal_comparison.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def analyze_scenario(self, scenario_data, results_root):
        """Analyze a single scenario"""
        # Get scenario identifiers
        scenario_id = scenario_data['scenario_id']
        base_scenario = scenario_data['base_scenario']
        
        # Create output directory
        scenario_dir = os.path.join(results_root, base_scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Rest of the analysis code...

    def _format_dict_as_table(self, d: Dict[str, Any], format_str: str = "{:,.2f}") -> str:
        """Helper function to format dictionary data as markdown table rows"""
        return '\n'.join([f"| {k} | {format_str.format(v)} |"
                         for k, v in d.items() if v and not pd.isna(v)])

    def create_global_comparison_report(self, all_scenarios_data: pd.DataFrame, results_root: str) -> None:
        """Create a markdown report comparing all scenarios"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        markdown = f"""# Global Scenarios Comparison Report
Generated on: {now}

## Investment Analysis

```
"""
        # Ensure numeric columns
        numeric_cols = ['npv_10y', 'npv_20y', 'npv_30y', 'annuity_30y', 
                       'initial_investment', 'annual_cost', 'annual_costs']
        for col in numeric_cols:
            if col in all_scenarios_data.columns:
                all_scenarios_data[col] = pd.to_numeric(all_scenarios_data[col], errors='coerce')
        
        # Sort by 30-year NPV
        sorted_scenarios = all_scenarios_data.sort_values('npv_30y', ascending=False)
        
        # Add header for full comparison table
        markdown += "Scenario".ljust(15)  # Reduced width for scenario number
        headers = ["Initial Inv.", "Annual Cost", "10y NPV", "20y NPV", "30y NPV", "Annuity"]
        for header in headers:
            markdown += header.ljust(20)
        markdown += "\n" + "-" * 125 + "\n"  # Adjusted length
        
        # Add data rows
        for idx, row in sorted_scenarios.iterrows():
            try:
                # Extract just the scenario number
                scenario_num = row['scenario_name'].split('_')[-1]
                markdown += (f"{scenario_num}".ljust(15) +  # Scenario number only
                           f"CHF {row.get('initial_investment', 0):,.0f}".replace(",", "'").ljust(20) +
                           f"CHF {row.get('annual_cost', 0):,.0f}".replace(",", "'").ljust(20) +
                           f"CHF {row.get('npv_10y', 0):,.0f}".replace(",", "'").ljust(20) +
                           f"CHF {row.get('npv_20y', 0):,.0f}".replace(",", "'").ljust(20) +
                           f"CHF {row.get('npv_30y', 0):,.0f}".replace(",", "'").ljust(20) +
                           f"CHF {row.get('annuity_30y', 0):,.0f}".replace(",", "'").ljust(20) + "\n")
            except (ValueError, TypeError):
                print(f"Warning: Invalid values for scenario {row['scenario_name']}")
                continue
        
        markdown += "```\n\n"

        # Add annual cost comparison plot with updated styling
        markdown += "## Annual Cost Comparison\n\n"
        
        plt.figure(figsize=(12, 6))
        # Set figure style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        valid_data = all_scenarios_data.dropna(subset=['annual_cost'])
        # Extract scenario numbers correctly - look for scenario_XX pattern
        scenarios = []
        for name in valid_data['scenario_name']:
            # Extract XX from scenario_XX or scenario_XX_something
            match = re.search(r'scenario_(\d+)', name)
            if match:
                scenarios.append(match.group(1))
            else:
                scenarios.append(name)  # fallback
        costs = valid_data['annual_cost']
        
        # Create plot with styling
        ax = plt.gca()
        plt.bar(scenarios, costs)
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        # Add grid
        plt.grid(True, axis='y', linestyle='--', alpha=0.7, color='grey')
        plt.grid(False, axis='x')
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Annual Cost (CHF)')
        plt.title('Annual Cost Comparison Across Scenarios')
        plt.tight_layout()
        
        cost_plot_path = os.path.join(results_root, 'annual_cost_comparison.png')
        plt.savefig(cost_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        markdown += f"![Annual Cost Comparison](annual_cost_comparison.png)\n\n"

        comparative_prompt = f"""Analyze the following scenarios data and provide a comparative analysis:
Scenarios Parameters:
{pd.read_csv('../data/working/scenarios_parameters.csv').to_string()}

Economic Comparison:
{sorted_scenarios[['scenario_name', 'initial_investment', 'annual_cost', 'npv_30y']].to_string()}

Key points to address:
1. Overall trends in cost effectiveness
2. Trade-offs between different generation mixes
3. Key success factors in the better performing scenarios
4. Recommendations for future scenario design

Limit the analysis to 400 words."""

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.context_prompt},
                {"role": "user", "content": comparative_prompt}
            ],
            model="gpt-4o-mini",
            store=True,
        )
        
        markdown += response.choices[0].message.content

        # Save the report
        report_path = os.path.join(results_root, "global_comparison_report.md")
        with open(report_path, 'w') as f:
            f.write(markdown)
        
        print(f"\nGlobal comparison report saved to '{report_path}'")

    def _generate_report_content(self, scenario_data: dict) -> str:
        """Generate the markdown report content for a scenario"""
        scenario_name = scenario_data.get('base_scenario', scenario_data['scenario_name'])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format generation data - filter out nan values
        generation_data = {k: v for k, v in scenario_data.items() 
                          if k.startswith('gen_') and not k.startswith('gen_cost_')
                          and pd.notna(v) and v != 0}
        cost_data = {k: v for k, v in scenario_data.items() 
                     if k.startswith('gen_cost_') and pd.notna(v) and v != 0}

        # Generate critique using OpenAI
        critique = self.generate_critique(scenario_data)

        # Create markdown report
        markdown = f"""# Scenario Analysis Report: {scenario_name}
Generated on: {now}

## Overview
![Annual Summary](figure/annual_summary.png)

<div style="display: flex; justify-content: space-between;">
<div style="width: 48%;">

## Financial Analysis
| Metric | Value |
|--------|--------|
| Initial Investment | CHF {scenario_data.get('initial_investment', 0):,.0f} |
| Annual Operating Cost | CHF {scenario_data.get('annual_cost', 0):,.0f} |
| NPV (10 years) | CHF {scenario_data.get('npv_10y', 0):,.0f} |
| NPV (20 years) | CHF {scenario_data.get('npv_20y', 0):,.0f} |
| NPV (30 years) | CHF {scenario_data.get('npv_30y', 0):,.0f} |

</div>
<div style="width: 48%;">

## Generation Analysis

### Annual Generation by Asset Type
| Asset Type | Generation (MWh) |
|------------|-----------------|
{self._format_dict_as_table(generation_data)}

</div>
</div>

### Generation Costs
| Asset Type | Cost (CHF) |
|------------|------------|
{self._format_dict_as_table(cost_data, "{:,.0f}")}

## Storage State of Charge
![Storage SOC Comparison](figure/storage_soc_comparison.png)

## AI Critical Analysis
{critique}

---
"""
        return markdown

    def _format_dict_as_table(self, d: Dict[str, Any], format_str: str = "{:,.0f}") -> str:
        """Format dictionary as markdown table rows with Swiss number formatting"""
        rows = []
        for k, v in d.items():
            key = k.replace('gen_', '').replace('gen_cost_', '').replace('capacity_factor_', '')
            try:
                # Format number with Swiss style (apostrophes as thousand separators)
                value = format_str.format(float(v)).replace(',', "'")
            except (ValueError, TypeError):
                value = str(v)
            rows.append(f"| {key} | {value} |")
        return '\n'.join(rows)

# Color mapping for technologies
TECH_COLORS = {
    'Gas': '#1f77b4',      # Blue
    'Nuclear': '#ff7f0e',  # Orange
    'Solar': '#2ca02c',    # Green
    'Solar Storage': '#101',  # Purple (as requested)
    'Wind': '#9467bd',     # Purple
    'Wind Storage': '#102'  # Brown (as requested)
}

def plot_winter_summer_generation(data, ax):
    # ... existing setup code ...
    
    # Create bars with updated colors
    winter_bars = ax.barh(y_pos, winter_values, 
                         color=[TECH_COLORS.get(tech, '#333333') for tech in techs],
                         alpha=0.8, label='Winter')
    
    summer_bars = ax.barh(y_pos, summer_values, 
                         color=[TECH_COLORS.get(tech, '#333333') for tech in techs],
                         alpha=0.4, label='Summer')  # Reduced alpha for summer
    
    # Remove bar value annotations
    
    # Update legend
    ax.legend(loc='center right', bbox_to_anchor=(1.15, 0.5),
             title='Season', frameon=False)
    
    # Add clearer season labels
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'Winter', 
            ha='left', va='bottom', fontsize=10)
    ax.text(ax.get_xlim()[1], ax.get_ylim()[1], 'Summer',
            ha='right', va='bottom', fontsize=10)
    
    # ... rest of plotting code ...
