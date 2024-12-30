from openai import OpenAI
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import os
import matplotlib.pyplot as plt

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

## Investment Analysis
- 10-year NPV: {scenario_data.get('npv_10y', 'N/A'):,.2f}
- 20-year NPV: {scenario_data.get('npv_20y', 'N/A'):,.2f}
- 30-year NPV: {scenario_data.get('npv_30y', 'N/A'):,.2f}
- Initial Investment: {scenario_data.get('initial_investment', 'N/A'):,.2f}
- Annual Operating Cost: {scenario_data.get('annual_cost', 'N/A'):,.2f}

## Seasonal Generation Patterns
![Winter Generation vs Demand](gen_vs_demand_winter.png)

## Generation Statistics

### Generation per Asset
```
{self._format_dict({k: v for k, v in scenario_data.items() if k.startswith('gen_')})}
```

### Generation Costs per Asset
```
{self._format_dict({k: v for k, v in scenario_data.items() if k.startswith('gen_cost_')})}
```

### Capacity Factors
```
{self._format_dict({k: v for k, v in scenario_data.items() if k.startswith('capacity_factor_')})}
```

## AI Critical Analysis
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
        """Create a side-by-side comparison of seasonal generation plots"""
        scenario_folder = os.path.join(results_root, scenario_name, "figure")
        
        # Create figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # Load and plot each seasonal image
        for ax, season in zip([ax1, ax2, ax3], ['winter', 'summer', 'autumn_spring']):
            img_path = os.path.join(scenario_folder, f'hist_total_gen_with_capacity_{season}.png')
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(season.replace('_', '/').title())
            else:
                ax.text(0.5, 0.5, f'No data for {season}', 
                       ha='center', va='center')
                ax.set_title(f'Missing: {season}')
        
        # Add overall title
        plt.suptitle(f'Seasonal Generation Comparison - {scenario_name}', 
                     fontsize=16, y=1.02)
        
        # Save combined plot
        plt.savefig(os.path.join(scenario_folder, 'seasonal_comparison.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

    def analyze_scenario(self, scenario_data: Dict[str, Any], results_root: str) -> None:
        """Analyze a single scenario and generate a report"""
        scenario_name = scenario_data['scenario_name']
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create seasonal comparison plot
        self._create_seasonal_comparison(scenario_name, results_root)
        
        # Format generation data
        generation_data = {k.replace('gen_', ''): v for k, v in scenario_data.items() 
                          if k.startswith('gen_') and not k.startswith('gen_cost_')}
        cost_data = {k.replace('gen_cost_', ''): v for k, v in scenario_data.items() 
                     if k.startswith('gen_cost_')}
        capacity_data = {k.replace('capacity_factor_', ''): v for k, v in scenario_data.items() 
                        if k.startswith('capacity_factor_')}

        # Generate critique using existing method
        critique = self.generate_critique(scenario_data)

        markdown = f"""# Scenario Analysis Report: {scenario_name}
Generated on: {now}

## Overview
![Annual Summary](figure/annual_summary.png)

## Seasonal Generation Patterns
![Seasonal Comparison](figure/seasonal_comparison.png)

## Financial Analysis
| Metric | Value |
|--------|--------|
| Initial Investment | €{scenario_data.get('initial_investment', 0):,.2f} |
| Annual Operating Cost | €{scenario_data.get('annual_cost', 0):,.2f} |
| NPV (10 years) | €{scenario_data.get('npv_10y', 0):,.2f} |
| NPV (20 years) | €{scenario_data.get('npv_20y', 0):,.2f} |
| NPV (30 years) | €{scenario_data.get('npv_30y', 0):,.2f} |

## Generation Analysis

### Annual Generation by Asset Type
| Asset Type | Generation (MWh) |
|------------|-----------------|
{self._format_dict_as_table(generation_data)}

### Generation Costs
| Asset Type | Cost (€) |
|------------|----------|
{self._format_dict_as_table(cost_data)}

### Capacity Factors
| Asset Type | Capacity Factor |
|------------|----------------|
{self._format_dict_as_table(capacity_data, "{:.2%}")}

## AI Critical Analysis
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
        markdown += "Scenario".ljust(30)
        headers = ["Initial Inv.", "Annual Cost", "10y NPV", "20y NPV", "30y NPV", "Annuity"]
        for header in headers:
            markdown += header.ljust(20)
        markdown += "\n" + "-" * 140 + "\n"
        
        # Add data rows
        for _, row in sorted_scenarios.iterrows():
            try:
                markdown += (f"{row['scenario_name']}".ljust(30) +
                           f"${row.get('initial_investment', 0):,.0f}".ljust(20) +
                           f"${row.get('annual_cost', 0):,.0f}".ljust(20) +
                           f"${row.get('npv_10y', 0):,.0f}".ljust(20) +
                           f"${row.get('npv_20y', 0):,.0f}".ljust(20) +
                           f"${row.get('npv_30y', 0):,.0f}".ljust(20) +
                           f"${row.get('annuity_30y', 0):,.0f}".ljust(20) + "\n")
            except (ValueError, TypeError):
                print(f"Warning: Invalid values for scenario {row['scenario_name']}")
                continue
        
        markdown += "```\n\n"

        # Add annual cost comparison plot
        markdown += "## Annual Cost Comparison\n\n"
        
        plt.figure(figsize=(12, 6))
        valid_data = all_scenarios_data.dropna(subset=['annual_cost'])
        scenarios = valid_data['scenario_name']
        costs = valid_data['annual_cost']
        
        plt.bar(scenarios, costs)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Annual Cost ($)')
        plt.title('Annual Cost Comparison Across Scenarios')
        plt.tight_layout()
        
        cost_plot_path = os.path.join(results_root, 'annual_cost_comparison.png')
        plt.savefig(cost_plot_path)
        plt.close()
        
        markdown += f"![Annual Cost Comparison](/data/results/annual_cost_comparison.png)\n\n"

        # Add AI comparative analysis
        markdown += """## AI Comparative Analysis of Energy Scenarios

Below is an analysis of the key trends and patterns observed across all scenarios:

"""
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
