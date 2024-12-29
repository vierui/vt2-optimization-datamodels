from openai import OpenAI
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import os

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

## Generation vs Demand
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

## Economic Analysis
Annual Cost: {scenario_data.get('annual_cost', 'N/A')}

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

    def analyze_scenario(self, scenario_data: Dict[str, Any], results_root: str) -> None:
        """Analyze a single scenario and create markdown report"""
        critique = self.generate_critique(scenario_data)
        self.create_markdown_report(scenario_data, critique, results_root)

    def create_global_comparison_report(self, all_scenarios_data: pd.DataFrame, results_root: str) -> None:
        """Create a markdown report comparing all scenarios"""
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Start with header
        markdown = f"""# Global Scenarios Comparison Report
Generated on: {now}

## Overview
Total Scenarios Analyzed: {len(all_scenarios_data)}
Valid Scenarios: {len(all_scenarios_data[all_scenarios_data['annual_cost'].notna()])}

## Economic Comparison
Scenarios ranked by annual cost:

"""
        
        # Add economic comparison
        economic_comparison = all_scenarios_data[['scenario_name', 'annual_cost']].sort_values('annual_cost')
        markdown += "```\n"
        for _, row in economic_comparison.iterrows():
            if pd.notna(row['annual_cost']):
                markdown += f"{row['scenario_name']}: {row['annual_cost']:,.2f}\n"
        markdown += "```\n\n"

        # Add generation mix comparison
        markdown += "## Generation Mix Comparison\n\n"
        gen_columns = [col for col in all_scenarios_data.columns if col.startswith('gen_')]
        
        for _, row in all_scenarios_data.iterrows():
            if pd.notna(row['annual_cost']):
                markdown += f"### {row['scenario_name']}\n```\n"
                for col in gen_columns:
                    tech_name = col.replace('gen_', '')
                    markdown += f"{tech_name}: {row[col]:,.2f} MW\n"
                markdown += "```\n\n"

        # Add capacity factors comparison
        markdown += "## Capacity Factors Comparison\n\n"
        cf_columns = [col for col in all_scenarios_data.columns if col.startswith('capacity_factor_')]
        
        markdown += "```\n"
        markdown += "Scenario".ljust(30) + "".join([col.replace('capacity_factor_', '').ljust(15) for col in cf_columns]) + "\n"
        markdown += "-" * (30 + 15 * len(cf_columns)) + "\n"
        
        for _, row in all_scenarios_data.iterrows():
            if pd.notna(row['annual_cost']):
                line = str(row['scenario_name']).ljust(30)  # Convert to string and justify
                for col in cf_columns:
                    if pd.notna(row[col]):
                        value_str = f"{row[col]:.2f}"
                    else:
                        value_str = "N/A"
                    line += value_str.ljust(15)
                markdown += line + "\n"
        markdown += "```\n\n"

        # Add AI comparative analysis
        markdown += """## AI Comparative Analysis

Below is an analysis of the key trends and patterns observed across all scenarios:

"""
        
        comparative_prompt = f"""Analyze the following scenarios data and provide a comparative analysis:

Economic Comparison:
{economic_comparison.to_string()}

Key points to address:
1. Overall trends in cost effectiveness
2. Trade-offs between different generation mixes
3. Key success factors in the better performing scenarios
4. Recommendations for future scenario design

Limit the analysis to 300 words."""

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.context_prompt},
                {"role": "user", "content": comparative_prompt}
            ],
            model="gpt-4o-mini",
            store=True,
        )
        
        markdown += response.choices[0].message.content

        # Save the global comparison report
        report_path = os.path.join(results_root, "global_comparison_report.md")
        with open(report_path, 'w') as f:
            f.write(markdown)
        
        print(f"\nGlobal comparison report saved to '{report_path}'")
