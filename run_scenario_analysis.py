import os
from dotenv import load_dotenv
from create_master_invest import InvestmentAnalysis
from scenario_critic import ScenarioCritic
from datetime import datetime

def create_markdown_report(results, critiques):
    """Create a markdown report from analysis results and critiques"""
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    markdown = f"""# Energy Scenario Analysis Report
Generated on: {now}

## Executive Summary
This report analyzes the top 5 energy scenarios based on Net Present Value (NPV) calculations.
Each scenario is evaluated for its economic viability and technical composition.

## Detailed Scenario Analysis

"""
    
    # Add each scenario analysis
    for scenario_name, critique in critiques.items():
        scenario_data = results.loc[scenario_name]
        
        markdown += f"""### Scenario: {scenario_name}

#### Technical Specifications
```
{scenario_data['installed_capacity']}
```

#### Financial Metrics
- Initial Investment: {scenario_data['initial_investment']}
- Annual Costs: {scenario_data['annual_costs']}
- NPV: {scenario_data['npv']}
- Annuity: {scenario_data['annuity']}

#### AI Critical Analysis
{critique}

---

"""
    
    return markdown

def main():
    # Load environment variables
    load_dotenv('../.env.local')
    api_key = os.getenv('OPENAPI_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key not found in .env.local file")

    # Run investment analysis
    analysis = InvestmentAnalysis()
    results = analysis.analyze_scenario(
        'data/results/scenario_results.csv',
        'data/working/master_gen.csv'
    )
    
    # Save Excel results first
    results.to_excel('data/results/full_analysis_results.xlsx')
    print("\nFull results saved to 'data/results/full_analysis_results.xlsx'")
    
    # Initialize critic with API key
    critic = ScenarioCritic(api_key)
    
    # Get top 5 scenarios and generate critiques
    top_5 = results.head(5)
    critiques = {}
    
    print("\nGenerating AI critiques for top 5 scenarios...")
    for idx, row in top_5.iterrows():
        critiques[idx] = critic.generate_critique(row.to_dict())
    
    # Create markdown report
    markdown_content = create_markdown_report(top_5, critiques)
    
    # Save markdown report
    report_path = 'data/results/scenario_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"\nDetailed analysis report saved to '{report_path}'")
    
    # Print the markdown content to console as well
    print("\nReport Content:")
    print(markdown_content)

if __name__ == "__main__":
    main() 