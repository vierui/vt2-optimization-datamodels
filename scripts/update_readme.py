import os
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory"""
    return "/Users/rvieira/Documents/Master/vt1-energy-investment-model"

def update_readme_with_scenarios():
    """Update README.md with links to all scenario reports"""
    project_root = get_project_root()
    readme_path = os.path.join(project_root, 'README.md')
    results_dir = os.path.join(project_root, 'data', 'results')
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        # If README doesn't exist, create it first
        create_readme_template(readme_path)
        with open(readme_path, 'r') as f:
            content = f.read()
    
    # Find all scenario reports
    scenario_reports = []
    for item in os.listdir(results_dir):
        scenario_dir = os.path.join(results_dir, item)
        if os.path.isdir(scenario_dir) and item.startswith('scenario_'):
            report_path = os.path.join(scenario_dir, f'{item}_analysis.md')
            if os.path.exists(report_path):
                scenario_reports.append(item)
    
    # Sort scenarios numerically
    scenario_reports.sort(key=lambda x: int(x.split('_')[1]))
    
    # Create the new scenario links section
    scenario_links = "\n### Individual Scenario Reports\n"
    for scenario in scenario_reports:
        scenario_links += f"- [{scenario}](data/results/{scenario}/{scenario}_analysis.md)\n"
    
    # Replace the existing scenario links section
    import re
    pattern = r"### Individual Scenario Reports\n(?:- \[.*?\]\(.*?\)\n)*"
    updated_content = re.sub(pattern, scenario_links, content)
    
    # Write the updated content
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    print(f"README.md updated with current scenario links at {readme_path}")

def create_readme_template(readme_path):
    """Create a new README.md file with the template content"""
    template_content = """# Energy Investment Analysis Platform

## Overview
This project analyzes different energy system scenarios using DC Optimal Power Flow (DCOPF) to evaluate various combinations of generation sources including nuclear, solar, wind, and storage systems. The analysis considers technical feasibility, economic efficiency, and system reliability across different seasonal patterns.

### Base Network Structure
The analysis is built around a base grid topology with two main load buses. This fundamental structure serves as the foundation for all scenario analyses:

<img src="figures/base_network_topography.png" width="500"/>

*The base network defines the core infrastructure upon which different generation scenarios are evaluated.*

### Scenario Analysis
Each scenario represents a unique combination of:
- Generation asset placement at specific buses
- Storage unit allocation
- Load scaling factors
- Seasonal variations (winter, summer, autumn/spring)

This modular approach allows us to evaluate various investment strategies while maintaining the core network constraints.

## Results

### Global Analysis
- [Global Comparison Report](data/results/global_comparison_report.md)

### Individual Scenario Reports
${scenario_links}

## Project Structure
```
├── data/
│   ├── working/          # Input data files
│   └── results/          # Generated results and analysis
├── scripts/
│   ├── core/            # Core processing modules
│   ├── visualization/   # Plotting and visualization
│   └── utils/          # Helper utilities
└── figures/             # Generated figures and diagrams
```

## Key Features
- DCOPF analysis for multiple scenarios
- Seasonal analysis (winter, summer, autumn/spring)
- Generation vs demand visualization
- AI-powered scenario critique
- Economic and technical feasibility assessment
- Modular scenario creation around base network topology
- Investment optimization for different time horizons

## Running the Analysis
1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key in `.env.local`:
```
OPENAPI_KEY=your_api_key_here
```

3. Run the main analysis:
```bash
python scripts/multi_scenario.py
```

## Visualization Examples
Each scenario analysis includes:
- Generation vs Demand plots for each season
- Generation mix analysis
- Capacity factor comparisons
- Economic metrics
- AI-generated critiques
- Network topology visualizations

## Contributing
Feel free to open issues or submit pull requests with improvements.

## License
[MIT License](LICENSE)
"""
    
    with open(readme_path, 'w') as f:
        f.write(template_content)
    
    print(f"Created new README.md at {readme_path}")

if __name__ == "__main__":
    project_root = get_project_root()
    readme_path = os.path.join(project_root, 'README.md')
    
    # Always create/update the full template first
    create_readme_template(readme_path)
    # Then update the scenario links
    update_readme_with_scenarios() 