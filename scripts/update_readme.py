import os

def update_readme_with_scenarios():
    """Update README.md with links to all scenario reports"""
    
    # Read the current README content
    with open('../README.md', 'r') as f:
        content = f.read()
    
    # Find all scenario reports in the results directory
    results_dir = '../data/results'
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
    
    # Write the updated content back to README.md
    with open('README.md', 'w') as f:
        f.write(updated_content)
    
    print("README.md updated with current scenario links")

def create_readme_template(readme_path):
    """Create a new README.md file with the template content"""
    template_content = """# Energy System Scenario Analysis

## Overview
This project analyzes different energy system scenarios using DC Optimal Power Flow (DCOPF) to evaluate various combinations of generation sources including nuclear, solar, wind, and storage systems. The analysis considers technical feasibility, economic efficiency, and system reliability across different seasonal patterns.

## Project Structure
```
├── data/
│   ├── working/          # Input data files
│   └── results/          # Generated results and analysis
├── scripts/              # Analysis scripts
├── report/              # LaTeX report files
└── ressources/          # Reference documents and resources
```

## Setup and Installation

This project uses Poetry for dependency management.

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd vt1-energy-investment-model
poetry install
```

3. Set up your OpenAI API key in `.env.local`:
```bash
echo "OPENAPI_KEY=your_api_key_here" > .env.local
```

## Running the Analysis

Activate the Poetry environment and run the analysis:
```bash
poetry shell
python scripts/multi_scenario.py
```

## Results

### Global Analysis
${global_analysis}

### Individual Scenario Reports
${scenario_links}

## Documentation

### Reference Materials
- [Anderson Power Systems Analysis](/ressources/Modelling-Analysis-Electric-Power-Systems_Anderson_2004.pdf)
- [Basic Power Flow Problem](/ressources/05_Basic-Power-Flow-Problem_Anderson_2004.pdf)
- [DC Optimal Power Flow](/ressources/06_Solution-Power-Flow-Problem_Anderson_2004.pdf)

### Report
The detailed analysis and methodology can be found in the [LaTeX report](/report/main.pdf).

## Visualization Examples
Each scenario analysis includes:
- Generation vs Demand plots for each season
- Generation mix analysis
- Capacity factor comparisons
- Economic metrics
- AI-generated critiques

## Project Dependencies
Key dependencies include:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- openai: AI-powered scenario analysis
- python-dotenv: Environment variable management

For a complete list of dependencies, see [pyproject.toml](/pyproject.toml).

## Contributing
Feel free to open issues or submit pull requests with improvements.

## License
[MIT License](/LICENSE)
"""
    
    with open(readme_path, 'w') as f:
        f.write(template_content)
    
    print(f"Created new README.md at {readme_path}")

if __name__ == "__main__":
    update_readme_with_scenarios() 