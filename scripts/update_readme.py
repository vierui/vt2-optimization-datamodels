import os

def update_readme_with_scenarios():
    # Get the project root directory (2 levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths relative to project root
    readme_path = os.path.join(project_root, 'README.md')
    results_dir = os.path.join(project_root, 'data', 'results')
    
    # Check if README exists, if not create it from template
    if not os.path.exists(readme_path):
        print("README.md not found. Creating from template...")
        create_readme_template(readme_path)
    
    # Read the template
    with open(readme_path, 'r') as f:
        template = f.read()
    
    # Get list of scenario folders
    scenario_folders = [f for f in os.listdir(results_dir) 
                       if os.path.isdir(os.path.join(results_dir, f)) 
                       and not f.startswith('.')]
    
    # Generate markdown links for each scenario using relative paths
    scenario_links = []
    for folder in sorted(scenario_folders):
        analysis_file = f"{folder}_analysis.md"
        if os.path.exists(os.path.join(results_dir, folder, analysis_file)):
            # Use relative path from repository root
            scenario_links.append(f"- [{folder}](/data/results/{folder}/{analysis_file})")
    
    # Add global comparison report if it exists
    global_report = os.path.join(results_dir, "global_comparison_report.md")
    if os.path.exists(global_report):
        global_analysis = "- [Global Comparison Report](/data/results/global_comparison_report.md)"
    else:
        global_analysis = "- Global Comparison Report (not generated yet)"
    
    # Replace placeholders with generated links
    readme_content = template.replace("${scenario_links}", "\n".join(scenario_links))
    readme_content = readme_content.replace("${global_analysis}", global_analysis)
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README.md updated at {readme_path}")

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