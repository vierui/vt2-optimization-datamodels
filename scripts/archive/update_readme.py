import os
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory"""
    return "/Users/rvieira/Documents/Master/vt2-optimization-datamodels"

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
    scenario_links = "\n### 📊 Detailed Scenario Analysis\n"
    for scenario in scenario_reports:
        scenario_links += f"- [{scenario}](data/results/{scenario}/{scenario}_analysis.md)\n"
    
    # Replace the existing scenario links section
    import re
    pattern = r"### 📊 Detailed Scenario Analysis\n(?:- \[.*?\]\(.*?\)\n)*"
    updated_content = re.sub(pattern, scenario_links, content)
    
    # Write the updated content
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    print(f"README.md updated with current scenario links at {readme_path}")

def create_readme_template(readme_path):
    """Create a new README.md file with the template content"""
    template_content = """# Energy Investment Analysis Platform 🔋💡

## Overview
This project presents a comprehensive investment framework for energy systems, focusing on optimal technology selection and 
placement of electrical generation, conversion, and storage assets. By combining DC Optimal Power Flow (DCOPF) simulations with 
investment analysis, the platform enables data-driven decisions for energy infrastructure planning.

<p align="center">
  <img src="figures/base_network_topography.png" width="600"/>
  <br>
  <em>Base network topology used for scenario analysis</em>
</p>

## 🌟 Key Features

- **Technical Analysis**
  - DC Optimal Power Flow (DCOPF) simulation
  - Multi-scenario analysis capability
  - Seasonal load profile evaluation
  - Storage integration modeling
  - Network constraint handling

- **Economic Assessment**
  - Net Present Value (NPV) calculations
  - Investment sensitivity analysis
  - Technology lifecycle costing
  - Operational cost optimization
  - Risk assessment tools

- **Decision Support**
  - AI-powered scenario analysis
  - Comparative technology assessment
  - Investment optimization
  - Visual analytics and reporting
  - Scenario-based planning

## 📈 Results & Analysis

### Key Findings

- Balanced technology mixes achieved lowest annuities (~1.35M CHF/year)
- Scenarios with renewable generation + storage showed optimal performance
- Gas-heavy scenarios demonstrated higher operational costs
- Storage sizing significantly impacts system economics

### 📊 Detailed Scenario Analysis
${scenario_links}

### 📑 Global Analysis
- [Complete Comparison Report](data/results/global_comparison_report.md)

## 🔧 Technical Implementation

### Project Structure
```
├── data/
│   ├── working/          # Input data and parameters
│   └── results/          # Analysis outputs
├── scripts/
│   ├── core/            # DCOPF and optimization engines
│   ├── visualization/   # Data visualization tools
│   └── utils/           # Helper functions
└── figures/             # Generated visualizations
```

### Technologies Used
- Python for core computation
- PuLP for linear programming
- Pandas for data management
- Matplotlib/Plotly for visualization
- OpenAI API for analysis enhancement

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-investment-model.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure OpenAI API (optional):
```bash
echo "OPENAI_API_KEY=your_key_here" > .env.local
```

4. Run the analysis:
```bash
python scripts/main.py
```

## 📊 Example Visualizations

<p align="center">
  <img src="figures/example_plot.png" width="700"/>
  <br>
  <em>Sample visualization of generation mix across scenarios</em>
</p>

## 📖 Documentation

- [Technical Documentation](docs/technical.md)
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions and feedback:
- 📧 Email: your.email@example.com
- 🌐 LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Special thanks to [Institution Name] for support
- Built using [list key libraries/tools]
- Inspired by [related works/papers]
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