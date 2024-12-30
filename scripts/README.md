# Energy System Scenario Analysis

## Overview
This project analyzes different energy system scenarios using DC Optimal Power Flow (DCOPF) to evaluate various combinations of generation sources including nuclear, solar, wind, and storage systems. The analysis considers technical feasibility, economic efficiency, and system reliability across different seasonal patterns.

## Project Structure
```
├── data/
│   ├── working/          # Input data files
│   └── results/          # Generated results and analysis
├── scripts/              # Analysis scripts
└── figures/              # Generated figures and diagrams
```

## Key Features
- DCOPF analysis for multiple scenarios
- Seasonal analysis (winter, summer, autumn/spring)
- Generation vs demand visualization
- AI-powered scenario critique
- Economic and technical feasibility assessment

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

## Results

### Global Analysis
- [Global Comparison Report](data/results/global_comparison_report.md)


### Individual Scenario Reports
- [scenario_1](data/results/scenario_1/scenario_1_analysis.md)
- [scenario_2](data/results/scenario_2/scenario_2_analysis.md)
- [scenario_3](data/results/scenario_3/scenario_3_analysis.md)
- [scenario_4](data/results/scenario_4/scenario_4_analysis.md)

## Visualization Examples
Each scenario analysis includes:
- Generation vs Demand plots for each season
- Generation mix analysis
- Capacity factor comparisons
- Economic metrics
- AI-generated critiques

## Contributing
Feel free to open issues or submit pull requests with improvements.

## License
[MIT License](LICENSE)
