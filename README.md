# Energy Investment Analysis Platform
*Currently under development*

### Overview
This project represents the second specialization project in my Master's studies in Business Engineering at ZHAW (Zurich University of Applied Sciences). It builds upon [an investment model for energy assets](https://github.com/vierui/vt1-energy-investment-model.git) in energy system modeling and optimization, with a focus on linear programming, investment metrics via scenario analysis featuring automated reporting and AI-assisted decision support.

The new platform creates an advanced mathematical modeling toolset for electrical energy system investment optimization. Using sophisticated analytics and forecasting techniques, the goal is to provide comprehensive decision support for energy market stakeholders and business decisions.

### Methodologies Implemented
- Switch from Linear Programming (LP) to Mixed Integer Linear Programming (MIP) to incorporate capital investment (CAPEX) with binary decision variables, enabling optimal asset portfolio selection based on actual availability.
- Forecasting time-series models using representative weeks for seasonal variations
- Stochastic Programming & Risk Assessment for electricity contracts on spot markets
- Data Validation Against Real-World Measurements

### Multi-Year Power Grid Optimization Framework
This framework implements a simplified multi-year power grid optimization tool with the following key features:

- **Single Binary Variable Approach**: Utilizes a streamlined approach with a single binary variable for each asset per year (no re-install or replacement complexity)
- **Annualized CAPEX Cost**: Calculates CAPEX using `capex_per_mw * p_nom / lifetime` formula
- **Seasonal Representation**: Models separate networks for winter, summer, and spring/autumn using representative weeks
- **Storage Optimization**: Supports energy storage with state-of-charge constraints and efficiencies
- **Visualization Tools**: Generates implementation plans, seasonal profiles, generation mix, and timeline plots
- **Detailed Production Cost Analysis**: Provides production cost breakdowns when using the `--analyze-costs` flag

While the included example primarily demonstrates thermal generators, the framework fully supports renewable generation including:
- Wind generation with time-varying profiles
- Solar generation with time-varying profiles
- Energy storage with configurable efficiency parameters

### Running the Optimization
To run a multi-year optimization, use the following command:

```
python scripts/main.py --grid-file path/to/grid/data --profiles-dir path/to/profiles --output-dir results
```

Optional flags:
- `--analyze-costs`: Run detailed production and cost analysis
- `--save-network`: Save the optimized network to a pickle file
- `--solver-options`: Pass JSON string with solver options (e.g. `'{"timelimit":3600}'`)

### Input Data Structure
The framework expects:
1. Grid data in CSV format (buses, lines, generators, loads, storage_units)
2. Time series profiles for loads, wind, and solar generation
3. Configuration via analysis.json for planning horizon, load growth, etc.

### Output Results
Results are saved to the specified output directory and include:
- Implementation plan (JSON)
- Seasonal resource profiles (plots)
- Generation mix visualizations
- Implementation timeline
- Detailed cost analysis (if requested)

For additional details, please refer to the code documentation or contact the repository owner.

### New Feature: Integrated Year-by-Year Optimization
This codebase now includes an integrated, year-by-year optimization method that offers several advantages:

- **Coherent Asset Selection**: Ensures a single coherent set of assets serves all seasons
- **Seasonal Interactions**: Accounts for interactions between seasons, particularly for energy storage
- **Realistic Planning**: Produces more realistic planning scenarios that reflect real-world constraints
- **Backward Compatibility**: Maintains compatibility with the existing codebase

To use the integrated optimization approach, run with the `--integrated` flag:

```
python scripts/main.py --grid-file data/grid/grid_data.csv --profiles-dir data/processed --integrated
```

---

This approach considers all seasons simultaneously in a single optimization problem, ensuring consistent asset decisions across the entire planning horizon.
