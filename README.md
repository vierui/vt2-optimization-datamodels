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

This approach considers all seasons simultaneously in a single optimization problem, ensuring consistent asset decisions across the entire planning horizon.
---