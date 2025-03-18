# Multi-Stage Investment Model for Power Systems Using a Chunk-Based Approach

This repository implements a multi-stage investment model for power systems using a chunk-based approach. The model integrates with a DC Optimal Power Flow (DCOPF) solver to optimize both investment decisions and operational dispatch.

## Overview

The chunk-based multi-stage investment model addresses the challenge of optimizing investment decisions for power systems over a long planning horizon. Key features include:

- **Multi-stage investment decisions** based on asset lifetimes
- **Chunk-based approach** to reduce problem size by dividing the planning horizon into chunks based on asset lifetimes
- **Integration with DCOPF** for operational feasibility and cost optimization
- **Support for different asset lifetimes and capital costs**
- **Flexible planning horizon**

## File Structure

- `scripts/dcopf_investment.py`: Main implementation of the multi-stage investment model
- `scripts/investment_utils.py`: Utility functions for chunk calculation and data preparation
- `scripts/dcopf_mip.py`: Mixed Integer Programming extension of the DCOPF solver
- `scripts/test_investment.py`: Test script to demonstrate the investment model

## Model Description

### Chunk-Based Approach

For each asset with lifetime L_i and planning horizon N:
1. Divide the planning horizon into m_i = floor(N / L_i) full chunks
2. If r_i = N mod L_i > 0, add a leftover chunk of length r_i
3. Each chunk represents a potential investment decision

### Binary Variables

For each asset i and chunk k:
- b_{i,k} = 1 if asset i is installed for chunk k
- b_{i,k} = 0 otherwise

### Cost Function

Total Cost = Operational Cost + Investment Cost

- **Operational Cost**: Sum of DCOPF dispatch costs for each chunk, scaled by the chunk duration
- **Investment Cost**: Sum of (C_i / L_i) * b_{i,k} for each asset i and chunk k, where C_i is the full capital cost

### Constraints

1. **Generation Capacity**: 0 <= P_{i,k,h} <= Pmax_i * b_{i,k}
2. **DC Power Flow**: Standard DCOPF constraints for each chunk
3. **Power Balance**: Supply must equal demand at each bus

## Usage

### Example

```python
from scripts.dcopf_investment import dcopf_investment

# Define asset lifetimes and capital costs
asset_lifetimes = {
    1: 20,  # Baseload: 20 years
    2: 15,  # Mid-merit: 15 years
    3: 10,  # Peaker: 10 years
    4: 7,   # Renewable: 7 years
    5: 5    # Storage: 5 years
}

asset_capex = {
    1: 1500000,  # Baseload: $1.5M/MW
    2: 1000000,  # Mid-merit: $1M/MW
    3: 500000,   # Peaker: $0.5M/MW
    4: 800000,   # Renewable: $0.8M/MW
    5: 400000    # Storage: $0.4M/MWh
}

# Run the investment model
investment_results = dcopf_investment(
    gen_time_series, branch, bus, demand_time_series,
    planning_horizon=10,
    start_year=2023,
    asset_lifetimes=asset_lifetimes,
    asset_capex=asset_capex,
    operational_periods_per_year=4,  # 4 typical seasons
    hours_per_period=24  # 24 hours per season
)

# Extract investment decisions
investment_df = investment_results['investment']
print(investment_df[investment_df['decision'] == 1])
```

### Running Tests

To run the test script:

```bash
python scripts/test_investment.py
```

## Results

The investment model returns a dictionary with the following keys:
- `investment`: DataFrame with investment decisions
- `generation_by_chunk`: Generation results for each chunk
- `flows_by_chunk`: Flow results for each chunk
- `investment_costs`: Investment costs by asset and chunk
- `cost_summary`: Summary of total, investment, and operational costs
- `status`: Solver status
- `chunk_periods`: Mapping of chunks to time periods
- `representative_periods`: Representative periods for each chunk

## Requirements

- Python 3.6+
- CPLEX Optimization Studio
- pandas, numpy, matplotlib

## References

The chunk-based approach for multi-stage investment is based on the idea of dividing the planning horizon into chunks corresponding to asset lifetimes, allowing for more efficient representation of investment decisions compared to a year-by-year approach. 