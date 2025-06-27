# Feedback KW12-13

## Simple test configuration
- 5-bus test system with loads distributed across buses
- Mix of mandatory assets (nuclear at bus 1) and optional assets requiring investment decisions
- Grey assets (like nuclear) have production costs but no investment decisions
- Green assets (renewables, storage) have zero production costs but require investment decisions

## Approach
### Integrated or 'Sequential' Optimization
- **Chosen Approach**: Integrated optimization that considers both investment and operational costs simultaneously --> (issues with the balance no cost for nuclear capex --> Heavily )
- **Alternative**: Simpler - Sequential approach (run DCOPF first, then make investment decisions based on results)

### Base DCOPF vs. Investment Model
Extensions
- **Binary Decision Variables**: Added binary variables for each investable asset to represent investment decisions (0 = don't build, 1 = build)
- **Generator Capacity Constraints**: Modified power output constraints to be conditional on investment decisions using the form `P[g,t] ≤ Pmax[g] * binary_var[g]`
- **Storage Capacity Constraints**: Similar constraints for storage charging/discharging: `P_charge[s,t] ≤ Pmax[s] * binary_var[s]` and `P_discharge[s,t] ≤ Pmax[s] * binary_var[s]`
- **Existing Investment Handling**: The model accepts already installed assets as input, fixing their binary variables to 1 with zero additional cost

## Multi-Year Planning Methodology

### Lifetime Handling
- The model tracks installation years and computes retirement years based on asset lifetimes
- At each year in the planning horizon, we check which assets reach end-of-life and mark them for retirement
- The `installation_timeline` dictionary maps each asset to its installation/retirement actions across the planning horizon

### Reinstallation After End-of-Life
When an asset reaches end-of-life:

- It's removed from the "installed assets" list 
- The next DCOPF optimization is run with updated `existing_investment` status
- The model then decides whether to reinstall the retired asset or choose different assets based on economic viability
- This happens automatically at each year step in the `investment_dcopf_planning` function

### Cost Function and Planning Horizon

The objective function combines both operational costs (OPEX) and investment costs (CAPEX):

**Mathematical Expression (Corrected):**
```
min Z = ∑(g∈G) ∑(t∈T) gencost[g] * P[g,t] + ∑(i∈I) b[i] * annual_capex[i] * planning_horizon
```

Where:
- `G` = set of generators
- `I` = set of assets requiring investment
- `b[i]` = binary variable for asset i (1 = build, 0 = don't build)
- `annual_capex[i]` = capex[i] / lifetime[i]
- `planning_horizon` = number of years in planning period

**Annuity-Based CAPEX Approach:**
Instead of modeling a single upfront cost at the time of installation, we use an annuity-based approach:
- The total CAPEX is converted to an annual payment over the asset's lifetime
- This annual cost is then applied for each year in the planning horizon
- This approach simplifies the model and allows for consistent comparison between assets with different lifetimes

**Code Implementation:**
```python
# For generators with operational costs (grey assets like nuclear)
problem.objective.set_sense(problem.objective.sense.minimize)
for g in G:
    for t in T:
        problem.objective.set_linear(gen_vars[g, t], gen_costs[g])

# For investment decisions - annuity-based approach
for asset_id in investment_required:
    # Check if already installed
    is_already_installed = existing_investment.get(asset_id, False)
    
    # Calculate annual equivalent cost using the annuity approach
    annual_capex = asset_capex[asset_id] / asset_lifetimes[asset_id]
    total_capex_over_horizon = annual_capex * planning_horizon
    
    # Add to objective with cost if not already installed
    if is_already_installed:
        obj_value = 0.0  # No additional cost
    else:
        obj_value = total_capex_over_horizon
        
    problem.variables.add(
        obj=[obj_value],
        types=["B"],
        names=[invest_vars[asset_id]]
    )
```

- **Important**: If an asset's lifetime exceeds the planning horizon, we still charge the full annualized cost for the entire planning period
- This creates a slight overcharging effect but provides a conservative estimate that properly values long-lived assets

## Results
### Current results
- The model successfully identifies optimal investment decisions across the planning horizon
- The visualization tools show installation timelines and active assets by year
- Results confirm economic viability of reinstalling certain assets after end-of-life. Given current model, it is normal to not see jumps in assets.
- **Cost Breakdown**:
  - Total System Cost: $4,227,336.64
  - CAPEX (Investment Cost): $4,214,285.71 (99.7%)
  - OPEX (Operational Cost): $13,050.92 (0.3%)
- **Selected Assets**: The model chose to install Assets 3, 4, and 7 (generators at buses 2 and 3, and storage at bus 4)
- **Optimal planning**:
![Installation Timeline](results/dcopf_planning/installation_timeline.png)


### Next : data and time-series mgmt.
- **Current implementation**: 
Uses a simple 24-hour demand curve multiplied by 365 days to represent annual demand
- **Next steps**: 
Implement weekly data using a seasonal model:
  - 3 representative weeks per season (winter, summer, spring/autumn)
  - Spring and autumn treated as one season with similar characteristics
  - Scaling to represent a full 52-week year using real load profile data
  - This will better capture seasonal variations in demand and renewable generation

