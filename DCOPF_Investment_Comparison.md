# Investment Model Implementation: Weekly Recap

## Key Additions to the DCOPF Model

### 1. Binary Investment Variables

Added binary decision variables `I_{i,c}` to represent investment decisions:
- `I_{i,c} = 1` if asset `i` is installed in lifetime period `c`
- `I_{i,c} = 0` otherwise

These variables extend the original DCOPF model to handle investment planning over a multi-year horizon.

### 2. Mandatory Asset Functionality

Implemented support for mandatory assets (pre-existing or must-build infrastructure):
- Added `investment_required` parameter to assets:
  - `investment_required = 0`: Asset is mandatory and always active
  - `investment_required = 1`: Asset requires investment decision
- This allows the model to incorporate existing infrastructure or policy-mandated assets
- Mandatory assets have their investment variables automatically set to 1

### 3. Lifetime-Based Approach

Implemented a novel approach to reduce problem size:
- Instead of using one binary variable per asset per year (traditional approach)
- Created "lifetime periods" based on asset lifetimes
- For asset with lifetime `L_i` years in planning horizon `N` years: 
  - Only `⌈N/L_i⌉` variables needed instead of `N`
  - Example: For 10-year lifetime in 20-year horizon: 2 variables instead of 20

### 4. New Constraints

Added constraints linking investment decisions to operational variables:
```
P_{i,t} ≤ P_{max,i} · ∑_{c∈C_{i,t}} I_{i,c}  ∀i∈I, t∈T
```

Where `C_{i,t}` represents the set of lifetime periods covering time period `t`.

Additionally implemented:
- Storage capacity constraints linked to investment decisions

## Code Implementation

1. **Binary Variable Creation**:
   ```python
   # Create binary investment variables for each asset and lifetime period
   investment_vars = {}
   for asset_id, lifetime_periods in lifetime_periods_mapping.items():
       for period_idx in range(len(lifetime_periods)):
           var_name = f"invest_{asset_id}_{period_idx}"
           investment_vars[(asset_id, period_idx)] = model.binary_var(name=var_name)
   ```

2. **Mandatory Asset Implementation**:
   ```python
   # Identify mandatory assets (where investment_required = 0)
   mandatory_assets = []
   for asset_id, data in assets_data.items():
       if data.get('investment_required', 1) == 0:
           mandatory_assets.append(asset_id)
   
   # Force investment for mandatory assets
   for asset_id in mandatory_assets:
       for period_idx in range(len(lifetime_periods_mapping[asset_id])):
           model.add_constraint(
               investment_vars[(asset_id, period_idx)] == 1,
               f"mandatory_{asset_id}_{period_idx}"
           )
   ```

3. **Lifetime Period Mapping**:
   ```python
   def calculate_lifetime_periods(planning_horizon, asset_lifetime):
       """Calculate lifetime periods based on asset lifetime and planning horizon."""
       periods = []
       start_year = 0
       
       while start_year < planning_horizon:
           end_year = min(start_year + asset_lifetime, planning_horizon)
           periods.append((start_year, end_year))
           start_year = end_year
           
       return periods
   ```

4. **Generation Capacity Constraints**:
   ```python
   # Link generation capacity to investment decisions
   for gen_id, t, bus, pmin, pmax in gen_data:
       if gen_id in assets_requiring_investment:
           # Sum of investment variables for this asset that cover time t
           investment_sum = model.sum(
               investment_vars[(gen_id, p)] 
               for p in periods_covering_time(gen_id, t)
           )
           
           # Generation can only occur if asset is installed
           model.add_constraint(
               gen_vars[(gen_id, t)] <= pmax * investment_sum,
               f"gen_capacity_{gen_id}_{t}"
           )
   ```

5. **Capital Cost Calculation**:
   ```python
   # Add capital costs to objective function
   investment_cost = model.sum(
       asset_capex[asset_id] * asset_capacities[asset_id] * investment_vars[(asset_id, p)]
       for asset_id in assets_requiring_investment
       for p in range(len(lifetime_periods_mapping[asset_id]))
   )
   ```

## Results and Visualization

The extended model successfully:
1. Makes investment decisions across the planning horizon
2. Respects asset lifetimes in the investment decisions
3. Handles both optional and mandatory assets appropriately
4. Balances capital and operational costs
5. Handles both traditional generation and storage assets
6. Shows which assets should be installed and when

The implementation includes visualization tools to display:
- Investment decisions by asset and year
- Generation patterns for assets over time periods
- Cost breakdowns between capital and operational expenses 