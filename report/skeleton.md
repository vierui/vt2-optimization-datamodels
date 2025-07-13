'''
vt2: report skeleton 
rui vieira 
03.07.2025
'''

---
# 0  Abstract
- What problem does it solve?
- How is it novel?
- Results in one paragraph?

---
# 1  Introduction
1.1 Context and Motivation ("bring up reason for goals")
- Rising share of renewables → need for integrated planning and short-term forecasting

1.2 VT1 recaps
- DC-power-flow LP prototype; static demand; manual scenario handling, solver, key data flows
- Bottlenecks

1.3 Goals
- (i) migrate optimisation core from LP to MILP with realistic unit-commitment and investment decisions
- (ii) add data-driven PV/Wind forecasting to feed the optimiser
- (iii) consolidate codebase (Poetry + tests) -> Not a goal but genuinily improved

---
# 2 Literature and Toolchain Review 
2.1 Mixed-Integer Programming in Power-System Planning
- Why binaries matter
- Common formulations (UC, storage)
2.2 Short-term PV/Wind Forecasting Methods
- Statistical (SARIMA) vs ML (XGBoost, TCN, etc.)
2.3 Solver Landscape and Selection
- Open-source vs commercial; rationale for choosing CPLEX
2.4 Python Ecosystem for Optimisation and ML
- CVXPY, docplex, Optuna, TensorFlow, XGBoost, etc.

---
# 3 · Problem Definition and Scope 
3.1 Planning Horizon and System Boundaries

3.2 Decision Variables and Constraints
- Investment binaries, UC binaries, storage SOC, DC flows

3.3 Forecast Horizon and accuracy Targets

3.4 KPIs
- Total cost, MILP solve time, forecast MAE/RMSE.

---
# 4 Methodology

4.1 Data Pipeline (+ and Representative Weeks)
- Raw weather + SCADA → DuckDB ; extraction of 3 × 168 h seasonal weeks.

4.2 LP → MILP Logic
	a) Formulation (mathematical model)
	b) Lifetime and Annuity CAPEX handling
	c) Chunk-based binary installation variables
	d) Branch-and-cut improvements (lazy cuts, mipgap)
	e) Solver Change (GLPK → CPLEX)
		- API adapter, licence handling
		- vectorised constraint build.

4.3 Forecasting Module
	a) feature Engineering (lags, solar geometry, weather)
	b) Model pool (SARIMA, Prophet, XGBoost, TCN)
	c) hyper-parameter Tuning
	d) Error Propagation Scenarios for MILP

4.4 Architecture
- Module map ( pre.py, network.py, optimization.py, forecast2/… )
- Poetry env, tests, CI pipeline.

---
# 5 Implementation 
5.1 Code Walk-through
- key classes, CLI entry points, config files

5.2 Data Structures + File Formats
- CSV → Parquet, YAML configs, MLflow logs and pipelines

5.3 Performance Profiling and Optimisation (→ OPTIONAL) 
- Matrix stuffing timings, CPLEX solve benchmarks. 

5.4 Testing & Validation Strategy (→ OPTIONAL)
– Pytest fixtures, 5-bus toy grid, CI results.

---
# 6 Results
6.1 Forecasting Accuracy
- Table of MAE / RMSE for aech model; plots (actual vs forecast).

6.2 MILP vs Legacy LP
- Cost reduction, unit-commitment realism,
- run-time overhead

6.3 Solver Impact (CPLEX vs GLPK)
- Solve time
- mipgap convergence, memory footprint.

6.4 Sensitivity and Scenario Analysis (TO REDO IF TIME LEFT + search again for files) (→ OPTIONAL) 
- congestion price-ing
- Lifetime sensitivity, discount-rate sweep.

6.5 Integrated Workflow Demo (→ OPTIONAL) 
- End-to-end run + gantt like result.

---
# 7 · Discussion
- Interpretation of key findings and changes /advantages LP vs MILP
- Trade-offs: model fidelity vs computation; ML accuracy vs complexity.
- Limitations (data quality, solver scalability, stochastic coupling still WIP, time, tuning of tuning).

---
# 8 · Conclusions and Outlook
- Achievements relative to goals.
- Improvements, dispatch, value

---
# References

---
# Appendices
A. full MILP algebra
B. Config/YAML samples
C. extra plots + test outputs
D. symbols glossary





