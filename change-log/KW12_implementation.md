Typical Weeks Framework for Power System Optimization

I've implemented a comprehensive framework for using typical weeks to represent a full year in your power system optimization models. This framework follows the approach you outlined in your proposal, allowing you to reduce the computational burden while still capturing seasonal variations.

Key Components:
Data Processing (data_processor.py)
Functions for loading CSV data
Season tagging and identification
Duration calculation for weighting
Week Selection (week_selection.py)
Manual week selection (select specific weeks)
Automated clustering for representative weeks
Daily profile extraction and analysis
Model Building (model_builder.py)
Creating multi-block datasets with block weights
Converting to generator and demand time series
Properly handling renewable capacity factors
Framework Integration (framework.py)
Main TypicalWeeksModel class for end-to-end workflow
Integration with existing DCOPF and investment planning models
Visualization and result annualization functions
Example Script (typical_weeks_example.py)
Complete examples using your test system
Generation of synthetic time series data
Both simple investment and multi-year planning examples

Key Features:
Block Weighting Approach
Each typical week represents a season and is assigned a weight (number of weeks in that season)
These weights are incorporated directly in the objective function to scale costs appropriately
Multiple Selection Methods
Manual selection of specific weeks
Automated clustering to find representative patterns
Storage Handling
Each block is modeled independently for storage state transitions
Proper initial and final conditions for each block
Result Annualization
Scaling results by block weights to get annualized values
Summarizing generation, prices, and flows

How to Use the Framework:
The framework is designed to be easy to use. You simply:
Create a TypicalWeeksModel instance
Load your time series data (load, wind, solar)
Extract typical weeks
Build the multi-block model
Prepare DCOPF inputs using your generator, branch, and bus data
Run the appropriate optimization model (DCOPF, investment, or planning)
Analyze and visualize the results
---