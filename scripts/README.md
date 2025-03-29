# Power Grid Optimization

A PyPSA-like implementation of a power grid optimization model with generation, storage, and transmission constraints.

## Project Structure

The project is organized in a modular way, following a similar approach to PyPSA:

- **components.py**: Contains component classes (Bus, Generator, Load, Storage, Line) with methods to create DataFrames
- **network.py**: Main Network class that holds all components and handles the optimization (similar to PyPSA's Network)
- **main.py**: Example script that creates a network and runs the optimization

## Usage

To run the optimization:

```bash
python main.py
```

You can also create your own networks:

```python
from network import Network

# Create an empty network
net = Network()

# Set time horizon
net.set_snapshots(24)  # 24 hour optimization

# Add components
net.add_bus('Bus1')
net.add_generator('Gen1', 'Bus1', capacity=100, cost=50)
net.add_load('Bus1', p_set=30)

# Run optimization
net.lopf(solver='CPLEX')

# Display results
net.summary()
```

## Requirements

- Python 3.x
- CVXPY
- CPLEX solver (must be installed and accessible to CVXPY)
- Pandas
- NumPy

## Model Description

The model implements a DC optimal power flow problem with:
- Generator dispatch decisions
- Storage charging/discharging decisions
- Power flow on transmission lines
- Nodal power balance constraints
- Storage energy balance dynamics
- Generator and transmission capacity limits

The objective is to minimize the total generation cost over a 24-hour period.

## PyPSA-like Structure

This implementation follows the PyPSA organization pattern:

- A central Network object contains all components
- Static data is stored in pandas DataFrames (buses, generators, lines, etc.)
- Time-series data is stored in specialized DataFrames (*_t attributes)
- The optimization method (lopf) builds and solves the optimization problem
- Results are stored back in the Network object for easy access and analysis 