#!/usr/bin/env python3

"""
solver.py

A wrapper module for backward compatibility.
This file simply imports and re-exports the functions from optimizer.py and investment_utils.py
"""

# Import from optimizer.py (now in the same directory)
from scripts.optimizer import run_investment_model

# Import from investment_utils.py (now in the same directory)
from scripts.investment_utils import (
    safe_float,
    ensure_float_list,
    calculate_lifetime_periods,
    create_lifetime_periods_mapping,
    create_representative_periods,
    create_typical_periods,
    create_gen_data_for_investment,
    scale_opex_by_lifetime_period,
    calculate_investment_costs
)

# Provide all the functions at the top level of this module for backward compatibility
__all__ = [
    'run_investment_model',
    'safe_float',
    'ensure_float_list',
    'calculate_lifetime_periods',
    'create_lifetime_periods_mapping',
    'create_representative_periods',
    'create_typical_periods',
    'create_gen_data_for_investment',
    'scale_opex_by_lifetime_period',
    'calculate_investment_costs'
] 