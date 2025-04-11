#!/usr/bin/env python3
"""
Post-processing module for the simplified multi-year approach.

We demonstrate:
  - Implementation plan generation
  - Possibly cost breakdown if desired
"""

import os
import json
import numpy as np

# A custom JSON encoder to handle e.g. numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_implementation_plan(integrated_network, output_dir="results"):
    """
    Build a simple plan describing in which year each generator or storage 
    is actually installed (binary=1).
    Then optionally we can parse any dispatch results if needed.

    integrated_network: 
      - Should have an 'integrated_results' dict with a 'variables' sub-dict 
        containing numeric values (not CVXPY variables).

    Returns: A dictionary summarizing the plan, also saved as JSON in output_dir.
    """

    if not hasattr(integrated_network, 'integrated_results') or integrated_network.integrated_results is None:
        print("[post.py] No integrated_results found on integrated_network.")
        return {}

    result = integrated_network.integrated_results
    if 'variables' not in result:
        print("[post.py] integrated_results has no 'variables'.")
        return {}

    var_dict = result['variables']

    gen_installed = var_dict.get('gen_installed', {})
    storage_installed = var_dict.get('storage_installed', {})

    # Initialize the plan with defaults
    plan = {
        'years': integrated_network.years,
        'generators': {},
        'storage': {},
        'objective_value': result.get('value', 0),
        'status': result.get('status', 'unknown')
    }

    # Extract generator installation
    # gen_installed keys look like (g, y) -> value
    # We'll see in which year it's "1"
    for (g, y), val in gen_installed.items():
        # That means generator g is installed in year y
        if val > 0.5:
            if g not in plan['generators']:
                plan['generators'][g] = {
                    'years_installed': []
                }
            plan['generators'][g]['years_installed'].append(y)

    # Similarly for storage
    for (s, y), val in storage_installed.items():
        if val > 0.5:
            if s not in plan['storage']:
                plan['storage'][s] = {
                    'years_installed': []
                }
            plan['storage'][s]['years_installed'].append(y)

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    plan_file = os.path.join(output_dir, "implementation_plan.json")
    try:
        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2, cls=NumpyEncoder)
        print(f"[post.py] Implementation plan saved to {plan_file}")
    except Exception as e:
        print(f"[post.py] Error saving plan JSON: {e}")

    return plan


def save_cost_breakdown(integrated_network, output_dir="results"):
    """
    Optional: If you store cost breakdown by year or season, you can produce 
    a separate JSON. Here is a skeleton that you can adapt.
    """

    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No integrated_results to derive cost breakdown.")
        return {}

    result = integrated_network.integrated_results
    # If you had separate operational/capital cost in your solution, 
    # you'd read them from the model or parse from the variables. 
    # Suppose you do it manually or it's not separated. Just store total.
    cost_info = {
        'objective': result.get('value', None),
        'status': result.get('status', 'unknown')
        # Could add more details if your solver tracked them
    }

    cost_file = os.path.join(output_dir, "cost_breakdown.json")
    try:
        with open(cost_file, "w") as f:
            json.dump(cost_info, f, indent=2, cls=NumpyEncoder)
        print(f"[post.py] Cost breakdown saved to {cost_file}")
    except Exception as e:
        print(f"[post.py] Error saving cost breakdown: {e}")

    return cost_info