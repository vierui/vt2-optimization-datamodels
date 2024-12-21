# %%
import subprocess
import re

def run_and_get_cost(scenario_path):
    result = subprocess.run(["python", "run_scenario.py", "--scenario", scenario_path], capture_output=True, text=True)
    # parse the output to find "Total Weekly Cost: X"
    match = re.search(r"Total Weekly Cost:\s+(\d+(\.\d+)?)", result.stdout)
    if match:
        return float(match.group(1))
    else:
        return None

winter_cost = run_and_get_cost("../data/scenarios/winter")
summer_cost = run_and_get_cost("../data/scenarios/summer")
autumn_spring_cost = run_and_get_cost("../data/scenarios/autumn_spring")

annual_cost = (winter_cost * 13) + (summer_cost * 13) + (autumn_spring_cost * 26)
print(f"Estimated winter weeks cost: {winter_cost * 13}")
print(f"Estimated summer weeks cost: {summer_cost * 13}")
print(f"Estimated annual cost: {annual_cost}")
# %%
'''
python run_scenario.py --scenario ../data/scenarios/winter > winter_result.txt
python run_scenario.py --scenario ../data/scenarios/summer > summer_result.txt
python run_scenario.py --scenario ../data/scenarios/autumn_spring > autumn_spring_result.txt

'''
