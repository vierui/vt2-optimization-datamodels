import tempfile, shutil, json, pandas as pd, numpy as np, pytest, os
from scripts.pre import process_data_for_optimization
from scripts.network import IntegratedNetwork, Network
from scripts.optimization import solve_multi_year_investment

@pytest.fixture(scope="session")
def toy_case(tmp_path_factory):
    """Creates a 2‑bus, 2‑season, 2‑year directory tree on disk."""
    root = tmp_path_factory.mktemp("toy")
    grid = root / "grid";  grid.mkdir()
    profiles = root / "profiles"; profiles.mkdir()

    # --------- static CSVs ---------
    pd.DataFrame([
        {"id":0, "name":"Slack", "v_nom":1.0},
        {"id":1, "name":"LoadBus", "v_nom":1.0}
    ]).to_csv(grid/"buses.csv", index=False)

    pd.DataFrame([{"id":0, "name":"L", "from_bus":0, "to_bus":1,
                   "susceptance":10, "capacity_mw":999}]).to_csv(grid/"lines.csv", index=False)

    pd.DataFrame([   # one cheap thermal at bus 0
        {"id":"G_T", "name":"Therm", "bus":0, "p_nom":500,
         "marginal_cost":10, "type":"thermal",
         "capex":0, "lifetime_years":20, "discount_rate":0.1}
    ]).to_csv(grid/"generators.csv", index=False)

    pd.DataFrame([{"id":"L1", "name":"Demand", "bus":1, "p_mw":100}]).to_csv(grid/"loads.csv", index=False)

    pd.DataFrame([]).to_csv(grid/"storages.csv")   # empty

    # --------- analysis.json with season weights ---------
    json.dump({
        "planning_horizon":{"years":[1,2]},
        "representative_weeks":{"winter":13,"summer":13,"spri_autu":26}
    }, open(grid/"analysis.json","w"))

    # --------- flat load profile (100 MW constant) ---------
    hrs = pd.date_range("2023-01-09", periods=168, freq="h")
    pd.DataFrame({"time":hrs,"value":1.0}).to_csv(profiles/"load-2023.csv", index=False)

    yield str(grid), str(profiles)         # return paths 