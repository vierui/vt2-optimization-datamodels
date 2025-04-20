from scripts.optimization import create_integrated_dcopf_problem
from scripts.network import IntegratedNetwork, Network
from scripts.pre import process_data_for_optimization

def test_single_build_suffices(toy_case):
    grid, profiles = toy_case
    data = process_data_for_optimization(grid, profiles)
    net = IntegratedNetwork(
        seasons=list(data['seasons_profiles']),
        years=[1,2],
        discount_rate=0.0,
        season_weights=data['season_weights']
    )
    # attach season networks with trivial snapshots
    for s in data['seasons_profiles']:
        n = Network(s); n.create_snapshots("2023-01-01", 1)
        n.buses = data['grid_data']['buses'].set_index('id')
        n.lines = data['grid_data']['lines'].set_index('id')
        n.generators = data['grid_data']['generators'].set_index('id')
        n.loads = data['grid_data']['loads'].set_index('id')
        net.add_season_network(s,n)

    problem = create_integrated_dcopf_problem(net)
    # No need to solve: check that exactly one gen_build var exists
    assert len(problem['variables']['gen_build']) == 2   # (G_T,1) (G_T,2) 