from scripts.optimization import dcopf
from scripts.network import IntegratedNetwork, Network
from scripts.pre import process_data_for_optimization

def test_operational_no_discount(toy_case):
    grid, profiles = toy_case
    data = process_data_for_optimization(grid, profiles)
    net = IntegratedNetwork(
        seasons=list(data['seasons_profiles']),
        years=[1,2],
        discount_rate=0.0,
        season_weights=data['season_weights']
    )
    for s in data['seasons_profiles']:
        n = Network(s); n.create_snapshots("2023-01-01", 1)
        n.buses = data['grid_data']['buses'].set_index('id')
        n.generators = data['grid_data']['generators'].set_index('id')
        n.loads = data['grid_data']['loads'].set_index('id')
        net.add_season_network(s,n)

    prob_dict = dcopf(net)
    obj = prob_dict['objective']
    # There must be **no** discount_rate parameter left in the coefficients
    assert "discount_rate" not in str(obj)  # coarse but effective 