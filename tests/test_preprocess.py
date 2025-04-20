from scripts.pre import process_data_for_optimization

def test_weights_sum(toy_case):
    grid, profiles = toy_case
    data = process_data_for_optimization(grid, profiles)
    w = data['season_weights']
    assert abs(sum(w.values())-52) < 1e-6
    assert w["winter"] == 13 