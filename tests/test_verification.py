from gameedgealloc.verification import verify_followers_and_prices
from gameedgealloc.models import SystemConfig
from gameedgealloc.workload import WorkloadConfig,generate_trace


def test_follower_concavity_and_elasticity_numeric():
    cfg=SystemConfig()
    tasks=generate_trace(WorkloadConfig(n_users=10,slots=1,seed=99)).slots[0]
    check=verify_followers_and_prices(tasks,cfg)
    assert check['negative_hessian_diagonals_all']
    assert check['demand_monotone_decreasing']
    assert check['price_map_monotone_decreasing']
    assert check['elasticity_half_bound_respected']
    assert check['slope_within_conservative_bound']
