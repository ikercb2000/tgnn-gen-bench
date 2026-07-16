# import modules

from tgnn_gen_bench.metrics import RandomWalkEntropy

from ._metric_graphs import weighted_graph

# random walk entropy tests


def test_random_walk_entropy_is_zero_when_walkers_always_stay_put() -> None:
    metric = RandomWalkEntropy(
        snapshot="s",
        n_samples=2,
        n_walks=8,
        horizon=2,
        stay_probability=1.0,
        seed=7,
    )

    assert metric.compute(weighted_graph()) == [0.0, 0.0]
