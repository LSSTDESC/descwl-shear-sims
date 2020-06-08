import numpy as np


def test_trivial_sim_smoke():
    from ..trivial_sim import make_trivial_sim

    seed = 74321
    noise = 0.001
    g1 = 0.02
    g2 = 0.0
    rng = np.random.RandomState(seed)
    _ = make_trivial_sim(
        rng=rng,
        noise=noise,
        g1=g1,
        g2=g2,
    )
