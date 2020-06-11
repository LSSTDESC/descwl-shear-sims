import numpy as np
from ..trivial_sim import TrivialSim


def test_trivial_sim_smoke():

    seed = 74321
    noise = 0.001
    g1 = 0.02
    g2 = 0.0
    rng = np.random.RandomState(seed)
    sim = TrivialSim(
        rng=rng,
        noise=noise,
        g1=g1,
        g2=g2,
    )

    _ = sim.gen_sim()
