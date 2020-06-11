import pytest
import numpy as np
from ..trivial_sim import TrivialSim


@pytest.mark.parametrize('dither', [True, False])
def test_trivial_sim_smoke(dither):

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
        dither=dither,
    )

    _ = sim.gen_sim()
