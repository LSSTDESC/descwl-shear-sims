import pytest
import numpy as np
from ..trivial_sim import TrivialSim


@pytest.mark.parametrize('dither,rotate', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_trivial_sim_smoke(dither, rotate):

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
        rotate=rotate,
    )

    _ = sim.gen_sim()


@pytest.mark.parametrize('epochs_per_band', [1, 2, 3])
def test_trivial_sim_epochs(epochs_per_band):

    bands = ['r', 'i', 'z']
    seed = 7421
    noise = 0.001
    g1 = 0.02
    g2 = 0.0
    rng = np.random.RandomState(seed)
    sim = TrivialSim(
        rng=rng,
        noise=noise,
        g1=g1,
        g2=g2,
        dither=True,
        rotate=True,
        bands=bands,
        epochs_per_band=epochs_per_band,
    )
    data = sim.gen_sim()

    for band in bands:
        assert band in data
        assert len(data[band]) == epochs_per_band
