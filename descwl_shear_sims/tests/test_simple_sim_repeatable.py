import numpy as np
import pytest
from ..simple_sim import Sim


@pytest.mark.parametrize('gals_type', ['exp', 'wldeblend'])
def test_simple_sim_noise_repeat(gals_type):
    seed = 100

    if gals_type == 'exp':
        gals_kws = {'density': 0}
    else:
        gals_kws = {}

    sim_plus = Sim(
        rng=np.random.RandomState(seed),
        gals=False,
        gals_type=gals_type,
        gals_kws=gals_kws,
        g1=0.02,
    )
    sim_minus = Sim(
        rng=np.random.RandomState(seed),
        gals=False,
        gals_type=gals_type,
        gals_kws=gals_kws,
        g1=-0.02,
    )

    data_plus = sim_plus.gen_sim()
    data_minus = sim_minus.gen_sim()

    for band in sim_plus.bands:
        for epoch in range(sim_plus.epochs_per_band):
            obs_plus = data_plus[band][epoch]
            obs_minus = data_minus[band][epoch]

            assert np.all(obs_plus.image == obs_minus.image)
            assert np.all(obs_plus.noise == obs_minus.noise)
            assert np.all(obs_plus.image != 0)
            assert np.all(obs_plus.noise != 0)
            assert np.all(obs_minus.image != 0)
            assert np.all(obs_minus.noise != 0)
