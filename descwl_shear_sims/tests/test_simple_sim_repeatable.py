import numpy as np
import pytest
from ..simple_sim import Sim


@pytest.mark.parametrize('gals_type', ['exp', 'wldeblend'])
def test_simple_sim_noise_repeat(gals_type):
    """
    test that the noise in the signal image is repeated for different shear
    values

    Test that the noise image, used for metacal corrections, is repeated

    We can test wldeblend here because there are no galaxies present
    """

    seed = 100

    sim_plus = Sim(
        rng=np.random.RandomState(seed),
        gals=False,
        gals_type=gals_type,
        g1=0.02,
    )
    sim_minus = Sim(
        rng=np.random.RandomState(seed),
        gals=False,
        gals_type=gals_type,
        g1=-0.02,
    )

    data_plus = sim_plus.gen_sim()
    data_minus = sim_minus.gen_sim()

    for band in sim_plus.bands:
        for epoch in range(sim_plus.epochs_per_band):
            obs_plus = data_plus[band][epoch]
            obs_minus = data_minus[band][epoch]

            # repeatability tests
            assert np.all(obs_plus.image.array == obs_minus.image.array)
            assert np.all(obs_plus.noise.array == obs_minus.noise.array)

            # sanity checks
            # testing plus is enough, we showed they are the same above
            assert np.all(obs_plus.image.array != 0)
            assert np.all(obs_plus.noise.array != 0)
            assert np.all(obs_plus.image.array != obs_plus.noise.array)

            expected_var = 1/obs_plus.weight.array[0, 0]

            nvar = obs_plus.noise.array.var()
            assert abs(nvar/expected_var-1) < 0.015
            nvar = obs_plus.image.array.var()
            assert abs(nvar/expected_var-1) < 0.015
