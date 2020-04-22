import numpy as np

from ..lsst_bits import SAT
from ..saturation import BAND_SAT_VALS
from ..simple_sim import Sim


def test_star_mask_keywords():
    """
    test star masking using the keyword to the sim
    """
    rng = np.random.RandomState(234)
    sim = Sim(
        rng=rng,
        bands=['r'],
        epochs_per_band=1,
        stars=True,
        stars_kws={
            'density': 3,
            'mag': 15,
        },
        sat_stars=True,
    )

    data = sim.gen_sim()

    se_obs = data['r'][0]
    mask = se_obs.bmask.array
    image = se_obs.image.array

    w = np.where((mask & SAT) != 0)
    assert w[0].size > 0

    assert np.all(image[w] == BAND_SAT_VALS['r'])


def test_star_mask_repeatable():
    """
    test star masking using the keyword to the sim
    """

    for trial in (1, 2):
        rng = np.random.RandomState(234)
        sim = Sim(
            rng=rng,
            bands=['r'],
            epochs_per_band=1,
            stars=True,
            stars_kws={
                'density': 3,
                'mag': 15,
            },
            sat_stars=True,
        )

        data = sim.gen_sim()

        se_obs = data['r'][0]
        mask = se_obs.bmask.array

        w = np.where((mask & SAT) != 0)

        if trial == 1:
            nmarked = w[0].size
        else:
            assert w[0].size == nmarked
