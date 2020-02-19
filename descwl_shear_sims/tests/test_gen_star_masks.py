import numpy as np

from ..gen_star_masks import StarMasks
from ..lsst_bits import SAT
from ..simple_sim import (
    Sim,
    SAT_VAL,
)


def test_star_mask_smoke():
    """
    make sure we can generate star masks
    """
    rng = np.random.RandomState(2342)
    ra = 200.0
    dec = 15.0

    StarMasks(
        rng=rng,
        center_ra=ra,
        center_dec=dec,
    )


def test_star_mask_works():
    """
    test star masking, adding them directly to existing mask image
    """
    rng = np.random.RandomState(234)
    sim = Sim(
        rng=rng,
        bands=['r'],
        epochs_per_band=1,
    )

    data = sim.gen_sim()
    center_ra = sim._world_origin.ra.deg
    center_dec = sim._world_origin.dec.deg

    # very high density to ensure we get one
    star_masks = StarMasks(
        rng=rng,
        center_ra=center_ra,
        center_dec=center_dec,
        density=10000,
    )
    se_obs = data['r'][0]
    wcs = se_obs.wcs
    mask = se_obs.bmask.array
    image = se_obs.image.array
    xvals, yvals = star_masks.set_mask_and_image(
        mask=mask,
        image=image,
        wcs=wcs,
        sat_val=SAT_VAL,
    )
    nstars = xvals.size
    assert nstars > 0

    w = np.where((mask & SAT) != 0)
    assert w[0].size > 0


def test_star_mask_keywords():
    """
    test star masking using the keyword to the sim
    """
    rng = np.random.RandomState(234)
    sim = Sim(
        rng=rng,
        bands=['r'],
        epochs_per_band=1,
        sat_stars=True,
        sat_stars_kws={'density': 10000},
    )

    data = sim.gen_sim()

    se_obs = data['r'][0]
    mask = se_obs.bmask.array

    w = np.where((mask & SAT) != 0)
    assert w[0].size > 0


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
            sat_stars=True,
            sat_stars_kws={'density': 10000},
        )

        data = sim.gen_sim()

        se_obs = data['r'][0]
        mask = se_obs.bmask.array

        w = np.where((mask & SAT) != 0)

        if trial == 1:
            nmarked = w[0].size
        else:
            assert w[0].size == nmarked
