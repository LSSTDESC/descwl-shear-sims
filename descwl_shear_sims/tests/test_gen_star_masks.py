"""
copy-paste from my (beckermr) personal code here
https://github.com/beckermr/metadetect-coadding-sims
"""
import numpy as np

from ..gen_star_masks import StarMasks
from ..lsst_bits import STAR, BLEED
from ..simple_sim import Sim


def test_star_mask_smoke():
    rng = np.random.RandomState(2342)
    ra = 200.0
    dec = 15.0

    StarMasks(
        rng=rng,
        center_ra=ra,
        center_dec=dec,
    )


def test_star_mask_works():
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
    nstars = star_masks.set_mask(mask=mask, wcs=wcs)
    assert nstars > 0

    w = np.where((mask & STAR) != 0)
    assert w[0].size > 0
    w = np.where((mask & BLEED) != 0)
    assert w[0].size > 0