import os
import numpy as np
import galsim
import pytest

from ..lsst_bits import SAT, BRIGHT
from ..saturation import BAND_SAT_VALS
from ..simple_sim import Sim
from ..gen_star_masks import add_bright_star_mask
from ..star_bleeds import add_bleed


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
@pytest.mark.parametrize('band', ('r', 'i', 'z'))
def test_star_bleed(band):
    dims = (100, 100)

    cen = [50, 50]

    image = np.zeros(dims)
    bmask = np.zeros(dims, dtype='i4')
    pos = galsim.PositionD(x=cen[1], y=cen[0])
    mag = 12
    band = 'i'

    add_bleed(
        image=image,
        bmask=bmask,
        pos=pos,
        mag=mag,
        band=band,
    )
    add_bright_star_mask(
        mask=bmask,
        x=pos.x,
        y=pos.y,
        radius=10,
        val=BRIGHT,
    )

    assert bmask[cen[0], cen[1]] == SAT | BRIGHT
    assert image[cen[0], cen[1]] == BAND_SAT_VALS[band]


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
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
        star_bleeds=True,
    )

    data = sim.gen_sim()

    se_obs = data['r'][0]
    mask = se_obs.bmask.array
    image = se_obs.image.array

    w = np.where((mask & SAT) != 0)
    assert w[0].size > 0
    assert np.all(image[w] == BAND_SAT_VALS['r'])

    w = np.where(mask & BRIGHT != 0)
    assert w[0].size > 0


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
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
            star_bleeds=True,
        )

        data = sim.gen_sim()

        se_obs = data['r'][0]
        mask = se_obs.bmask.array

        w = np.where((mask & SAT) != 0)

        if trial == 1:
            nmarked = w[0].size
        else:
            assert w[0].size == nmarked
