import os
import pytest
import numpy as np

from ..simple_sim import SimpleSim
from ..lsst_bits import SAT, BRIGHT
from ..saturation import BAND_SAT_VALS


def test_simple_sim_fixed_stars():
    sim = SimpleSim(
        rng=120,
        gals_kws={'density': 10},
        stars=True,
    )
    sim.gen_sim()


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_simple_sim_sample_stars():
    sim = SimpleSim(
        gals_type='wldeblend',
        rng=335,
        stars=True,
        stars_type='sample',
    )
    sim.gen_sim()


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_simple_sim_sample_star_density():
    min_density = 2
    max_density = 20
    sim = SimpleSim(
        rng=335,
        gals_type='wldeblend',
        stars=True,
        stars_type='sample',
        stars_kws={
            'density': {
                'min_density': min_density,
                'max_density': max_density,
            }
        },
    )
    sim.gen_sim()

    assert min_density < sim.star_density < max_density


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_simple_sim_sample_star_minmag_smoke():
    sim = SimpleSim(
        rng=335,
        gals_type='wldeblend',
        stars=True,
        stars_type='sample',
        stars_kws={
            'min_mag': 20,
        },
    )
    sim.gen_sim()


@pytest.mark.parametrize('subtract_bright', [False, True])
def test_simple_sim_bright_stars(subtract_bright):
    """
    make sure we get saturation and bright marked for
    bright stars
    """
    sim = SimpleSim(
        rng=10,
        coadd_dim=51,
        buff=0,
        gals=False,
        layout_type='grid',
        layout_kws={'dim': 1},
        stars=True,
        stars_kws={
            'density': 1,
            'mag': 15,
            'subtract_bright': subtract_bright,
        },
        saturate=True,
    )
    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    for band in sim.bands:
        for epoch in range(sim.epochs_per_band):
            # make sure this call works
            mask = data[band][epoch].bmask.array
            image = data[band][epoch].image.array

            w = np.where((mask & SAT) != 0)
            assert w[0].size > 0
            if not subtract_bright:
                assert np.all(image[w] == BAND_SAT_VALS[band])

            w = np.where(mask & BRIGHT != 0)
            assert w[0].size > 0
