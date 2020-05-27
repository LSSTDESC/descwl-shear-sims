import os
import pytest
import numpy as np

from ..se_obs import SEObs
from ..simple_sim import Sim
from ..sim_constants import ZERO_POINT


def test_simple_sim_smoke():
    sim = Sim(rng=10, gals_kws={'density': 10})
    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    assert sim.star_density == 0.0

    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            epoch_obs = data[band][epoch]
            assert isinstance(epoch_obs, SEObs)
            assert epoch_obs.noise is not None


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
@pytest.mark.parametrize('make_round', [False, True])
def test_simple_sim_wldeblend(make_round):
    """
    no way to test roundness because we currently cannot
    turn off the noise for wldeblend
    """
    gals_kws = {
        'make_round': make_round,
    }
    sim = Sim(
        rng=101,
        gals_kws=gals_kws,
        gals_type='wldeblend',
    )
    sim.gen_sim()

    if make_round:
        from ..galaxy_builder import RoundGalaxyBuilder
        assert isinstance(sim._builders['r'], RoundGalaxyBuilder)


def test_simple_sim_cap_radius_smoke():
    sim = Sim(rng=10, cap_radius=1, gals_kws={'density': 10})
    assert sim.buff == 0.0

    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            epoch_obs = data[band][epoch]
            assert isinstance(epoch_obs, SEObs)
            assert epoch_obs.noise is not None


def test_simple_sim_noise():
    sim = Sim(rng=10, gals_kws={'density': 10})
    data = sim.gen_sim()
    for band in sim.bands:
        for epoch in range(sim.epochs_per_band):
            epoch_obs = data[band][epoch]

            assert epoch_obs.noise is not None

            nvar = epoch_obs.noise.array.var()
            expected_var = 1/epoch_obs.weight.array[0, 0]
            assert abs(nvar/expected_var-1) < 0.015


def test_bad_keys():

    stars_kws = {
        'type': 'sample',
    }
    with pytest.raises(ValueError):
        _ = Sim(rng=10, stars=True, stars_kws=stars_kws)

    gals_kws = {
        'density': 10,
    }
    with pytest.raises(ValueError):
        _ = Sim(rng=10, gals_type='wldeblend', gals_kws=gals_kws)


def test_simple_sim_double_call_raises():
    sim = Sim(rng=10, gals_kws={'density': 10})
    sim.gen_sim()
    with pytest.raises(RuntimeError):
        sim.gen_sim()


def test_simple_sim_se_shape():
    """
    test we get the right shape out.
    """

    se_dim = 1024
    sim = Sim(rng=10, se_dim=se_dim, gals_kws={'density': 10})
    data = sim.gen_sim()

    for band, bdata in data.items():
        for se_obs in bdata:
            s = se_obs.image.array.shape
            assert s[0] == se_dim and s[1] == se_dim


def test_simple_sim_band_wcs():
    """
    make sure the cacheing code is consistent
    """
    sim = Sim(
        rng=10,
        epochs_per_band=1,
        gals_kws={'density': 10},
    )
    sim.gen_sim()

    for band in sim.bands:
        assert sim._get_wcs_for_band(band) == sim._band_wcs_objs[band]


def test_simple_sim_grid_smoke():
    sim = Sim(
        rng=10,
        layout_type='grid',
        layout_kws={'dim': 10}
    )
    data = sim.gen_sim()

    for band, bdata in data.items():
        for se_obs in bdata:
            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                axs.imshow(se_obs.image.array)
                assert False


def test_simple_sim_grid_only_stars_smoke():
    sim = Sim(
        rng=10,
        layout_type='grid',
        layout_kws={'dim': 10},
        gals=False,
        stars=True,
    )
    data = sim.gen_sim()

    for band, bdata in data.items():
        for se_obs in bdata:
            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                axs.imshow(se_obs.image.array)
                assert False


def test_simple_sim_grid_stars_and_gals_smoke():

    # on a grid, these are only used to determine the
    # fraction of each type
    star_density = 10
    gal_density = 10
    sim = Sim(
        rng=10,
        layout_type='grid',
        layout_kws={'dim': 10},
        gals=True,
        gals_kws={'density': gal_density},
        stars=True,
        stars_kws={'density': star_density},
        noise_per_band=0,
    )

    # density gets reset
    star_frac = star_density/(star_density + gal_density)
    assert sim.star_density == sim.mean_density*star_frac

    data = sim.gen_sim()

    for band, bdata in data.items():
        for se_obs in bdata:
            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                axs.imshow(se_obs.image.array)
                assert False


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
@pytest.mark.parametrize('gals_type', ['exp', 'wldeblend'])
def test_simple_sim_mag_zp(gals_type):
    """
    simulate a single star and make sure we get back the right
    magnitude
    """

    star_mag = 16
    sim = Sim(
        rng=10,
        epochs_per_band=1,
        bands=['r'],
        coadd_dim=33,
        buff=0,
        edge_width=2,
        gals=False,
        gals_type=gals_type,
        layout_type='grid',
        layout_kws={'dim': 1},
        stars=True,
        stars_type='fixed',
        stars_kws={'mag': star_mag},
    )

    data = sim.gen_sim()

    for band, bdata in data.items():
        for se_obs in bdata:
            im = se_obs.image.array
            mag = ZERO_POINT - 2.5*np.log10(im.sum())
            assert abs(mag-star_mag) < 0.1
