import os
import pytest
import numpy as np
from ..trivial_sim import (
    make_trivial_sim,
    make_galaxy_catalog,
    StarCatalog,
    make_psf,
    make_ps_psf,
    get_se_dim,
)


@pytest.mark.parametrize('dither,rotate', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_trivial_sim_smoke(dither, rotate):

    seed = 74321
    rng = np.random.RandomState(seed)

    coadd_dim = 341
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=30,
        layout="grid",
    )

    psf = make_psf(psf_type="gauss")
    _ = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=351,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=dither,
        rotate=rotate,
    )


def test_trivial_sim():

    bands = ["i"]
    seed = 7421
    coadd_dim = 201
    psf_dim = 47
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=30,
        layout="grid",
    )

    psf = make_psf(psf_type="moffat")
    sim_data = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        bands=bands,
    )

    assert 'coadd_dims' in sim_data
    assert sim_data['coadd_dims'] == [coadd_dim]*2
    assert 'psf_dims' in sim_data
    assert sim_data['psf_dims'] == [psf_dim]*2

    band_data = sim_data['band_data']
    for band in bands:
        assert band in band_data


@pytest.mark.parametrize("psf_type", ["gauss", "moffat", "ps"])
def test_trivial_sim_psf_type(psf_type):

    seed = 431
    rng = np.random.RandomState(seed)

    coadd_dim = 101
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=5,
        layout="grid",
    )

    if psf_type == "ps":
        se_dim = get_se_dim(coadd_dim=coadd_dim)
        psf = make_ps_psf(rng=rng, dim=se_dim)
    else:
        psf = make_psf(psf_type=psf_type)

    _ = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=True,
        rotate=True,
    )


@pytest.mark.parametrize('epochs_per_band', [1, 2, 3])
def test_trivial_sim_epochs(epochs_per_band):

    seed = 7421
    bands = ["r", "i", "z"]
    coadd_dim = 301
    psf_dim = 47

    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=10,
        layout="grid",
    )

    psf = make_psf(psf_type="gauss")
    sim_data = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        bands=bands,
        epochs_per_band=epochs_per_band,
    )

    band_data = sim_data['band_data']
    for band in bands:
        assert band in band_data
        assert len(band_data[band]) == epochs_per_band


@pytest.mark.parametrize("layout", ("grid", "random"))
def test_trivial_sim_layout(layout):
    seed = 7421
    coadd_dim = 201
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=30,
        layout=layout,
    )

    psf = make_psf(psf_type="gauss")
    _ = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )


@pytest.mark.parametrize(
    "cosmic_rays, bad_columns",
    [(True, True),
     (True, False),
     (False, True),
     (True, True)],
)
def test_trivial_sim_defects(cosmic_rays, bad_columns):
    seed = 7421
    coadd_dim = 201
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        layout="grid",
        buff=30,
    )

    psf = make_psf(psf_type="gauss")
    _ = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        cosmic_rays=cosmic_rays,
        bad_columns=bad_columns,
    )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_trivial_sim_wldeblend():
    seed = 7421
    coadd_dim = 201
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="wldeblend",
        coadd_dim=coadd_dim,
        buff=30,
    )

    psf = make_psf(psf_type="moffat")
    _ = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_trivial_sim_stars():
    seed = 7421
    coadd_dim = 201
    buff = 30
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="wldeblend",
        coadd_dim=coadd_dim,
        buff=buff,
    )

    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
    )

    psf = make_psf(psf_type="moffat")
    _ = make_trivial_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )
