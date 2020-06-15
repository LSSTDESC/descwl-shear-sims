import pytest
import numpy as np
from ..trivial_sim import make_trivial_sim


@pytest.mark.parametrize('dither,rotate', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_trivial_sim_smoke(dither, rotate):

    seed = 74321
    rng = np.random.RandomState(seed)
    _ = make_trivial_sim(
        rng=rng,
        noise=0.001,
        coadd_dim=351,
        buff=50,
        layout='grid',
        g1=0.02,
        g2=0.00,
        dither=dither,
        rotate=rotate,
    )


def test_trivial_sim():

    bands = ["i"]
    seed = 7421
    coadd_dim = 201
    psf_dim = 47
    rng = np.random.RandomState(seed)
    sim_data = make_trivial_sim(
        rng=rng,
        noise=0.001,
        coadd_dim=coadd_dim,
        buff=3,
        psf_dim=psf_dim,
        layout="grid",
        g1=0.02,
        g2=0.00,
        bands=bands,
    )

    assert 'coadd_dims' in sim_data
    assert sim_data['coadd_dims'] == [coadd_dim]*2
    assert 'psf_dims' in sim_data
    assert sim_data['psf_dims'] == [psf_dim]*2

    band_data = sim_data['band_data']
    for band in bands:
        assert band in band_data


@pytest.mark.parametrize('epochs_per_band', [1, 2, 3])
def test_trivial_sim_epochs(epochs_per_band):

    bands = ['r', 'i', 'z']
    seed = 7421
    coadd_dim = 301
    psf_dim = 47
    rng = np.random.RandomState(seed)
    sim_data = make_trivial_sim(
        rng=rng,
        noise=0.001,
        coadd_dim=coadd_dim,
        buff=10,
        psf_dim=psf_dim,
        layout='grid',
        g1=0.02,
        g2=0.00,
        dither=True,
        rotate=True,
        bands=bands,
        epochs_per_band=epochs_per_band,
    )

    band_data = sim_data['band_data']
    for band in bands:
        assert band in band_data
        assert len(band_data[band]) == epochs_per_band


@pytest.mark.parametrize("layout", ("grid", "random"))
def test_trivial_sim_layout(layout):

    seed = 812
    psf_dim = 47
    rng = np.random.RandomState(seed)
    _ = make_trivial_sim(
        rng=rng,
        noise=0.001,
        coadd_dim=351,
        buff=10,
        psf_dim=psf_dim,
        layout=layout,
        g1=0.02,
        g2=0.00,
    )
