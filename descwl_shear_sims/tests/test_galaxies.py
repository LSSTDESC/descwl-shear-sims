import pytest
import numpy as np
from ..galaxies import (
    WLDeblendGalaxyCatalog,
    make_galaxy_catalog,
    FixedGalaxyCatalog,
    GalaxyCatalog,
    FixedPairGalaxyCatalog,
    PairGalaxyCatalog,
)
from ..psfs import make_fixed_psf
from ..sim import make_sim
from ..shear import ShearConstant

shear_obj = ShearConstant(g1=0.02, g2=0.)


@pytest.mark.parametrize('layout', ('pair', 'random', 'hex'))
@pytest.mark.parametrize('gal_type', ('fixed', 'varying'))
@pytest.mark.parametrize('morph', ('exp', 'dev', 'bd', 'bdk'))
def test_galaxies_smoke(layout, gal_type, morph):
    """
    test sim can run and is repeatable.  This is relevant as we now support
    varying galaxies
    """
    seed = 74321

    for trial in (1, 2):
        rng = np.random.RandomState(seed)

        sep = 4.0  # arcseconds

        coadd_dim = 100
        buff = 10
        bands = ['i']

        gal_config = {
            'hlr': 1.0,
            'mag': 22,
            'morph': morph,
        }

        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            gal_type=gal_type,
            gal_config=gal_config,
            layout=layout,
            sep=sep,
        )
        if layout == 'pair':
            if gal_type == 'fixed':
                assert isinstance(galaxy_catalog, FixedPairGalaxyCatalog)
            elif gal_type == 'varying':
                assert isinstance(galaxy_catalog, PairGalaxyCatalog)
        else:
            if gal_type == 'fixed':
                assert isinstance(galaxy_catalog, FixedGalaxyCatalog)
            elif gal_type == 'varying':
                assert isinstance(galaxy_catalog, GalaxyCatalog)

        psf = make_fixed_psf(psf_type='gauss')
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            bands=bands,
            shear_obj=shear_obj,
            psf=psf,
        )

        if trial == 1:
            image = sim_data['band_data']['i'][0].image.array
        else:
            new_image = sim_data['band_data']['i'][0].image.array

            assert np.all(image == new_image)


def test_wlgalaxies_selection():
    seed = 74321
    rng = np.random.RandomState(seed)
    coadd_dim = 100
    buff = 10

    for _ in ["g_ab", "r_ab", "i_ab"]:
        galaxy_catalog = WLDeblendGalaxyCatalog(
            pixel_scale=0.2,
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
            select_observable=_,
            select_upper_limit=27,
            select_lower_limit=25,
        )
        assert np.min(galaxy_catalog._wldeblend_cat[_]) >= 25.0
        assert np.max(galaxy_catalog._wldeblend_cat[_]) <= 27.0
    galaxy_catalog = WLDeblendGalaxyCatalog(
        pixel_scale=0.2,
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout="random",
        select_observable=["r_ab", "z_ab"],
        select_upper_limit=[27, 26],
        select_lower_limit=[25, 22],
    )
    assert np.min(galaxy_catalog._wldeblend_cat["r_ab"]) >= 25.0
    assert np.max(galaxy_catalog._wldeblend_cat["r_ab"]) <= 27.0
    assert np.min(galaxy_catalog._wldeblend_cat["z_ab"]) >= 22.0
    assert np.max(galaxy_catalog._wldeblend_cat["z_ab"]) <= 26.0
    return

