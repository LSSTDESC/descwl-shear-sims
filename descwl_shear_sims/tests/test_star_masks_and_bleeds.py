import os
import numpy as np
import galsim
import pytest

from ..lsst_bits import get_flagval
from ..saturation import BAND_SAT_VALS
from ..masking import add_bright_star_mask
from ..artifacts.star_bleeds import add_bleed
from ..stars import StarCatalog
from ..galaxies import make_galaxy_catalog
from ..psfs import make_fixed_psf
from ..sim import make_sim


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
@pytest.mark.parametrize('band', ('r', 'i', 'z'))
def test_star_mask_and_bleed(band):
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
        bmask=bmask,
        x=pos.x,
        y=pos.y,
        radius=10,
        val=get_flagval('BRIGHT'),
    )

    assert bmask[cen[0], cen[1]] == get_flagval('SAT') | get_flagval('BRIGHT')
    assert image[cen[0], cen[1]] == BAND_SAT_VALS[band]


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_star_mask_in_sim():
    """
    test star masking using the keyword to the sim
    """
    rng = np.random.RandomState(234)

    bands = ['r']
    coadd_dim = 100
    buff = 0
    star_density = 100
    psf = make_fixed_psf(psf_type='moffat')

    some_were_saturated = False
    for i in range(1000):
        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type='fixed',
            coadd_dim=coadd_dim,
            buff=buff,
            layout='random',
        )
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=star_density,
        )
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=coadd_dim,
            bands=bands,
            psf=psf,
            g1=0, g2=0,
            star_bleeds=True,
        )

        exp = sim_data['band_data'][bands[0]][0]

        # import lsst.afw.display as afw_display
        # display = afw_display.getDisplay(backend='ds9')
        # display.mtv(exp)
        # display.scale('log', 'minmax')

        mask = exp.mask.array
        image = exp.image.array

        wsat = np.where((mask & get_flagval('SAT')) != 0)
        wbright = np.where(mask & get_flagval('BRIGHT') != 0)
        if (wsat[0].size > 0 and
                np.all(image[wsat] == BAND_SAT_VALS['r']) and
                wbright[0].size > 0):

            some_were_saturated = True
            break

    assert some_were_saturated


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_star_mask_in_sim_repeatable():
    """
    test star masking using the keyword to the sim
    """

    seed = 234

    bands = ['r']
    coadd_dim = 100
    buff = 0
    star_density = 100
    psf = make_fixed_psf(psf_type='moffat')

    for step in (0, 1):
        rng = np.random.RandomState(seed)
        for i in range(1000):
            galaxy_catalog = make_galaxy_catalog(
                rng=rng,
                gal_type='fixed',
                coadd_dim=coadd_dim,
                buff=buff,
                layout='random',
            )
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=coadd_dim,
                buff=buff,
                density=star_density,
            )
            sim_data = make_sim(
                rng=rng,
                galaxy_catalog=galaxy_catalog,
                star_catalog=star_catalog,
                coadd_dim=coadd_dim,
                bands=bands,
                psf=psf,
                g1=0, g2=0,
                star_bleeds=True,
            )

            exp = sim_data['band_data'][bands[0]][0]

            mask = exp.mask.array
            wsat = np.where((mask & get_flagval('SAT')) != 0)

            if wsat[0].size > 0:

                if step == 0:
                    nmarked = wsat[0].size
                    break
                else:
                    assert wsat[0].size == nmarked
                    break
