import os
import numpy as np
import galsim
import pytest

from descwl_shear_sims.lsst_bits import get_flagval
from descwl_shear_sims.saturation import BAND_SAT_VALS
from descwl_shear_sims.artifacts.star_bleeds import add_bleed
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.sim import make_sim


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
@pytest.mark.parametrize('band', ('r', 'i', 'z'))
def test_bleed(band):
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

    assert bmask[cen[0], cen[1]] == get_flagval('SAT')
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

    some_were_bright = False
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

        nbright = len(sim_data['bright_info'])
        if nbright > 0:
            some_were_bright = True

        for bi in sim_data['bright_info']:

            assert 'world_pos' in bi
            assert isinstance(bi['world_pos'], galsim.CelestialCoord)

            assert 'radius_pixels' in bi
            assert bi['radius_pixels'] >= 0

        exp = sim_data['band_data'][bands[0]][0]

        mask = exp.mask.array
        image = exp.image.array

        wsat = np.where((mask & get_flagval('SAT')) != 0)
        if (
            wsat[0].size > 0 and
            np.all(image[wsat] == BAND_SAT_VALS['r'])
        ):

            some_were_saturated = True
            break

    assert some_were_bright and some_were_saturated


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


if __name__ == '__main__':
    test_star_mask_in_sim()
