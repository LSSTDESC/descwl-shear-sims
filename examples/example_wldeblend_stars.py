"""
example with WeakLensingDeblending galaxies and stars and power spectrum psf

bleed trails are turned on for bright stars
"""
import os
import numpy as np

import lsst.afw.image as afw_image
import lsst.afw.geom as afw_geom
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.stars import make_star_catalog
from descwl_shear_sims.psfs import make_ps_psf
from descwl_shear_sims.sim import make_sim, get_se_dim


def go():
    if 'CATSIM_DIR' not in os.environ:
        # this contains the galaxy and star catalogs for generatig
        # WeakLensingDeblending galaxies and stars
        print('you need CATSIM_DIR defined to run this example')

    seed = 761
    rng = np.random.RandomState(seed)

    dither = True
    rotate = True
    coadd_dim = 351
    psf_dim = 51
    bands = ['r', 'i']
    buff = 30

    # this makes WeakLensingDeblending galaxies

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type='wldeblend',
        coadd_dim=coadd_dim,
        buff=buff,
        layout='random',
    )

    # stars with the high density so we get some
    # bright ones
    star_config = {'density': 100}
    star_catalog = make_star_catalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        star_config=star_config,
    )

    # power spectrum psf
    se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
    psf = make_ps_psf(rng=rng, dim=se_dim)

    # generate simulated data, see below for whats in this dict
    data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        bands=bands,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=dither,
        rotate=rotate,
        star_bleeds=True,
    )

    # data is a dict with the following keys.
    # band_data: a dict, keyed by band name, with values that are a list of
    #   exps
    # coadd_wcs: is a DM wcs for use in coadding
    # psf_dims: is the psf dim we sent in (psf_dim, psf_dim)
    # coadd_dims: shape of the coadd image (dim, dim)
    # coadd_bbox: is an lsst Box2I, for use in coadding
    # bright_info: is a structured array with position and mask info for bright
    #   objects
    # se_wcs: list of WCS for each single epoch image

    for key in ['band_data', 'coadd_wcs', 'psf_dims', 'coadd_bbox', 'bright_info']:
        assert key in data

    for band in bands:
        assert band in data['band_data']
        assert isinstance(data['band_data'][band][0], afw_image.ExposureF)

    assert isinstance(data['coadd_wcs'], afw_geom.SkyWcs)

    assert data['psf_dims'] == (psf_dim, )*2

    extent = data['coadd_bbox'].getDimensions()
    edims = (extent.getX(), extent.getY())
    assert edims == (coadd_dim, )*2

    # we might have bright objects, depends on how the star
    # sampling went
    bright_info = data['bright_info']
    if bright_info.size != 0:
        print('got bright', bright_info.size, 'bright objects')
        for name in ['ra', 'dec', 'radius_pixels']:
            assert name in bright_info.dtype.names


if __name__ == '__main__':
    go()
