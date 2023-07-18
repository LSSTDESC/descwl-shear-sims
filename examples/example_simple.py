"""
simple example with a grid of exponential galaxies

The images are dithered and rotated
"""
import numpy as np

import lsst.afw.image as afw_image
import lsst.afw.geom as afw_geom
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.sim import make_sim


def go():
    seed = 74321
    rng = np.random.RandomState(seed)

    coadd_dim = 351
    psf_dim = 51
    bands = ['r', 'i']

    # this makes a grid of fixed exponential galaxies
    # with default properties. One exposure per band

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type='fixed',
        coadd_dim=coadd_dim,
        buff=30,
        layout='grid',
    )

    # gaussian psf
    psf = make_fixed_psf(psf_type='gauss')

    # generate simulated data, see below for whats in this dict
    data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        bands=bands,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=True,
        rotate=True,
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

    # we should have no bright objects
    assert data['bright_info'].size == 0


if __name__ == '__main__':
    go()
