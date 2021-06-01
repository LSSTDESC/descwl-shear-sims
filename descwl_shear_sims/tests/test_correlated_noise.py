import pytest
import numpy as np
from ..sim import make_sim
from ..psfs import make_fixed_psf
from ..galaxies import make_galaxy_catalog
from numba import njit


@njit
def get_cov(image):

    cov = np.zeros((3, 3))

    nrow, ncol = image.shape
    for row in range(1, nrow-1):
        for col in range(1, ncol-1):

            cenval = image[row, col]
            for row_offset in range(-1, 1+1):
                for col_offset in range(-1, 1+1):
                    val = image[row+row_offset, col+col_offset]

                    cov[row_offset+1, col_offset+1] += cenval*val
    npix = (nrow - 2)*(ncol - 2)
    cov *= 1.0/npix
    return cov


def test_correlated_noise():
    """
    use mag 37 galaxies so the image is purely noise,
    and compare statistics with the coadded noise image
    """
    coadd_mod = pytest.importorskip('descwl_coadd.coadd')

    rng0 = np.random.RandomState(97756)
    ntrial = 100
    gal_config = {'mag': 37, 'hlr': 0.5}

    for i in range(ntrial):
        seed = rng0.randint(0, 2**30)
        rng = np.random.RandomState(seed)

        coadd_dim = 301
        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type='exp',
            coadd_dim=coadd_dim,
            buff=0,
            layout="grid",
            gal_config=gal_config,
        )

        psf = make_fixed_psf(psf_type="gauss")
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            g1=0.02,
            g2=0.00,
            psf=psf,
            dither=True,
            rotate=True,
        )

        mbc = coadd_mod.MultiBandCoadds(
            data=sim_data['band_data'],
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            byband=False,
            loglevel="debug",
        )

        coadd_obs = mbc.coadds['all']
        image = coadd_obs.image
        noise = coadd_obs.noise

        icov = get_cov(image)
        ncov = get_cov(noise)

        if i == 0:
            icov_sum = icov
            ncov_sum = ncov
        else:
            icov_sum += icov
            ncov_sum += ncov

    icov = icov_sum/ntrial
    ncov = ncov_sum/ntrial
    fracdiff = ncov/icov - 1

    assert np.all(np.abs(fracdiff) < 0.01)
