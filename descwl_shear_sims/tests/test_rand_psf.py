"""
Copied from https://github.com/beckermr/metadetect-sims under BSD
"""

import numpy as np
import galsim

# import pytest
import lsst.geom as geom
import lsst.afw.image as afw_image

from ..psfs import make_rand_psf, make_dm_psf
from ._wcs import make_sim_wcs, SCALE
from ..constants import RAND_PSF_FWHM_STD, RAND_PSF_E_STD


def _get_fwhm_g1g2(psf_im):
    mom = galsim.hsm.FindAdaptiveMom(psf_im)
    return (
        mom.moments_sigma * SCALE * 2.355,
        mom.observed_shape.e1,
        mom.observed_shape.e2,
    )


def test_rand_psf_smoke():
    rng = np.random.RandomState(seed=10)
    psf = make_rand_psf(
        psf_type='gauss',
        rng=rng,
    )
    assert isinstance(psf, galsim.GSObject)


def test_rand_dmpsf_smoke():

    dim = 20
    psf_dim = 31
    rng = np.random.RandomState(seed=999)

    ntrial = 100
    fwhmvals = np.zeros(ntrial)
    e1vals = np.zeros(ntrial)
    e2vals = np.zeros(ntrial)
    for i in range(ntrial):
        masked_image = afw_image.MaskedImageF(dim, dim)
        exp = afw_image.ExposureF(masked_image)

        gspsf = make_rand_psf(
            psf_type='gauss',
            rng=rng,
        )

        galsim_wcs = make_sim_wcs(dim)
        # dm_wcs = make_dm_wcs(galsim_wcs)
        dmpsf = make_dm_psf(gspsf, psf_dim, galsim_wcs)

        exp.setPsf(dmpsf)
        psf = exp.getPsf()

        x = 8.5
        y = 10.1

        pos = geom.Point2D(x=x, y=y)

        # this one is always centered
        msim = psf.computeKernelImage(pos)
        assert msim.array.shape == (psf_dim, psf_dim)

        gsim = galsim.ImageD(msim.array, scale=SCALE)
        fwhm, e1, e2 = _get_fwhm_g1g2(gsim)

        fwhmvals[i] = fwhm
        e1vals[i] = e1
        e2vals[i] = e2

    fwhmstd = fwhmvals.std()
    e1std = e1vals.std()
    e2std = e2vals.std()
    assert fwhmstd/RAND_PSF_FWHM_STD - 1 < 0.05
    assert e1std/RAND_PSF_E_STD - 1 < 0.05
    assert e2std/RAND_PSF_E_STD - 1 < 0.05
