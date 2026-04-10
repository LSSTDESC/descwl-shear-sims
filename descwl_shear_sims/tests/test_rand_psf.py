"""
Copied from https://github.com/beckermr/metadetect-sims under BSD
"""

import numpy as np
import galsim
import pytest

# import pytest
import lsst.geom as geom
import lsst.afw.image as afw_image

from ..psfs import make_rand_psf, make_dm_psf
from ._wcs import make_sim_wcs, SCALE
from ..constants import RAND_PSF_FWHM_MEAN, RAND_PSF_FWHM_STD, RAND_PSF_E_STD


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
        psf_fwhm_mean=0.85,
        psf_fwhm_std=0.12,
        psf_fwhm_min=0.6,
        psf_fwhm_max=1.5,
        psf_e_std=0.03,
        psf_e_max=0.2,
        rng=rng,
    )
    assert isinstance(psf, galsim.GSObject)


@pytest.mark.parametrize('defaults', [True, False])
def test_rand_dmpsf_smoke(defaults):

    if defaults:
        fwhm_mean = RAND_PSF_FWHM_MEAN
        fwhm_std = RAND_PSF_FWHM_STD
        e_std = RAND_PSF_E_STD
    else:
        fwhm_mean = 0.85
        fwhm_std = 0.12
        e_std = 0.03

    dim = 20
    psf_dim = 31
    rng = np.random.RandomState(seed=999)

    ntrial = 500
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
        gspsf = make_rand_psf(
            psf_type='gauss',
            psf_fwhm_mean=fwhm_mean,
            psf_fwhm_std=fwhm_std,
            psf_e_std=e_std,
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

    afwhm_mean = fwhmvals.mean()
    afwhm_std = fwhmvals.std()

    ae1std = e1vals.std()
    ae2std = e2vals.std()

    fwhm_fracdiff = afwhm_mean / fwhm_mean - 1
    fwhm_std_fracdiff = afwhm_std / fwhm_std - 1
    e1std_fracdiff = ae1std / e_std - 1
    e2std_fracdiff = ae2std / e_std - 1

    assert fwhm_fracdiff < 0.05
    assert fwhm_std_fracdiff < 0.05
    assert e1std_fracdiff < 0.05
    assert e2std_fracdiff < 0.05
