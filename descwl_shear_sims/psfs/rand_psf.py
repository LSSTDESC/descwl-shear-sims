from numpy import log, sqrt
import galsim
from ..constants import (
    RAND_PSF_FWHM_MEAN,
    RAND_PSF_FWHM_STD,
    RAND_PSF_FWHM_MIN,
    RAND_PSF_FWHM_MAX,
    RAND_PSF_E_STD,
    RAND_PSF_E_MAX,
    FIXED_MOFFAT_BETA,
)


def make_rand_psf(psf_type, rng):
    """
    A simple PSF object with random FWHM and shape

    Parameters
    ----------
    psf_type: string
        'gauss' or 'moffat'

    Returns
    -------
    Gaussian or Moffat
    """

    fwhm = _get_fwhm(rng)
    e1, e2 = _get_e1e2(rng)

    if psf_type == "gauss":
        psf = galsim.Gaussian(fwhm=fwhm)
    elif psf_type == "moffat":
        psf = galsim.Moffat(fwhm=fwhm, beta=FIXED_MOFFAT_BETA)
    else:
        raise ValueError("bad psf_type '%s'" % psf_type)

    psf = psf.shear(e1=e1, e2=e2)

    return psf


def _get_fwhm(rng):
    ln_mean = log(
        RAND_PSF_FWHM_MEAN**2 / sqrt(RAND_PSF_FWHM_MEAN**2 + RAND_PSF_FWHM_STD**2)
    )  # noqa
    ln_sigma = sqrt(log(1+(RAND_PSF_FWHM_STD/RAND_PSF_FWHM_MEAN)**2))

    while True:
        fwhm = rng.lognormal(
            mean=ln_mean,
            sigma=ln_sigma,
        )
        if RAND_PSF_FWHM_MIN < fwhm < RAND_PSF_FWHM_MAX:
            break

    return fwhm


def _get_e1e2(rng):
    while True:
        e1, e2 = rng.normal(scale=RAND_PSF_E_STD, size=2)
        e = sqrt(e1**2 + e2**2)
        if e < RAND_PSF_E_MAX:
            break

    return e1, e2
