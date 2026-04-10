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


def make_rand_psf(
    psf_type,
    rng,
    psf_fwhm_mean=RAND_PSF_FWHM_MEAN,
    psf_fwhm_std=RAND_PSF_FWHM_STD,
    psf_fwhm_min=RAND_PSF_FWHM_MIN,
    psf_fwhm_max=RAND_PSF_FWHM_MAX,
    psf_e_std=RAND_PSF_E_STD,
    psf_e_max=RAND_PSF_E_MAX,
):
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

    fwhm = _get_fwhm(
        rng,
        fwhm_mean=psf_fwhm_mean,
        fwhm_std=psf_fwhm_std,
        fwhm_min=psf_fwhm_min,
        fwhm_max=psf_fwhm_max,
    )
    e1, e2 = _get_e1e2(
        rng,
        e_std=psf_e_std,
        e_max=psf_e_max,
    )

    if psf_type == "gauss":
        psf = galsim.Gaussian(fwhm=fwhm)
    elif psf_type == "moffat":
        psf = galsim.Moffat(fwhm=fwhm, beta=FIXED_MOFFAT_BETA)
    else:
        raise ValueError("bad psf_type '%s'" % psf_type)

    psf = psf.shear(e1=e1, e2=e2)

    return psf


def _get_fwhm(rng, fwhm_mean, fwhm_std, fwhm_min, fwhm_max):
    ln_mean = log(
        fwhm_mean**2 / sqrt(fwhm_mean**2 + fwhm_std**2)
    )  # noqa
    ln_sigma = sqrt(log(1+(fwhm_std/fwhm_mean)**2))

    while True:
        fwhm = rng.lognormal(
            mean=ln_mean,
            sigma=ln_sigma,
        )
        if fwhm_min < fwhm < fwhm_max:
            break

    return fwhm


def _get_e1e2(rng, e_std, e_max):
    while True:
        e1, e2 = rng.normal(scale=e_std, size=2)
        e = sqrt(e1**2 + e2**2)
        if e < e_max:
            break

    return e1, e2
