import copy
import numpy as np
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


class RandPSF(object):
    """
    A simple PSF object with random FWHM and shape

    Parameters
    ----------
    psf: galsim.GSObject
        The psf object
    offset: galsim.PositionD
        Should match the offset of the image thei psf corresponds to
    psf_dim: int
        The dimension of the PSF image that will be created
    wcs: galsim WCS
        E.g. a wcs returned by make_wcs
    """
    def __init__(self, *, psf_type, offset, psf_dim, wcs, rng):
        from numpy import log, sqrt
        self._psf_type = psf_type
        self._offset = offset
        self._psf_dim = psf_dim
        self._wcs = wcs
        self._rng = rng

        self._ln_mean = log(
            RAND_PSF_FWHM_MEAN**2 / sqrt(RAND_PSF_FWHM_MEAN**2 + RAND_PSF_FWHM_STD**2)
        )  # noqa
        self._ln_sigma = sqrt(log(1+(RAND_PSF_FWHM_STD/RAND_PSF_FWHM_MEAN)**2))

    def __call__(self, *, x, y, center_psf, get_offset=False):
        """
        center_psf is ignored ,just there for compatibility

        Parameters
        ----------
        x: float
            x image position
        y: float
            y image position
        cener_psf: bool
            Center the psf, this is ignored
        get_offset: bool, optional
            If True, return the offset

        Returns
        -------
        A galsim Image, and optionally the offset as a PositonD
        """
        image_pos = galsim.PositionD(x=x, y=y)

        offset = copy.deepcopy(self._offset)

        if center_psf:
            print("ignoring request to center psf, using internal offset")

        gsimage = self._psf.drawImage(
            nx=self._psf_dim,
            ny=self._psf_dim,
            offset=offset,
            wcs=self._wcs.local(image_pos=image_pos),
        )
        if get_offset:
            if offset is None:
                offset = galsim.PositionD(x=0.0, y=0.0)
            return gsimage, offset
        else:
            return gsimage

    def _get_psf_object(self, psf_type):
        """
        Make a fixed PSF

        Parameters
        ----------
        psf_type: string
            'gauss' or 'moffat'

        Returns
        -------
        Gaussian or Moffat
        """

        fwhm = self.get_fwhm()
        e1, e2 = self.get_e1e2()

        if self._psf_type == "gauss":
            psf = galsim.Gaussian(fwhm=fwhm)
        elif self._psf_type == "moffat":
            psf = galsim.Moffat(fwhm=fwhm, beta=FIXED_MOFFAT_BETA)
        else:
            raise ValueError("bad psf_type '%s'" % psf_type)

        psf = psf.shear(e1=e1, e2=e2)

        return psf

    def get_fwhm(self):
        while True:
            fwhm = self._rng.lognormal(
                mean=self._ln_mean,
                sigma=self._ln_sigma,
            )
            if RAND_PSF_FWHM_MIN < fwhm < RAND_PSF_FWHM_MAX:
                break
        return fwhm

    def get_e1e2(self):
        while True:
            e1, e2 = self._rng.normal(scale=RAND_PSF_E_STD, size=2)
            e = np.sqrt(e1**2 + e2**2)
            if e < RAND_PSF_E_MAX:
                break
        return e1, e2
