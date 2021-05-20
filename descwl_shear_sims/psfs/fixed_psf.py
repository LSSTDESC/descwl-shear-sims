import copy
import galsim
from ..constants import FIXED_PSF_FWHM, FIXED_MOFFAT_BETA


def make_fixed_psf(*, psf_type):
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
    if psf_type == "gauss":
        psf = galsim.Gaussian(fwhm=FIXED_PSF_FWHM)
    elif psf_type == "moffat":
        psf = galsim.Moffat(fwhm=FIXED_PSF_FWHM, beta=FIXED_MOFFAT_BETA)
    else:
        raise ValueError("bad psf_type '%s'" % psf_type)

    return psf


class FixedPSF(object):
    """
    A simple fixed PSF object

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
    def __init__(self, *, psf, offset, psf_dim, wcs):
        self._psf = psf
        self._offset = offset
        self._psf_dim = psf_dim
        self._wcs = wcs

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
