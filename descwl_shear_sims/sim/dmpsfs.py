import galsim
import lsst.afw.image as afw_image
import lsst.geom as geom
from lsst.meas.algorithms import ImagePsf
from .psfs import PowerSpectrumPSF


def make_dm_psf(psf, psf_dim, wcs):
    """
    convert a sim PSF to a DM Stack PSF

    Parameters
    ----------
    psf: GSObject or PowerSpectrumPSF
        The sim psf
    psf_dim: int
        Dimension of the psfs to draw, must be odd
    wcs: galsim WCS
        WCS for drawing

    Returns
    -------
    Either a FixedDMPSF or a PowerSpectrumDMPSF
    """
    if isinstance(psf, galsim.GSObject):
        return FixedDMPSF(psf, psf_dim, wcs)
    elif isinstance(psf, PowerSpectrumPSF):
        return PowerSpectrumDMPSF(psf, psf_dim, wcs)
    else:
        raise ValueError('bad psf: %s' % type(psf))


class FixedDMPSF(ImagePsf):
    """
    A class representing a fixed galsim GSObject as the psf

    When offsetting no image interpolation is done.  Real psfs have an
    interpolation to offset (different from interpolating coefficients)
    """
    def __init__(self, gspsf, psf_dim, wcs):
        """
        Parameters
        ----------
        gspsf: GSObject
            A galsim GSObject representing the psf
        psf_dim: int
            Dimension of the psfs to draw, must be odd
        wcs: galsim WCS
            WCS for drawing
        """
        ImagePsf.__init__(self)

        if psf_dim % 2 == 0:
            raise ValueError('psf dims must be odd, got %s' % psf_dim)

        self._psf_dim = psf_dim
        self._wcs = wcs
        self._gspsf = gspsf

    def computeImage(self, image_pos):  # noqa
        """
        compute an image at the specified image position, centered in the
        postage stamp with appropriate offset

        Parameters
        ----------
        pos: geom.Point2D
            A point in the original image at which evaluate the kernel
        """

        x = image_pos.getX()
        y = image_pos.getY()

        offset_x = x - int(x + 0.5)
        offset_y = y - int(y + 0.5)

        offset = (offset_x, offset_y)

        return self._make_image(image_pos, offset=offset)

    def computeKernelImage(self, image_pos):  # noqa
        """
        compute a centered kernel image appropriate for convolution

        Parameters
        ----------
        pos: geom.Point2D
            A point in the original image at which evaluate the kernel
        """

        return self._make_image(image_pos)

    def _get_gspsf(self, image_pos):
        """
        Get the GSObject representing the PSF

        Parameters
        ----------
        pos: galsim Position
            The position at which to evaluate the psf.  This is a fixed
            psf so the position is ignored
        """
        return self._gspsf

    def _make_image(self, image_pos, offset=None):
        """
        make the image, including a possible offset

        Parameters
        ----------
        image_pos: geom.Point2D
            A point in the original image at which evaluate the kernel
        offset: tuple, optional
            The (x, y) offset, default None
        """
        dim = self._psf_dim

        x = image_pos.getX()
        y = image_pos.getY()

        gs_pos = galsim.PositionD(x=x, y=y)
        gspsf = self._get_gspsf(gs_pos)

        gsimage = gspsf.drawImage(
            nx=dim,
            ny=dim,
            offset=offset,
            wcs=self._wcs.local(image_pos=gs_pos),
        )

        origin = -(dim//2)
        dims = (dim, )*2
        bbox = geom.Box2I(geom.Point2I(origin), geom.Extent2I(dims))

        image = gsimage.array.astype('f8')
        aimage = afw_image.ImageD(bbox)
        aimage.array[:, :] = image
        return aimage


class PowerSpectrumDMPSF(FixedDMPSF):
    """
    A class representing a power spectrum psf

    When offsetting no image interpolation is done.  Real psfs have an
    interpolation to offset (different from interpolating coefficients)
    """
    def __init__(self, pspsf, psf_dim, wcs):
        """
        Parameters
        ----------
        pspsf: PowerSpectrumPSF
            The power spectrum psf
        psf_dim: int
            Dimension of the psfs to draw, must be odd
        wcs: galsim WCS
            WCS for drawing
        """
        ImagePsf.__init__(self)

        if psf_dim % 2 == 0:
            raise ValueError('psf dims must be odd, got %s' % psf_dim)

        self._psf_dim = psf_dim
        self._wcs = wcs
        self._pspsf = pspsf

    def _get_gspsf(self, pos):
        """
        Get the GSObject representing the PSF at the specified
        location

        Parameters
        ----------
        pos: galsim Position
            The position at which to evaluate the psf.
        """

        return self._pspsf.getPSF(pos)
