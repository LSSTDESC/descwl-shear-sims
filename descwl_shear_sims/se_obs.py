"""A container for SE observations."""
import galsim


class SEObs(object):
    """A simple container for an SE image and associated data products.

    This class provides error checking on the various attributes. It can
    also be extended to convert between representations in packages like
    ngmix and the DM stack.

    Parameters
    ----------
    image : galsim.Image
        The SE image.
    weight : galsim.Image
        The SE image weight map.
    wcs : galsim.BaseWCS or children
        The WCS mapping for the SE image.
    psf_function : callable
        A callable that returns a representation of the
        PSF at a given image pixel location as a `galsim.Image`. Its signature
        should be

            def psf_function(*, x, y, center_psf):
                pass

        where `(x, y)` are the column and row image positions at which to
        draw the PSF. The `center_psf` keyword, if `True`, should draw the PSF
        on a pixel center at the center of an odd image. Otherwise the PSF should
        have the same subpixel offset as implied by the position (i.e.,
        `x-int(x+0.5)`, `y-int(y+0.5)`).
    noise : galsim.Image, optional
        A noise field associated with the image.
    bmask : galsim.Image, optional
        An optional bit mask associated with the image.
    ormask : galsim.Image, optional
        An optional "or" bit mask associated with the image. This can be used
        to store extra mask bits computed by taking the logical "or" from any
        input data products.

    Attributes
    ----------
    image
    weight
    wcs
    noise
    bmask
    ormask

    Methods
    -------
    get_psf(x, y, center_psf=False)
        Draw the PSF at the given image location.
    """
    def __init__(
            self, *, image, weight, wcs, psf_function,
            noise=None, bmask=None, ormask=None):
        # we use the setters here to get error checking on types in init
        self.image = image
        self.weight = weight
        self.wcs = wcs
        self.noise = noise
        self.bmask = bmask
        self.ormask = ormask
        self._psf_function = psf_function

    @property
    def image(self):
        """The SE image as a `galsim.Image`"""
        return self._image

    @image.setter
    def image(self, image):
        """Set the image to given a `galsim.Image`"""
        if not isinstance(image, galsim.Image):
            raise ValueError("The image must be a `galsim.Image` or subclass!")
        self._image = image

    @property
    def weight(self):
        """The SE weight map as a `galsim.Image`"""
        return self._weight

    @weight.setter
    def weight(self, weight):
        """Set the weight map to given a `galsim.Image`"""
        if not isinstance(weight, galsim.Image):
            raise ValueError("The weight must be a `galsim.Image` or subclass!")
        self._weight = weight

    @property
    def wcs(self):
        """The SE WCS as a `galsim.BaseWCS`"""
        return self._wcs

    @wcs.setter
    def wcs(self, wcs):
        """Set the WCS to given a `galsim.BaseWCS`"""
        if not isinstance(wcs, galsim.BaseWCS):
            raise ValueError("The WCS must be a `galsim.BaseWCS` or subclass!")
        self._wcs = wcs

    @property
    def noise(self):
        """The SE noise field as a `galsim.Image`"""
        return self._noise

    @noise.setter
    def noise(self, noise):
        """Set the noise field to given a `galsim.Image`"""
        if noise is not None and not isinstance(noise, galsim.Image):
            raise ValueError("The noise field must be a `galsim.Image` or subclass!")
        self._noise = noise

    @property
    def bmask(self):
        """The SE bit mask as a `galsim.Image`"""
        return self._bmask

    @bmask.setter
    def bmask(self, bmask):
        """Set the bit mask to given a `galsim.Image`"""
        if bmask is not None and not isinstance(bmask, galsim.Image):
            raise ValueError("The bit mask must be a `galsim.Image` or subclass!")
        self._bmask = bmask

    @property
    def ormask(self):
        """The "or" mask as a `galsim.Image`"""
        return self._ormask

    @ormask.setter
    def ormask(self, ormask):
        """Set the "or" mask to given a `galsim.Image`"""
        if ormask is not None and not isinstance(ormask, galsim.Image):
            raise ValueError("The \"or\" mask must be a `galsim.Image` or subclass!")
        self._ormask = ormask

    def get_psf(self, x, y, center_psf=False, get_offset=False):
        """Draw the PSF at the given image location.

        Parameters
        ----------
        x : float
            The column/x position at which to draw the PSF.
        y : float
            The row/y position at which to draw the PSF.
        center_psf : bool, optional
            If True, the PSF model at (x, y) will be drawn with its center
            on the central pixel of the image. Otherwise, the PSF is drawn with
            the center having the same subpixel offset as implied by the input
            position (i.e., `x-int(x+0.5)` and `y-int(y+0.5)`).
        get_offse: bool
            If True, return the offset used when drawing the psf.

        Returns
        -------
        psf : galsim.Image
            An image of the PSF (including the image pixel).
        """
        return self._psf_function(
            x=x,
            y=y,
            center_psf=center_psf,
            get_offset=get_offset,
        )
