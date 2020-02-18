import numpy as np
from numba import njit
import galsim
import ngmix
from .randsphere import randcap

from .lsst_bits import STAR, BLEED


class StarMasks(object):
    """
    make fake masks around the given position with

    - star mask radii distributed as a clipped log normal
    - bleed trail length a fixed factor larger than the diameter
      of the star mask
    - bleed widths either 1 or 2


    Parameters
    ----------
    rng: numpy.RandomState
        Random number generator to use
    center_ra: float
        Stars will be generated in a disk around this point
    center_dec: float
        Stars will be generated in a disk around this point
    radius_degrees: float
        Stars will be in a disk with this radius
    density: float
        Number of saturated stars per square degree
    pixel_scale: float
        pixel scale in arcsec/pixel
    radmean: float
        Mean star mask radius in pixels for log normal distribution
    radstd: float
        Radius standard deviation in pixels for log normal distribution
    radmin: float
        Minimum radius in pixels of star masks.  The log normal values
        will be clipped to more than this value
    radmax: float
        Maximum radius in pixels of star masks.  The log normal values
        will be clipped to less than this value
    bleed_length_fac: float
        The bleed length is this factor times the *diameter* of the circular
        star mask
    """
    def __init__(self, *,
                 rng,
                 center_ra,
                 center_dec,
                 radius_degrees=5,
                 density=200,
                 radmean=3,
                 radstd=5,
                 radmin=3,
                 radmax=500,
                 bleed_length_fac=2):

        area = np.pi*radius_degrees**2
        count = rng.poisson(lam=density*area)

        stars = make_star_struct(size=count)
        self.stars = stars
        stars['ra'], stars['dec'] = randcap(
            rng=rng,
            nrand=count,
            ra=center_ra,
            dec=center_dec,
            radius=radius_degrees,
        )

        rpdf_ub = ngmix.priors.LogNormal(radmean, radstd, rng=rng)
        rpdf = ngmix.priors.Bounded1D(rpdf_ub, (radmin, radmax))

        stars['radius'] = rpdf.sample(size=count)
        stars['bleed_width'] = get_bleed_width(stars['radius'])
        stars['bleed_length'] = get_bleed_length(
            star_mask_rad=stars['radius'],
            bleed_length_fac=bleed_length_fac,
        )

    def set_mask(self, *, mask, wcs):
        """
        set bits in the input mask for contained stars

        Currently only stars for which the center is in
        the image are marked

        Parameters
        ----------
        mask: array
            Array of integer type
        wcs: wcs object
            A wcs object with the sky2image method
        """

        stars = self.stars

        xvals, yvals = wcs.radecToxy(
            stars['ra'], stars['dec'], galsim.degrees,
        )

        ny, nx = mask.shape
        nstars = 0
        for i in range(stars.size):
            x = xvals[i]
            y = yvals[i]
            if (x >= 0 and x < nx and y >= 0 and y < ny):

                add_star_and_bleed(
                    mask=mask,
                    x=x,
                    y=y,
                    radius=stars['radius'][i],
                    bleed_width=stars['bleed_width'][i],
                    bleed_length=stars['bleed_length'][i],
                )
                nstars += 1

        return nstars


def make_star_struct(size=1):
    dt = [
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('radius', 'f8'),  # pixels
        ('bleed_width', 'f8'),  # pixels
        ('bleed_length', 'f8'),  # pixels
    ]
    return np.zeros(size, dtype=dt)


def add_star_and_bleed(*,
                       mask,
                       x, y,
                       radius,
                       bleed_width,
                       bleed_length):
    """
    Add a circular star mask and bleed trail mask to
    the input mask image

    Parameters
    ----------
    mask: array
        Integer image
    x, y: floats
        The center position of the circle
    radius: float
        Radius of circle in pixels
    bleed_width: float
        Width of bleed in pixels
    bleed_length: float
        Length of bleed in pixels
    """
    add_star(
        mask=mask,
        x=x,
        y=y,
        radius=radius,
    )

    add_bleed(
        mask=mask,
        x=x,
        y=y,
        width=bleed_width,
        length=bleed_length,
    )


def get_bleed_width(star_mask_rad):
    """
    width is always odd
    """

    width = np.zeros(star_mask_rad.size, dtype='i4')
    width[:] = 1

    w, = np.where(star_mask_rad > 10)
    if w.size > 0:
        width[w] = 3

    return width


def get_bleed_length(*, star_mask_rad, bleed_length_fac):
    """
    length is always odd
    """
    diameter = star_mask_rad*2
    length = (bleed_length_fac*diameter).astype('i4')

    w, = np.where((length % 2) == 0)
    if w.size > 0:
        length[w] -= 1

    length.clip(min=1, out=length)
    return length


@njit
def add_star(*, mask, x, y, radius):
    """
    Add a circular star mask to the input mask image

    Parameters
    ----------
    mask: array
        Integer image
    x, y: floats
        The center position of the circle
    radius: float
        Radius of circle in pixels
    """

    radius2 = radius**2
    ny, nx = mask.shape

    for iy in range(ny):
        y2 = (y-iy)**2
        if y2 > radius2:
            continue

        for ix in range(nx):
            x2 = (x-ix)**2
            if x2 > radius2:
                continue

            rad = x2 + y2

            if rad > radius2:
                continue

            mask[iy, ix] |= STAR


@njit
def add_bleed(*, mask, x, y, width, length):
    """
    Add a bleed trail mask to the input mask image

    Parameters
    ----------
    mask: array
        Integer image
    x, y: floats
        The center position of the circle
    width: float
        Width of bleed in pixels
    length: float
        Length of bleed in pixels
    """

    ny, nx = mask.shape

    xpad = (width-1)//2
    ypad = (length-1)//2

    xmin = x - xpad
    xmax = x + xpad

    ymin = y - ypad
    ymax = y + ypad

    for iy in range(ymin, ymax+1):
        if iy < 0 or iy > (ny-1):
            continue

        for ix in range(xmin, xmax+1):
            if ix < 0 or ix > (nx-1):
                continue
            mask[iy, ix] |= BLEED
