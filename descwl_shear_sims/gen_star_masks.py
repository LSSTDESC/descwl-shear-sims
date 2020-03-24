from numba import njit
import ngmix

from .lsst_bits import SAT
from .saturation import BAND_SAT_VALS


class StarMaskPDFs(object):
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
    radmean: float
        Mean star mask radius in pixels for log normal distribution,
        default 1
    radstd: float
        Radius standard deviation in pixels for log normal distribution,
        default 5
    radmin: float
        Minimum radius in pixels of star masks.  The log normal values
        will be clipped to more than this value. Default 2
    radmax: float
        Maximum radius in pixels of star masks.  The log normal values
        will be clipped to less than this value. Default 20.
    bleed_length_fac: float
        The bleed length is this factor times the *diameter* of the circular
        star mask, default 4
    """
    def __init__(self, *,
                 rng,
                 radmean=1,
                 radstd=5,
                 radmin=2,
                 radmax=20,
                 bleed_length_fac=4):

        rpdf_ub = ngmix.priors.LogNormal(radmean, radstd, rng=rng)
        self.radius_pdf = ngmix.priors.Bounded1D(rpdf_ub, (radmin, radmax))
        self.bleed_length_fac = bleed_length_fac

    def sample(self):
        """
        sample radius, bleed_width, and bleed_length
        """

        res = {}
        res['radius'] = self.radius_pdf.sample()
        res['bleed_width'] = get_bleed_width(res['radius'])
        res['bleed_length'] = get_bleed_length(
            star_mask_rad=res['radius'],
            bleed_length_fac=self.bleed_length_fac,
        )
        return res


def add_star_and_bleed(*,
                       mask,
                       image,
                       band,
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

    sat_val = BAND_SAT_VALS[band]

    add_star(
        mask=mask,
        image=image,
        x=x,
        y=y,
        radius=radius,
        sat_val=sat_val,
    )

    add_bleed(
        mask=mask,
        image=image,
        x=x,
        y=y,
        width=bleed_width,
        length=bleed_length,
        sat_val=sat_val,
    )


def get_bleed_width(star_mask_rad):
    """
    width is always odd
    """

    if star_mask_rad > 10:
        width = 3
    else:
        width = 1

    return width


def get_bleed_length(*, star_mask_rad, bleed_length_fac):
    """
    length is always odd
    """
    diameter = star_mask_rad*2
    length = int(bleed_length_fac*diameter)

    if (length % 2) == 0:
        length -= 1

    if length < 1:
        length = 1

    return length


@njit
def add_star(*, mask, image, x, y, radius, sat_val):
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
    sat_val: float
        Value at saturation
    """

    intx = int(x)
    inty = int(y)

    radius2 = radius**2
    ny, nx = mask.shape

    for iy in range(ny):
        y2 = (inty-iy)**2
        if y2 > radius2:
            continue

        for ix in range(nx):
            x2 = (intx-ix)**2
            rad2 = x2 + y2

            if rad2 > radius2:
                continue

            mask[iy, ix] |= SAT
            image[iy, ix] = sat_val


@njit
def add_bleed(*, mask, image, x, y, width, length, sat_val):
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
    sat_val: float
        Value at saturation
    """

    ny, nx = mask.shape

    xpad = (width-1)//2
    ypad = (length-1)//2

    intx = int(x)
    inty = int(y)

    xmin = intx - xpad
    xmax = intx + xpad

    ymin = inty - ypad
    ymax = inty + ypad

    for iy in range(ymin, ymax+1):
        if iy < 0 or iy > (ny-1):
            continue

        for ix in range(xmin, xmax+1):
            if ix < 0 or ix > (nx-1):
                continue
            mask[iy, ix] |= SAT
            image[iy, ix] = sat_val
