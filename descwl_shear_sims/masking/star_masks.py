import numpy as np
import galsim
from numba import njit
from ..lsst_bits import get_flagval


@njit
def add_bright_star_mask(*, bmask, x, y, radius, val):
    """
    Add a circular bright star mask to the input mask image

    Parameters
    ----------
    bmask: array
        Integer image
    x, y: floats
        The center position of the circle
    radius: float
        Radius of circle in pixels
    val: int
        Val to "or" into bmask
    """

    intx = int(x)
    inty = int(y)

    radius2 = radius**2
    ny, nx = bmask.shape

    for iy in range(ny):
        y2 = (inty-iy)**2
        if y2 > radius2:
            continue

        for ix in range(nx):
            x2 = (intx-ix)**2
            rad2 = x2 + y2

            if rad2 > radius2:
                continue

            bmask[iy, ix] |= val


def calculate_and_add_bright_star_mask(
    *,
    image,
    bmask,
    shift,
    wcs,
    origin,  # pixel origin
    threshold,
):
    """
    Get a list of psf convolved objects for a variable psf

    Parameters
    ----------
    image: array
        numpy array representing the image
    bmask: array
        numpy array representing the bitmask
    shift: array
        scalar array with fields dx and dy, which are du, dv offsets in sky
        coords.
    wcs: galsim wcs
        For the SE image
    origin: galsim.PositionD
        Origin of SE image (with offset included)
    threshold: float
        The mask will extend to where the profile reaches this value
    """

    jac_wcs = wcs.jacobian(world_pos=wcs.center)

    shift_pos = galsim.PositionD(
        x=shift['dx'],
        y=shift['dy'],
    )
    pos = jac_wcs.toImage(shift_pos) + origin

    radius = calculate_bright_star_mask_radius(
        image=image,
        objrow=pos.y,
        objcol=pos.x,
        threshold=threshold,
    )
    add_bright_star_mask(
        bmask=bmask,
        x=pos.x,
        y=pos.y,
        radius=radius,
        val=get_flagval('BRIGHT'),
    )


@njit
def calculate_bright_star_mask_radius(*, image, objrow, objcol, threshold):
    """
    get the radius at which the profile drops to the specified threshold

    Parameters
    ----------
    image: 2d array
        The image
    objrow: float
        The row position of the object center
    objcol: float
        The column position of the object center
    threshold: float
        The mask will extend to where the profile reaches this value

    Returns
    -------
    radius: float
        The radius
    """

    nrows, ncols = image.shape
    radius2 = 0.0

    for row in range(nrows):
        row2 = (objrow - row)**2

        for col in range(ncols):
            col2 = (objcol - col)**2

            tradius2 = row2 + col2
            if tradius2 < radius2:
                # we are already within a previously calculated radius
                continue

            val = image[row, col]
            if val > threshold:
                radius2 = tradius2

    radius = np.sqrt(radius2)
    return radius
