import numpy as np
from numba import njit


@njit
def add_bright_star_mask(*, bmask, x, y, radius_pixels, val):
    """
    Add a circular bright star mask to the input mask image

    Parameters
    ----------
    bmask: array
        Integer image
    x, y: floats
        The center position of the circle
    radius_pixels: float
        Radius of circle in pixels
    val: int
        Val to "or" into bmask
    """

    intx = int(x)
    inty = int(y)

    radius2 = radius_pixels**2
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


# def calculate_and_add_bright_star_mask(
#     *,
#     image,
#     bmask,
#     image_pos,
#     threshold,
# ):
#     """
#     Get a list of psf convolved objects for a variable psf
#
#     Parameters
#     ----------
#     image: array
#         numpy array representing the image
#     bmask: array
#         numpy array representing the bitmask
#     image_pos: galsim.PositionD
#         Center of object in image
#     threshold: float
#         The mask will extend to where the profile reaches this value
#
#     Returns
#     -------
#     radius_pixels: float
#         Radius of mask in pixels
#     """
#     from ..lsst_bits import get_flagval
#
#     radius_pixels = calculate_bright_star_mask_radius(
#         image=image,
#         objrow=image_pos.y,
#         objcol=image_pos.x,
#         threshold=threshold,
#     )
#     add_bright_star_mask(
#         bmask=bmask,
#         x=image_pos.x,
#         y=image_pos.y,
#         radius_pixels=radius_pixels,
#         val=get_flagval('BRIGHT'),
#     )
#     return radius_pixels
