from numba import njit


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
