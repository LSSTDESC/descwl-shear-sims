from numba import njit
from .sim_constants import ZERO_POINT

# saturation value for visit images, rescaled to
# our zero point
BAND_SAT_VALS = {
    'g': 140000 * 10.0**(0.4*(ZERO_POINT-32.325)),  # from example images
    'r': 140000 * 10.0**(0.4*(ZERO_POINT-32.16)),
    'i': 140000 * 10.0**(0.4*(ZERO_POINT-31.825)),
    'z': 140000 * 10.0**(0.4*(ZERO_POINT-31.50)),
}


@njit
def saturate_image_and_mask(*, image, bmask, sat_val, flagval):
    """
    clip image values at saturation and set the SAT mask bit.  Note
    if the bmask already has SAT set, then the value will also be set

    Parameters
    ----------
    image: ndarray
        The image to clip
    bmask: ndarray
        The bitmask image
    sat_val: float
        The saturation value to use
    """
    ny, nx = image.shape

    for row in range(ny):
        for col in range(nx):
            if (bmask[row, col] & flagval) != 0 or image[row, col] > sat_val:
                image[row, col] = sat_val
                bmask[row, col] |= flagval
