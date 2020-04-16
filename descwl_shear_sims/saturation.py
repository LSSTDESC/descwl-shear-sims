import numpy as np
from numba import njit
from .lsst_bits import SAT
from .sim_constants import ZERO_POINT

# saturation value for visit images, rescaled to
# our zero point
BAND_SAT_VALS = {
    'g': 140000 * 10.0**(0.4*(ZERO_POINT-32.325)),  # from example images
    'r': 140000 * 10.0**(0.4*(ZERO_POINT-32.16)),
    'i': 140000 * 10.0**(0.4*(ZERO_POINT-31.825)),
    'z': 140000 * 10.0**(0.4*(ZERO_POINT-31.50)),
}

# From the LSST science book at 15 seconds
# convert for 30 second exposures
ltwo = np.log10(2)
BAND_STAR_MAG_SAT = {
    'u': 14.7 + ltwo,
    'g': 15.7 + ltwo,
    'r': 15.8 + ltwo,
    'i': 15.8 + ltwo,
    'z': 15.3 + ltwo,
}


@njit
def saturate_image_and_mask(*, image, mask, sat_val):
    """
    clip image values at saturation and set the SAT mask bit

    Parameters
    ----------
    image: ndarray
        The image to clip
    mask: ndarray
        The mask image
    sat_val: float
        The saturation value to use
    """
    ny, nx = image.shape

    for row in range(ny):
        for col in range(nx):
            if image[row, col] > sat_val:
                image[row, col] = sat_val
                mask[row, col] |= SAT
