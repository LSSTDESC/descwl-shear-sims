from numba import njit
from .lsst_bits import SAT
from .sim_constants import ZERO_POINT

# saturation value for images
BAND_SAT_VALS = {
    'g': 150000 * 10.0**(0.4*(ZERO_POINT-32.325)),  # from example images
    'r': 150000 * 10.0**(0.4*(ZERO_POINT-32.325)),  # TODO get value and zp
    'i': 150000 * 10.0**(0.4*(ZERO_POINT-32.325)),  # TODO get value and zp
    'z': 150000 * 10.0**(0.4*(ZERO_POINT-32.325)),  # TODO get value and zp
}

# From the LSST science book
# mag to saturate for 30 second exposures, need this for the
# longer exposures, so for now just add one TODO
BAND_STAR_MAG_SAT = {
    'u': 14.7+1,
    'g': 15.7+1,
    'r': 15.8+1,
    'i': 15.8+1,
    'z': 15.3+1,
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
