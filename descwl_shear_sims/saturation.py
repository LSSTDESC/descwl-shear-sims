from numba import njit
from .lsst_bits import SAT

# saturation value for images
BAND_SAT_VALS = {
    'g': 150000,  # from example images
    'r': 150000,  # TODO get value
    'i': 150000,  # TODO get value
    'z': 150000,  # TODO get value
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
