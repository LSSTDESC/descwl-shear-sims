import numpy as np
import lsst.afw.image as afw_image
import galsim
from ..artifacts import (
    generate_cosmic_rays,
    generate_bad_columns,
)
from ..lsst_bits import get_flagval


def get_bmask_and_set_image(*, image, rng, cosmic_rays, bad_columns):
    """
    get a bitmask for the image, including optional cosmic rays and bad columns.
    If bad columns are set, the image is set to zero in those pixels

    Parameters
    ----------
    image: galsim.Image type
        The image
    rng: np.random.RandomState
        The random state object
    cosmic_rays: bool
        Whether to add cosmic rays
    bad_columns: bool
        Whether to add bad columns

    Returns
    -------
    galsim.Image of type int32
    """
    shape = image.array.shape

    mask = np.zeros(shape, dtype=np.int64)

    if cosmic_rays:

        # bool mask
        c_mask = generate_cosmic_rays(
            shape=shape,
            rng=rng,
            mean_cosmic_rays=1,
        )
        mask[c_mask] |= get_flagval('CR') + get_flagval('SAT')

    if bad_columns:
        # bool mask
        bc_msk = generate_bad_columns(
            shape=shape,
            rng=rng,
            mean_bad_cols=1,
        )
        mask[bc_msk] |= afw_image.Mask.getPlaneBitMask('BAD')
        image.array[bc_msk] = 0.0

    return galsim.Image(
        mask,
        bounds=image.bounds,
        wcs=image.wcs,
        dtype=np.int32,
    )
