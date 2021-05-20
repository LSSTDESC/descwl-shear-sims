import numpy as np
import lsst.afw.image as afw_image
import galsim
from ..artifacts import (
    generate_edge_mask,
    generate_cosmic_rays,
    generate_bad_columns,
)
from ..lsst_bits import get_flagval


def get_bmask(*, image, rng, cosmic_rays, bad_columns):
    """
    get a bitmask for the image, including EDGE and
    optional cosmic rays and bad columns

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

    mask = generate_edge_mask(shape=shape, edge_width=5)

    if cosmic_rays:

        # bool mask
        c_mask = generate_cosmic_rays(
            shape=shape,
            rng=rng,
            mean_cosmic_rays=1,
        )
        mask[c_mask] |= get_flagval('CR') + get_flagval('SAT')

        # wait to do this later
        # image.array[cmask] = BAND_SAT_VALS[band]

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
