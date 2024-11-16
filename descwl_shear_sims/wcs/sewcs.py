from typing import List, Tuple

import galsim
import numpy as np

from ..constants import SCALE, WORLD_ORIGIN
from . import make_wcs


def make_se_wcs(
    *,
    pixel_scale: float = SCALE,
    image_origin: galsim.PositionD,
    world_origin: galsim.CelestialCoord = WORLD_ORIGIN,
    dither: bool = False,
    rotate: bool = False,
    theta: float | None = None,
    rng=None,
):
    """
    This function generates a single exposure galsim WCS

    Parameters
    ----------

    pixel_scale: float
        image pixel scale, default: 0.2
    image_origin: galsim.PositionD
        Image origin position
    world_origin: galsim.CelestialCoord
        Origin on the sky
    dither: bool, optional
        whether to do dither or not, default: False
    rotate: bool
        whether to do rotation or not, default: False
    theta: float
        rotation angle, optional, default: None
    rng: numpy.random.RandomState
        random number generator, optional, default: None

    Returns
    ----------
    Galsim WCS of the single exposure
    """

    if dither:
        # do a small offset of the origin
        assert rng is not None
        dither_range = 0.5
        off = rng.uniform(low=-dither_range, high=dither_range, size=2)
        offset = galsim.PositionD(x=off[0], y=off[1])
        image_origin = image_origin + offset

    if rotate:
        # roate the single exposure
        if theta is None:
            assert rng is not None
            theta = rng.uniform(low=0, high=2 * np.pi)
    else:
        theta = None

    return make_wcs(
        scale=pixel_scale,
        theta=theta,
        image_origin=image_origin,
        world_origin=world_origin,
    )
