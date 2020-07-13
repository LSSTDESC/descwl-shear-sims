from numba import njit
import numpy as np
import galsim
from ..star_bleeds import add_bleed, get_max_mag_with_bleed
from ..gen_masks import (
    generate_basic_mask, generate_cosmic_rays, generate_bad_columns,
)
from ..gen_star_masks import add_bright_star_mask
from ..lsst_bits import BAD_COLUMN, COSMIC_RAY, SAT, BRIGHT


def add_bleeds(*, image, origin, bmask, shifts, mags, band):
    """
    Add a bleed for each saturated object

    Parameters
    ----------
    image: galsim Image
        Image will be modified to have saturated values in the
        bleed
    origin: galsim.PositionD
        Origin of image in pixels
    bmask: galsim Image
        Mask will be modified to have saturated values in the
        bleed
    shifts: array
        Fields dx and dy.
    mags: list
        List of mags
    band: string
        Filter band

    Returns
    --------
    None
    """

    wcs = image.wcs

    jac_wcs = wcs.jacobian(world_pos=wcs.center)
    max_mag = get_max_mag_with_bleed(band=band)

    for i in range(shifts.size):
        mag = mags[i]
        if mag < max_mag:
            shift_pos = galsim.PositionD(
                x=shifts['dx'][i],
                y=shifts['dy'][i],
            )
            pos = jac_wcs.toImage(shift_pos) + origin

            add_bleed(
                image=image.array,
                bmask=bmask.array,
                pos=pos,
                mag=mag,
                band=band,
            )


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

    mask = generate_basic_mask(shape=shape, edge_width=5)

    if cosmic_rays:

        # bool mask
        c_mask = generate_cosmic_rays(
            shape=shape,
            rng=rng,
            mean_cosmic_rays=1,
        )
        mask[c_mask] |= COSMIC_RAY + SAT

        # wait to do this later
        # image.array[cmask] = BAND_SAT_VALS[band]

    if bad_columns:
        # bool mask
        bc_msk = generate_bad_columns(
            shape=shape,
            rng=rng,
            mean_bad_cols=1,
        )
        mask[bc_msk] |= BAD_COLUMN
        image.array[bc_msk] = 0.0

    return galsim.Image(
        mask,
        bounds=image.bounds,
        wcs=image.wcs,
        dtype=np.int32,
    )


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
        val=BRIGHT,
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
