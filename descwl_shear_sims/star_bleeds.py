import functools
import os
import fitsio
import numpy as np
from glob import glob
import esutil as eu
from numba import njit
from .saturation import BAND_SAT_VALS


def add_bleed(*, image, bmask, pos, mag, band):
    """
    add a bleed mask at the specified location

    Parameters
    ----------
    bmask: array
        The bit mask array
    pos: position
        must have pos.x, pos.y
    mag: float
        The magnitude of the object.  The nearest match
        in the bleed catalog is used.
    band: string
        The filter band
    """

    assert image.shape == bmask.shape

    bleed_stamp = get_bleed_stamp(mag=mag, band=band)

    stamp_nrow, stamp_ncol = bleed_stamp.shape
    stamp_cen = (np.array(bleed_stamp.shape) - 1)/2
    stamp_cen = stamp_cen.astype('i4')

    row_off_left = stamp_cen[0]
    col_off_left = stamp_cen[1]

    bmask_row = int(pos.y)
    bmask_col = int(pos.x)
    print('stamp dims:', stamp_nrow, stamp_ncol)
    print('bmask pos:', bmask_row, bmask_col)

    bmask_start_row = bmask_row - row_off_left
    bmask_start_col = bmask_col - col_off_left

    _add_bleed(
        image=image,
        bmask=bmask,
        stamp=bleed_stamp,
        start_row=bmask_start_row,
        start_col=bmask_start_col,
        val=BAND_SAT_VALS[band],
    )


@njit
def _add_bleed(*, image, bmask, stamp, start_row, start_col, val):
    """
    or the stamp into the indicated bitmask image and set the
    saturation value
    """
    nrows, ncols = bmask.shape

    stamp_nrows, stamp_ncols = stamp.shape

    for row in range(stamp_nrows):
        bmask_row = start_row + row
        if bmask_row < 0 or bmask_row > (nrows-1):
            continue

        for col in range(stamp_ncols):
            bmask_col = start_col + col
            if bmask_col < 0 or bmask_col > (ncols-1):
                continue

            mask_val = stamp[row, col]
            bmask[bmask_row, bmask_col] |= mask_val
            image[bmask_row, bmask_col] = val


def get_bleed_stamp(*, mag, band):
    """
    get an example bleed stamp


    Parameters
    ----------
    mag: float
        The magnitude of the object.  The nearest match in the bleed catalog is
        used.
    band: string
        The filter band

    Returns
    -------
    array
    """
    bleeds = get_cached_bleeds()[band]

    index = bleeds['mag'].searchsorted(mag)

    stamp = bleeds['stamp'][index].copy()
    stamp = stamp.reshape(
        bleeds['stamp_nrow'][index],
        bleeds['stamp_ncol'][index],
    )
    return stamp


@functools.lru_cache(maxsize=1)
def get_cached_bleeds():
    """
    get a dict keyed by filter with example bleeds

    the read is cached
    """
    if 'CATSIM_DIR' not in os.environ:
        raise OSError('CATSIM_DIR not defined')

    dir = os.environ['CATSIM_DIR']
    pattern = os.path.join(dir, 'extracted-*.fits.gz')

    flist = glob(pattern)
    assert len(flist) != 0

    dlist = []
    for f in flist:
        with fitsio.FITS(f, vstorage='object') as fits:
            d = fits[1].read()
        dlist.append(d)

    data = eu.numpy_util.combine_arrlist(dlist)
    s = data['mag'].argsort()
    data = data[s]

    # same for all bands for now
    return {
        'g': data,
        'r': data,
        'i': data,
        'z': data,
    }
