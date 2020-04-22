import functools
import os
import fitsio
import numpy as np
from glob import glob
import esutil as eu


def add_bleed(*, bmask, pos, mag, band):
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
    bleed_stamp = get_bleed_stamp(mag=mag, band=band)

    stamp_nrow, stamp_ncol = bleed_stamp.shape
    stamp_cen = (np.array([stamp_nrow, stamp_ncol]) - 1)/2
    stamp_cen = stamp_cen.astype('i4')

    stamp_start_row = 0
    stamp_end_row = stamp_nrow-1
    stamp_start_col = 0
    stamp_end_col = stamp_ncol-1

    row_off_left = stamp_cen[0]
    row_off_right = stamp_nrow - stamp_cen[0]
    col_off_left = stamp_cen[1]
    col_off_right = stamp_ncol - stamp_cen[1]

    bmask_nrow, bmask_ncol = bmask.shape
    bmask_row = int(pos.y)
    bmask_col = int(pos.x)

    bmask_start_row = bmask_row - row_off_left
    bmask_end_row = bmask_row + row_off_right
    bmask_start_col = bmask_col - col_off_left
    bmask_end_col = bmask_col + col_off_right

    if bmask_start_row < 0:
        stamp_start_row = 0 - bmask_start_row
        bmask_start_row = 0

    if bmask_start_col < 0:
        stamp_start_col = 0 - bmask_start_col
        bmask_start_col = 0

    if bmask_end_row > (bmask_nrow - 1):
        stamp_end_row = (bmask_nrow - 1) - bmask_end_row
        bmask_end_row = bmask_nrow-1

    if bmask_end_col > (bmask_ncol - 1):
        stamp_end_col = (bmask_ncol - 1) - bmask_end_col
        bmask_end_col = bmask_ncol-1

    bmask[
        bmask_start_row:bmask_end_row+1,
        bmask_start_col:bmask_end_col+1,
    ] |= bleed_stamp[
        stamp_start_row:stamp_end_row+1,
        stamp_start_col:stamp_end_col+1,
    ]


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
