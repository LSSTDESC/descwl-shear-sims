import functools
import os
import galsim
import numpy as np
from glob import glob
import esutil as eu
from numba import njit
from ..saturation import BAND_SAT_VALS
from ..lsst_bits import get_flagval


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

    bmask_start_row = bmask_row - row_off_left
    bmask_start_col = bmask_col - col_off_left

    _add_bleed(
        image=image,
        bmask=bmask,
        stamp=bleed_stamp,
        start_row=bmask_start_row,
        start_col=bmask_start_col,
        val=BAND_SAT_VALS[band],
        flagval=get_flagval('SAT'),
    )


@njit
def _add_bleed(*, image, bmask, stamp, start_row, start_col, val, flagval):
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
            if mask_val & flagval != 0:
                bmask[bmask_row, bmask_col] |= flagval
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
    if index > bleeds.size-1:
        index = bleeds.size-1

    stamp = bleeds['stamp'][index].copy()
    stamp = stamp.reshape(
        bleeds['stamp_nrow'][index],
        bleeds['stamp_ncol'][index],
    )
    return stamp


def get_max_mag_with_bleed(*, band):
    """
    get the largest mag that has a bleed
    """
    # they are sorted by mag
    bleeds = get_cached_bleeds()[band]
    return bleeds['mag'][-1]


@functools.lru_cache(maxsize=1)
def get_cached_bleeds():
    """
    get a dict keyed by filter with example bleeds

    the data are sorted for mag, so one can use searchsorted
    to get a match

    the read is cached
    """
    import fitsio

    if 'CATSIM_DIR' not in os.environ:
        raise OSError('CATSIM_DIR not defined')

    dir = os.environ['CATSIM_DIR']
    bdict = {}

    for band in ['g', 'r', 'i', 'z']:
        pattern = os.path.join(dir, 'extracted-*-%s-*.fits.gz' % band)

        flist = glob(pattern)
        assert len(flist) != 0

        dlist = []
        for f in flist:
            with fitsio.FITS(f, vstorage='object') as fits:
                d = fits[1].read()

            d = remove_off_center(d)
            dlist.append(d)

        data = eu.numpy_util.combine_arrlist(dlist)
        s = data['mag'].argsort()
        data = data[s]
        bdict[band] = data

    return bdict


def remove_off_center(data):
    """
    remove examples where they are too far off center
    """

    keep = []
    for i in range(data.size):

        d = data[i]

        lower_len = d['row'] + 1
        upper_len = d['stamp_nrow'] - d['row']

        if abs(upper_len - lower_len) < 5:
            keep.append(i)

    return data[keep]
