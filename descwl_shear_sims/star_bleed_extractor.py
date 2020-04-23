import os
import numpy as np
from numba import njit
from .lsst_bits import SAT
import esutil as eu
import fitsio


def extract_bleeds(*, image_file, cat_file, out_file):
    """
    extract star saturation/bleed regions to a file

    Parameters
    ----------
    image_file: str
        path to the image file
    cat_file: str
        path to the catalog file with entries that look like
        SourceID Flux Realized_flux xPix yPix flags GalSimType
    out_file: str
        path for output file
    """
    import lsst.afw.image as afw_image

    print('will read from:', image_file)
    print('will write to:', out_file)

    assert image_file != out_file
    exp = afw_image.ExposureF(image_file)
    calib = exp.getPhotoCalib()

    magzero = 2.5*np.log10(calib.getInstFluxAtZeroMagnitude())

    cat = _read_catalog(fname=cat_file, magzero=magzero)

    mask = exp.mask.array

    print('extracting stamps')
    keep = np.ones(cat.size, dtype='bool')

    for i in range(cat.size):
        row = int(cat['row_orig'][i])
        col = int(cat['col_orig'][i])

        if mask[row, col] & SAT == 0:
            keep[i] = False
            continue

        row_start, row_end, col_start, col_end = _get_bleed_bbox(
            mask=mask,
            row=row,
            col=col,
        )

        stamp_full = mask[
            row_start:row_end+1,
            col_start:col_end+1,
        ]
        stamp = stamp_full*0
        w = np.where(stamp_full & SAT != 0)
        stamp[w] = SAT

        cat['row'][i] = row - row_start
        cat['col'][i] = col - col_start
        cat['stamp'][i] = stamp
        cat['stamp_nrow'][i] = stamp.shape[0]
        cat['stamp_ncol'][i] = stamp.shape[1]

    w, = np.where(keep)
    print('keeping: %d/%d' % (w.size, cat.size))
    cat = cat[w]

    print('writing:', out_file)
    with fitsio.FITS(out_file, 'rw', vstorage='object', clobber=True) as fits:
        fits.write(cat)


def extract_bleeds_flist(*, calexp_flist):
    """
    for each of the input calexp, extract star saturation/bleed regions to a
    file

    Parameters
    ----------
    calexp_flist: list
        The list of calexp files
    """
    fdict_list = _get_fdict_list(calexp_flist)

    for i, fdict in enumerate(fdict_list):
        print('-'*70)
        print('%d/%d' % (i+1, len(fdict_list)))
        extract_bleeds(
            image_file=fdict['calexp'],
            cat_file=fdict['catfile'],
            out_file=fdict['exfile'],
        )

    return fdict_list


def _get_fdict_list(calexps):
    """
    get the list of dicts for inputs and outputs

    Parameters
    ----------
    calexp_flist: list
        The list of calexp files

    Returns
    -------
    list of dict, wach dict with keys 'calexp', 'catfile', 'exfile'
    """
    flist = []
    for calexp in calexps:
        cs = calexp[7:].split('-')

        ind_string = cs[0]
        ind = int(ind_string)
        band = cs[1]
        raft = cs[2]
        sensor = cs[3]
        det = cs[4].split('.')[0]

        d = {
            'ind_string': ind_string,
            'ind': ind,
            'raft': raft,
            'sensor': sensor,
            'band': band,
            'det': det,
        }

        catfile = 'centroid_%(ind)d_%(raft)s_%(sensor)s_%(band)s.txt' % d
        assert os.path.exists(catfile)

        exfile = 'extracted-%(ind_string)s-%(band)s-%(raft)s-%(sensor)s-%(det)s.fits.gz' % d  # noqa

        flist.append({
            'calexp': calexp,
            'catfile': catfile,
            'exfile': exfile,
        })

    return flist


@njit
def _get_bleed_bbox(*, mask, row, col):
    """
    get range of rows and cols that include the bleed
    """

    row_start = row
    while mask[row_start, col] & SAT != 0:
        row_start -= 1

    row_end = row
    while mask[row_end, col] & SAT != 0:
        row_end += 1

    col_start = col
    while mask[row, col_start] & SAT != 0:
        col_start -= 1

    col_end = col
    while mask[row, col_end] & SAT != 0:
        col_end += 1

    return row_start, row_end, col_start, col_end


def _read_catalog(*, fname, magzero):
    print('reading', fname, 'with zero point', magzero)

    dt = [
        ('id', 'i8'),
        ('mag', 'f4'),
        ('row_orig', 'f4'),
        ('col_orig', 'f4'),
        ('row', 'f4'),
        ('col', 'f4'),
        ('stamp', 'O'),
        ('stamp_nrow', 'i4'),
        ('stamp_ncol', 'i4'),
    ]

    dlist = []
    with open(fname) as fobj:
        for line in fobj:
            ls = line.split()
            if ls[0] == 'SourceID':
                continue

            d = np.zeros(1, dtype=dt)
            d['id'] = int(ls[0])
            flux = float(ls[2])
            d['mag'] = magzero - 2.5*np.log10(flux)
            # d['row_orig'] = float(ls[4])
            # d['col_orig'] = float(ls[3])
            d['row_orig'] = float(ls[3])
            d['col_orig'] = float(ls[4])

            dlist.append(d)

    cat = eu.numpy_util.combine_arrlist(dlist)
    print('read:', cat.size, 'from', fname)
    return cat
