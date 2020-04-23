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
