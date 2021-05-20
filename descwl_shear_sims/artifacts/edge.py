import numpy as np

from ..lsst_bits import get_flagval


def generate_edge_mask(*, shape, edge_width):
    """
    generate a basic mask with edges marked

    Parameters
    ----------
    shape: tuple
        2-element tuple for shape of bitmask
    edge_width: int
        Width of border to marked EDGE
    """

    ny, nx = shape
    bmask = np.zeros(shape, dtype=np.int64)

    edgeflag = get_flagval('EDGE')

    ew = edge_width
    bmask[0:ew, :] = edgeflag
    bmask[ny-ew:, :] = edgeflag
    bmask[:, 0:ew] = edgeflag
    bmask[:, nx-ew:] = edgeflag

    return bmask
