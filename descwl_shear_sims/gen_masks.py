"""
copy-paste from my (beckermr) personal code here
https://github.com/beckermr/metadetect-coadding-sims/blob/master/coadd_mdetsims/masking.py
"""
import numpy as np

from .lsst_bits import EDGE


def generate_basic_mask(*, shape, edge_width):
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

    ew = edge_width
    bmask[0:ew, :] = EDGE
    bmask[ny-ew:, :] = EDGE
    bmask[:, 0:ew] = EDGE
    bmask[:, nx-ew:] = EDGE

    return bmask


def generate_cosmic_rays(
        *, shape, mean_cosmic_rays=1, min_length=10, max_length=30,
        rng=None,
        thick=False,
):
    """Generate a binary mask w/ cosmic rays.

    This routine generates cosmic rays by choosing a random
    position in the image, a random angle, and a random
    length of the cosmic ray track. Then pixels along the line
    determined by the length of the track, the position, and the angle
    are marked. The width of the track is broadened a bit when
    the row or column changes.

    The total number of cosmic rays is Poisson distributed according to
    `mean_cosmic_rays`.

    The length of the track is chosen uniformly between `min_length` and
    `max_length`.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the mask to generate.
    mean_cosmic_rays : int, optional
        The mean number of cosmic rays.
    min_length : int, optional
        The minimum length of the track.
    max_length : int, optional
        The maximum length of the track.
    rng : np.random.RandomState or None, optional
        An RNG to use. If none is provided, a new `np.random.RandomState`
        state instance will be created.
    thick: bool
        If thick, make the cosmics thicker.  Default False.

    Returns
    -------
    msk : np.ndarray, shape `shape`
        A boolean mask marking the locations of the cosmic rays.
    """
    msk = np.zeros(shape)
    rng = rng or np.random.RandomState()
    n_cosmic_rays = rng.poisson(mean_cosmic_rays)

    for _ in range(n_cosmic_rays):
        y = rng.randint(0, msk.shape[0]-1)
        x = rng.randint(0, msk.shape[1]-1)
        angle = rng.uniform() * 2.0 * np.pi
        n_pix = rng.randint(min_length, max_length+1)
        cosa = np.cos(angle)
        sina = np.sin(angle)

        _x_prev = None
        _y_prev = None
        for _ in range(n_pix):
            _x = int(x + 0.5)
            _y = int(y + 0.5)
            if _y >= 0 and _y < msk.shape[0] and _x >= 0 and _x < msk.shape[1]:
                if thick:
                    if _x_prev is not None and _y_prev is not None:
                        if _x_prev != _x:
                            msk[_y, _x_prev] = 1
                        if _y_prev != _y:
                            msk[_y_prev, _x] = 1

                msk[_y, _x] = 1
                _x_prev = _x
                _y_prev = _y
            x += cosa
            y += sina

    return msk.astype(bool)


def generate_bad_columns(
        *, shape, mean_bad_cols=1,
        gap_prob=0.30,
        min_gap_frac=0.1,
        max_gap_frac=0.3,
        rng=None,
):
    """Generate a binary mask w/ bad columns.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the mask to generate.
    mean_bad_cols : float, optional
        The mean of the Poisson distribution for the total number of
        bad columns to generate.
    gap_prob : float
        The probability that the bad column has a gap in it.
    min_gap_frac : float
        The minimum fraction of the image that the gap spans.
    max_gap_frac : floatn
        The maximum fraction of the image that the gap spans.
    rng : np.random.RandomState or None, optional
        An RNG to use. If none is provided, a new `np.random.RandomState`
        state instance will be created.

    Returns
    -------
    msk : np.ndarray, shape `shape`
        A boolean mask marking the locations of the bad columns.
    """

    msk = np.zeros(shape)
    rng = rng or np.random.RandomState()
    n_bad_cols = rng.poisson(mean_bad_cols)

    for _ in range(n_bad_cols):

        x = rng.choice(msk.shape[1])

        # set the mask first
        msk[:, x] = 1

        # add gaps
        if rng.uniform() < gap_prob:
            gfrac = rng.uniform(min_gap_frac, max_gap_frac)
            glength = int(msk.shape[0] * gfrac)
            gloc = rng.randint(0, msk.shape[0] - glength)
            msk[gloc:gloc+glength, x] = 0

    return msk.astype(bool)
