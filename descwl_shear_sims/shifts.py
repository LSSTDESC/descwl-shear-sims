import numpy as np

from .constants import (
    RANDOM_DENSITY,
    GRID_N_ON_SIDE,
    SCALE,
)


def get_shifts(
    *,
    rng,
    layout,
    coadd_dim=None,
    buff=None,
    nobj=None,
    sep=None,
):
    """
    make position shifts for objects

    rng: numpy.random.RandomState
        Numpy random state
    coadd_dim: int
        Dimensions of final coadd
    buff: int
        Buffer region where no objects will be drawn
    layout: string
        'grid' or 'random'
    nobj: int, optional
        Optional number of objects to draw, defaults to None
        in which case a poission deviate is draw according
        to RANDOM_DENSITY
    sep: float, optional
        The separation in arcseconds for layout='pair'
    """

    if layout == 'pair':

        if sep is None:
            raise ValueError(f'send sep= for layout {layout}')

        shifts = get_pair_shifts(rng=rng, sep=sep)
    else:

        if coadd_dim is None:
            raise ValueError(f'send coadd_dim= for layout {layout}')
        if buff is None:
            raise ValueError(f'send buff= for layout {layout}')

        if layout == 'grid':
            shifts = get_grid_shifts(
                rng=rng,
                dim=coadd_dim,
                n_on_side=GRID_N_ON_SIDE,
            )
        elif layout == 'random':
            # area covered by objects
            if nobj is None:
                area = ((coadd_dim - 2*buff)*SCALE/60)**2
                nobj_mean = area * RANDOM_DENSITY
                nobj = rng.poisson(nobj_mean)

            shifts = get_random_shifts(
                rng=rng,
                dim=coadd_dim,
                buff=buff,
                size=nobj,
            )
        else:
            raise ValueError("bad layout: '%s'" % layout)

    return shifts


def get_grid_shifts(*, rng, dim, n_on_side):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    n_on_side: int
        Number of objects on each side

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    spacing = dim/(n_on_side+1)*SCALE

    ntot = n_on_side**2

    # ix/iy are really on the sky
    grid = spacing*(np.arange(n_on_side) - (n_on_side-1)/2)

    shifts = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])

    i = 0
    for ix in range(n_on_side):
        for iy in range(n_on_side):
            dx = grid[ix] + SCALE*rng.uniform(low=-0.5, high=0.5)
            dy = grid[iy] + SCALE*rng.uniform(low=-0.5, high=0.5)

            shifts['dx'][i] = dx
            shifts['dy'][i] = dy
            i += 1

    return shifts


def get_random_shifts(*, rng, dim, buff, size):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    n_on_side: int
        Number of objects on each side

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    halfwidth = (dim - 2*buff)/2.0

    low = -halfwidth*SCALE
    high = halfwidth*SCALE

    shifts = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])

    shifts['dx'] = rng.uniform(low=low, high=high, size=size)
    shifts['dy'] = rng.uniform(low=low, high=high, size=size)

    return shifts


def get_pair_shifts(*, rng, sep):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    sep: float
        Separation of pair in arcsec

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    shifts = np.zeros(2, dtype=[('dx', 'f8'), ('dy', 'f8')])

    angle = rng.uniform(low=0, high=np.pi)
    shift_radius = sep / 2

    xdither, ydither = SCALE*rng.uniform(low=-0.5, high=0.5, size=2)

    dx1 = np.cos(angle)*shift_radius
    dy1 = np.sin(angle)*shift_radius
    dx2 = -dx1
    dy2 = -dy1

    dx1 += xdither
    dy1 += ydither

    dx2 += xdither
    dy2 += ydither

    shifts['dx'][0] = dx1
    shifts['dy'][0] = dy1

    shifts['dx'][1] = dx2
    shifts['dy'][1] = dy2

    return shifts
