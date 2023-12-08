import numpy as np

from .constants import (
    RANDOM_DENSITY,
    GRID_SPACING,
    HEX_SPACING,
    SCALE,
)


def get_shifts(
    *,
    rng,
    layout,
    coadd_dim=None,
    buff=0,
    pixel_scale=SCALE,
    nobj=None,
    sep=None,
):
    """
    make position shifts for objects

    rng: numpy.random.RandomState
        Numpy random state
    coadd_dim: int
        Dimensions of final coadd
    buff: int, optional
        Buffer region where no objects will be drawn.  Default 0.
    pixel_scale: float
        pixel scale
    layout: string
        'grid', 'pair', 'hex', or 'random'
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

        shifts = get_pair_shifts(rng=rng, sep=sep, pixel_scale=pixel_scale)
    else:

        if coadd_dim is None:
            raise ValueError(f'send coadd_dim= for layout {layout}')

        if layout == 'grid':
            shifts = get_grid_shifts(
                rng=rng,
                dim=coadd_dim,
                buff=buff,
                pixel_scale=pixel_scale,
                spacing=GRID_SPACING,
            )
        elif layout == 'random':
            # area covered by objects
            if nobj is None:
                # in units of square arcmin
                area = ((coadd_dim - 2 * buff) * pixel_scale / 60)**2
                nobj_mean = max(area * RANDOM_DENSITY, 1)
                nobj = rng.poisson(nobj_mean)

            shifts = get_random_shifts(
                rng=rng,
                dim=coadd_dim,
                buff=buff,
                pixel_scale=pixel_scale,
                size=nobj,
            )
        elif layout == 'random_disk':
            # randomly distributed in a circle
            # area covered by objects
            if nobj is None:
                radius = (coadd_dim/2. - buff) * pixel_scale / 60.
                area = np.pi*radius**2
                nobj_mean = max(area * RANDOM_DENSITY, 1)
                nobj = rng.poisson(nobj_mean)

            shifts = get_random_disk_shifts(
                rng=rng,
                dim=coadd_dim,
                buff=buff,
                pixel_scale=pixel_scale,
                size=nobj,
            )
        elif layout == 'hex':
            shifts = get_hex_shifts(
                rng=rng,
                dim=coadd_dim,
                buff=buff,
                pixel_scale=pixel_scale,
                spacing=HEX_SPACING,
            )
        else:
            raise ValueError("bad layout: '%s'" % layout)

    return shifts


def get_hex_shifts(*, rng, dim, buff, pixel_scale, spacing):
    """
    get a set of hex grid shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    buff: int, optional
        Buffer region where no objects will be drawn.
    pixel_scale: float
        pixel scale
    spacing: float
        Spacing of the hexagonal lattice

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    from hexalattice.hexalattice import create_hex_grid

    width = (dim - 2*buff) * pixel_scale
    n_on_side = int(width / spacing) + 1

    nx = int(n_on_side * np.sqrt(2))
    # the factor of 0.866 makes sure the grid is square-ish
    ny = int(n_on_side * np.sqrt(2) / 0.8660254)

    # here the spacing between grid centers is 1
    hg, _ = create_hex_grid(nx=nx, ny=ny, rotate_deg=rng.uniform() * 360)

    # convert the spacing to right number of pixels
    # we also recenter the grid since it comes out centered at 0,0
    hg *= spacing
    upos = hg[:, 0].ravel()
    vpos = hg[:, 1].ravel()

    # dither
    upos += pixel_scale * rng.uniform(low=-0.5, high=0.5, size=upos.shape[0])
    vpos += pixel_scale * rng.uniform(low=-0.5, high=0.5, size=vpos.shape[0])

    pos_bounds = (-width/2, width/2)
    msk = (
        (upos >= pos_bounds[0])
        & (upos <= pos_bounds[1])
        & (vpos >= pos_bounds[0])
        & (vpos <= pos_bounds[1])
    )
    upos = upos[msk]
    vpos = vpos[msk]

    ntot = upos.shape[0]
    shifts = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])
    shifts["dx"] = upos
    shifts["dy"] = vpos

    return shifts


def get_grid_shifts(*, rng, dim, buff, pixel_scale, spacing):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    buff: int, optional
        Buffer region where no objects will be drawn.
    pixel_scale: float
        pixel scale
    spacing: float
        Spacing of the lattice

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    width = (dim - 2*buff) * pixel_scale
    n_on_side = int(dim / spacing * pixel_scale)

    ntot = n_on_side**2

    # ix/iy are really on the sky
    grid = spacing*(np.arange(n_on_side) - (n_on_side-1)/2)

    shifts = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])

    i = 0
    for ix in range(n_on_side):
        for iy in range(n_on_side):
            dx = grid[ix] + pixel_scale * rng.uniform(low=-0.5, high=0.5)
            dy = grid[iy] + pixel_scale * rng.uniform(low=-0.5, high=0.5)

            shifts['dx'][i] = dx
            shifts['dy'][i] = dy
            i += 1

    pos_bounds = (-width/2, width/2)
    msk = (
        (shifts['dx'] >= pos_bounds[0])
        & (shifts['dx'] <= pos_bounds[1])
        & (shifts['dy'] >= pos_bounds[0])
        & (shifts['dy'] <= pos_bounds[1])
    )
    shifts = shifts[msk]

    return shifts


def get_random_shifts(*, rng, dim, buff, pixel_scale, size):
    """
    get a set of random shifts in a square, with random shifts at the pixel
    scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    buff: int, optional
        Buffer region where no objects will be drawn.
    pixel_scale: float
        pixel scale
    size: int
        Number of objects to draw.

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    halfwidth = (dim - 2*buff)/2.0
    if halfwidth < 0:
        print(dim, buff, halfwidth)
        # prevent user using a buffer that is too large
        raise ValueError("dim - 2*buff < 0")

    low = -halfwidth * pixel_scale
    high = halfwidth * pixel_scale

    shifts = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])

    shifts['dx'] = rng.uniform(low=low, high=high, size=size)
    shifts['dy'] = rng.uniform(low=low, high=high, size=size)

    return shifts


def get_random_disk_shifts(*, rng, dim, buff, pixel_scale, size):
    """Gets a set of random shifts on a disk, with random shifts at the
    pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    buff: int, optional
        Buffer region where no objects will be drawn.
    pixel_scale: float
        pixel scale
    size: int
        Number of objects to draw.

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    radius = (dim - 2*buff) / 2.0 * pixel_scale
    if radius < 0:
        # prevent user using a buffer that is too large
        raise ValueError("dim - 2*buff < 0")
    radius_square = radius**2.

    # evenly distributed within a radius, min(nx, ny)*rfrac
    rarray = np.sqrt(radius_square*rng.rand(size))   # radius
    tarray = rng.uniform(0., 2*np.pi, size)   # theta (0, pi/nrot)

    shifts = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])
    shifts['dx'] = rarray*np.cos(tarray)
    shifts['dy'] = rarray*np.sin(tarray)
    return shifts


def get_pair_shifts(*, rng, sep, pixel_scale=SCALE):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    sep: float
        Separation of pair in arcsec
    pixel_scale: float
        pixel scale

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    shifts = np.zeros(2, dtype=[('dx', 'f8'), ('dy', 'f8')])

    angle = rng.uniform(low=0, high=np.pi)
    shift_radius = sep / 2

    xdither, ydither = pixel_scale * rng.uniform(low=-0.5, high=0.5, size=2)

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
