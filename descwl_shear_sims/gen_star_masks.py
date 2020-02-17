import numpy as np
from numba import njit
import esutil as eu
import ngmix


def add_star_and_bleed_masks(*,
                             rng,
                             mask,
                             wcs,
                             density=200,
                             pixel_scale=0.2,
                             radmean=3,
                             radstd=5,
                             radmin=3,
                             radmax=50,
                             bleed_length_fac=2,
                             mask_value=1):
    """
    make fake masks with

    - star mask radii distributed as a clipped log normal
    - bleed trail length a fixed factor larger than the diameter
      of the star mask
    - bleed widths either 1 or 2


    Parameters
    ----------
    mask: ndarray
        Integer mask image
    rng: numpy.RandomState
        Random number generator to use
    density: float
        Number of saturated stars per square degree
    pixel_scale: float
        pixel scale in arcsec/pixel
    radmean: float
        Mean radius for log normal distribution
    radstd: float
        Radius standard deviation for log normal distribution
    radmin: float
        Minimum radius of star masks
    radmax: float
        Maximum radius of star masks
    bleed_length_fac: float
        The bleed length is this factor times the *diameter* of the circular
        star mask
    mask_value: int
        Value to put in mask

    Returns
    -------
    nstars: int
        Number of stars added
    """

    dims = mask.shape

    x, y = get_locations_sqdeg(
        rng=rng,
        density=density,
        dims=dims,
        pixel_scale=pixel_scale,
    )

    rpdf = ngmix.priors.LogNormal(radmean, radstd, rng=rng)
    nstars = x.size
    for i in range(nstars):
        star_mask_rad = get_star_mask_rad(
            pdf=rpdf,
            radmin=radmin,
            radmax=radmax,
        )
        bleed_width = get_bleed_width(
            rng=rng,
            star_mask_rad=star_mask_rad,
        )
        bleed_length = get_bleed_length(
            star_mask_rad=star_mask_rad,
            bleed_length_fac=bleed_length_fac,
        )
        add_star_and_bleed(
            mask=mask,
            x=x[i],
            y=y[i],
            radius=star_mask_rad,
            bleed_width=bleed_width,
            bleed_length=bleed_length,
            value=mask_value,
        )

    return nstars


def add_star_and_bleed(*,
                       mask,
                       x, y,
                       radius,
                       bleed_width,
                       bleed_length,
                       value):

    add_star(
        mask=mask,
        x=x,
        y=y,
        radius=radius,
        value=value,
    )

    add_bleed(
        mask=mask,
        x=x,
        y=y,
        width=bleed_width,
        length=bleed_length,
        value=value,
    )


def get_star_mask_rad(*, pdf, radmin, radmax):

    while True:
        radius = pdf.sample()
        if radmin < radius < radmax:
            break

    print('radius:', radius)
    return radius


def get_bleed_width(*, rng, star_mask_rad):
    """
    width is always odd
    """

    if star_mask_rad > 10:
        return 3
    else:
        return 1


def get_bleed_length(*, star_mask_rad, bleed_length_fac):
    diameter = star_mask_rad*2
    length = int(bleed_length_fac*diameter)

    if length % 2 == 0:
        length -= 1

    if length < 0:
        length = 1

    return length


@njit
def add_star(*, mask, x, y, radius, value):

    radius2 = radius**2
    ny, nx = mask.shape

    for iy in range(ny):
        y2 = (y-iy)**2
        if y2 > radius2:
            continue

        for ix in range(nx):
            x2 = (x-ix)**2
            if x2 > radius2:
                continue

            rad = x2 + y2

            if rad > radius2:
                continue

            mask[iy, ix] |= value


@njit
def add_bleed(*, mask, x, y, length, width, value):
    """
    add odd width and odd length bleed
    """

    ny, nx = mask.shape

    xpad = (width-1)//2
    ypad = (length-1)//2

    xmin = x - xpad
    xmax = x + xpad

    ymin = y - ypad
    ymax = y + ypad

    for iy in range(ymin, ymax+1):
        if iy < 0 or iy > (ny-1):
            continue

        for ix in range(xmin, xmax+1):
            if ix < 0 or ix > (nx-1):
                continue
            mask[iy, ix] |= value


def get_locations_sqdeg(*, rng, density, dims, pixel_scale):

    area_degrees = (
        dims[0]*dims[1] * pixel_scale**2/3600.0**2
    )
    assert area_degrees < 1

    count = rng.poisson(lam=density)
    x, y = _get_locations_sqdeg(
        rng=rng,
        count=count,
        dims=dims,
        pixel_scale=pixel_scale,
    )
    w, = np.where(
        (x >= 0) & (x < dims[1]) &
        (y >= 0) & (y < dims[0])
    )
    print('found:', w.size)
    x = x[w]
    y = y[w]
    return x, y


def _get_locations_sqdeg(*, rng, count, dims, pixel_scale):

    y = _get_locations_deg(
        rng=rng,
        count=count,
        dim=dims[0],
        pixel_scale=pixel_scale,
    )
    x = _get_locations_deg(
        rng=rng,
        count=count,
        dim=dims[1],
        pixel_scale=pixel_scale,
    )

    return x, y


def _get_locations_deg(*, rng, count, dim, pixel_scale):

    one_deg_pix = 1.0*3600/pixel_scale
    cen = (dim-1.0)/2.0
    min_fake_pix = cen - one_deg_pix/2
    max_fake_pix = cen + one_deg_pix/2

    vals = rng.uniform(
        low=min_fake_pix,
        high=max_fake_pix,
        size=count,
    )

    return vals.astype('i4')


def show_mask(*, mask):
    import images
    images.view(mask)


def dotest(ntry=1000):
    rng = np.random.RandomState()

    pixel_scale = 0.2

    # 1 arcmin
    size_arcmin = 1.0
    dims = [int(size_arcmin*60/pixel_scale)]*2

    mask = np.zeros(dims, dtype='i4')

    nwith = 0
    for i in range(ntry):
        mask[:, :] = 0

        nstars = add_star_and_bleed_masks(
            mask=mask,
            rng=rng,
            pixel_scale=pixel_scale,
        )
        if nstars > 0:
            show_mask(mask=mask)
            import fitsio
            fitsio.write('test.fits', mask, clobber=True)
            if input('hit a key (q to quit): ') == 'q':
                return

    frac = nwith/ntry
    print('frac with star: %d/%d %g' % (nwith, ntry, frac))


if __name__ == '__main__':
    dotest()
