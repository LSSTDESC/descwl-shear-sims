import copy
import galsim
import numpy as np
from .se_obs import SEObs

SCALE = 0.2
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)

GRID_N_ON_SIDE = 6
RANDOM_DENSITY = 80  # per square arcmin

DEFAULT_TRIVIAL_SIM_CONFIG = {
    'psf_dim': 51,
    'coadd_dim': 351,
    'buff': 50,
    'layout': 'grid',
    'dither': False,
    'rotate': False,
    'bands': ['i'],
    'epochs_per_band': 1,
}


def make_trivial_sim(
    *,
    rng,
    noise,
    coadd_dim,
    buff,
    layout,
    g1,
    g2,
    psf_dim=51,
    dither=False,
    rotate=False,
    bands=['i'],
    epochs_per_band=1,
):
    """
    Make simulation data

    Parameters
    ----------
    rng: numpy.random.RandomState
        Numpy random state
    noise: float
        Noise level for images
    coadd_dim: int
        Default 351
    buff: int
        Buffer region where no objects will be drawn, default 50
    layout: string
        'grid' or 'random'
    g1: float
        Shear g1 for galaxies
    g2: float
        Shear g2 for galaxies

    """
    offsets = get_offsets(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout=layout,
    )

    galaxy_catalog = FixedGalaxyCatalog(offsets=offsets, g1=g1, g2=g2)
    psf = galsim.Gaussian(fwhm=0.8)

    noise_per_epoch = noise*np.sqrt(epochs_per_band)

    se_dim = (
        int(np.ceil(coadd_dim * np.sqrt(2))) + 20
    )

    band_data = {}
    for band in bands:
        all_obj = galaxy_catalog.get_objects(band=band)
        seobj_list = []
        for epoch in range(epochs_per_band):
            seobs = make_seobs(
                rng=rng,
                noise=noise_per_epoch,
                all_obj=all_obj,
                dim=se_dim,
                psf=psf,
                psf_dim=psf_dim,
                dither=dither,
                rotate=rotate,
            )
            seobj_list.append(seobs)

        band_data[band] = seobj_list

    coadd_wcs = make_coadd_wcs(coadd_dim)

    return {
        'band_data': band_data,
        'coadd_wcs': coadd_wcs,
        'psf_dims': [psf_dim]*2,
        'coadd_dims': [coadd_dim]*2,
    }


class FixedGalaxyCatalog(object):
    """
    Galaxies of fixed galsim type, flux, and size

    Same for all bands
    """
    def __init__(self, *, offsets, g1, g2):
        self.offsets = offsets
        self.g1 = g1
        self.g2 = g2

    def get_objects(self, *, band):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        list of galsim objects
        """
        objlist = []
        for i in range(self.offsets.size):
            obj = galsim.Exponential(
                half_light_radius=0.5,
            ).shift(
                dx=self.offsets['dx'][i],
                dy=self.offsets['dy'][i]
            )
            objlist.append(obj)

        all_obj = galsim.Add(objlist)
        all_obj = all_obj.shear(g1=self.g1, g2=self.g2)
        return all_obj


def make_wcs(*, scale, image_origin, world_origin, theta=None):
    mat = np.array(
        [[scale, 0.0],
         [0.0, scale]],
    )
    if theta is not None:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        rot = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]],
        )
        mat = np.dot(mat, rot)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def make_coadd_wcs(coadd_dim):
    coadd_dims = [coadd_dim]*2
    coadd_cen = (np.array(coadd_dims)-1)/2
    coadd_origin = galsim.PositionD(x=coadd_cen[1], y=coadd_cen[0])
    return make_wcs(
        scale=SCALE,
        image_origin=coadd_origin,
        world_origin=WORLD_ORIGIN,
    )


def get_offsets(
    *,
    rng,
    coadd_dim,
    buff,
    layout,
):
    """
    make position offsets for objects

    rng: numpy.random.RandomState
        Numpy random state
    coadd_dim: int
        Dimensions of final coadd
    buff: int
        Buffer region where no objects will be drawn
    layout: string
        'grid' or 'random'
    """

    if layout == 'grid':
        offsets = get_grid_offsets(
            rng=rng,
            dim=coadd_dim,
            n_on_side=GRID_N_ON_SIDE,
        )
    elif layout == 'random':
        # area covered by objects
        area = ((coadd_dim - 2*buff)*SCALE/60)**2
        nobj_mean = area * RANDOM_DENSITY
        nobj = rng.poisson(nobj_mean)
        offsets = get_random_offsets(
            rng=rng,
            dim=coadd_dim,
            buff=buff,
            size=nobj,
        )
    else:
        raise ValueError("bad layout: '%s'" % layout)

    return offsets


def get_grid_offsets(*, rng, dim, n_on_side):
    """
    get a set of gridded offsets, with random offsets at the pixel scale

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
    offsets: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    spacing = dim/(n_on_side+1)*SCALE

    ntot = n_on_side**2

    # ix/iy are really on the sky
    grid = spacing*(np.arange(n_on_side) - (n_on_side-1)/2)

    offsets = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])

    i = 0
    for ix in range(n_on_side):
        for iy in range(n_on_side):
            dx = grid[ix] + SCALE*rng.uniform(low=-0.5, high=0.5)
            dy = grid[iy] + SCALE*rng.uniform(low=-0.5, high=0.5)

            offsets['dx'][i] = dx
            offsets['dy'][i] = dy
            i += 1

    return offsets


def get_random_offsets(*, rng, dim, buff, size):
    """
    get a set of gridded offsets, with random offsets at the pixel scale

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
    offsets: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    halfwidth = (dim - 2*buff)/2.0

    low = -halfwidth*SCALE
    high = halfwidth*SCALE

    offsets = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])

    offsets['dx'] = rng.uniform(low=low, high=high, size=size)
    offsets['dy'] = rng.uniform(low=low, high=high, size=size)

    return offsets


class FixedPSF(object):
    def __init__(self, *, psf, offset, psf_dim, wcs):
        self._psf = psf
        self._offset = offset
        self._psf_dim = psf_dim
        self._wcs = wcs

    def __call__(self, *, x, y, center_psf, get_offset=False):
        """
        center_psf is ignored
        """
        image_pos = galsim.PositionD(x=x, y=y)

        offset = copy.deepcopy(self._offset)

        if center_psf:
            print("ignoring request to center psf, using internal offset")

        gsimage = self._psf.drawImage(
            nx=self._psf_dim,
            ny=self._psf_dim,
            offset=offset,
            wcs=self._wcs.local(image_pos=image_pos),
        )
        if get_offset:
            return gsimage, offset
        else:
            return gsimage


def make_seobs(
    *,
    rng,
    noise,
    all_obj,
    dim,
    psf,
    psf_dim,
    dither=False,
    rotate=False,
):
    """
    make a grid sim with trivial pixel scale, fixed sized
    exponentials and gaussian psf

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    noise: float
        Gaussian noise level
    g1: float
        Shear g1
    g2: float
        Shear g2
    dither: bool
        If set to True, dither randomly by a pixel width
    rotate: bool
        If set to True, rotate the image randomly
    """

    dims = [dim]*2
    cen = (np.array(dims)-1)/2

    se_origin = galsim.PositionD(x=cen[1], y=cen[0])
    if dither:
        dither_range = 0.5
        off = rng.uniform(low=-dither_range, high=dither_range, size=2)
        offset = galsim.PositionD(x=off[0], y=off[1])
        se_origin = se_origin + offset
    else:
        offset = None

    if rotate:
        theta = rng.uniform(low=0, high=2*np.pi)
    else:
        theta = None

    se_wcs = make_wcs(
        scale=SCALE,
        theta=theta,
        image_origin=se_origin,
        world_origin=WORLD_ORIGIN,
    )

    # we want to fit them into the coadd region
    convolved_objects = galsim.Convolve(all_obj, psf)

    # everything gets shifted by the dither offset
    image = convolved_objects.drawImage(
        nx=dim,
        ny=dim,
        wcs=se_wcs,
        offset=offset,
    )
    weight = image.copy()
    weight.array[:, :] = 1.0/noise**2

    image.array[:, :] += rng.normal(scale=noise, size=dims)
    noise_image = image.copy()
    noise_image.array[:, :] = rng.normal(scale=noise, size=dims)

    bmask = galsim.Image(
        np.zeros(dims, dtype='i4'),
        bounds=image.bounds,
        wcs=image.wcs,
        dtype=np.int32,
    )

    psf_obj = FixedPSF(psf=psf, offset=offset, psf_dim=psf_dim, wcs=se_wcs)

    return SEObs(
        image=image,
        noise=noise_image,
        weight=weight,
        wcs=se_wcs,
        psf_function=psf_obj,
        bmask=bmask,
    )


def get_trivial_sim_config(config=None):
    out_config = copy.deepcopy(DEFAULT_TRIVIAL_SIM_CONFIG)

    if config is not None:
        out_config.update(config)
    return out_config
