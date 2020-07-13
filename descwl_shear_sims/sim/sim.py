import os
from numba import njit
import copy
import galsim
import numpy as np
import descwl
from ..se_obs import SEObs
from ..cache_tools import cached_catalog_read
from ..ps_psf import PowerSpectrumPSF
from ..stars import load_sample_stars, sample_star_density, get_star_mag
from ..saturation import saturate_image_and_mask, BAND_SAT_VALS
from ..lsst_bits import BAD_COLUMN, COSMIC_RAY, SAT, BRIGHT
from ..gen_star_masks import add_bright_star_mask
from ..gen_masks import (
    generate_basic_mask, generate_cosmic_rays, generate_bad_columns,
)
from ..star_bleeds import add_bleed, get_max_mag_with_bleed

PSF_FWHM = 0.8
MOFFAT_BETA = 2.5
SCALE = 0.2
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)

GRID_N_ON_SIDE = 6
RANDOM_DENSITY = 80  # per square arcmin
ZERO_POINT = 30.0

DEFAULT_SIM_CONFIG = {
    "gal_type": "exp",
    "psf_type": "gauss",
    "psf_dim": 51,
    "coadd_dim": 351,
    "buff": 50,
    "layout": "grid",
    "dither": False,
    "rotate": False,
    "bands": ["i"],
    "epochs_per_band": 1,
    "noise_factor": 1.0,
    "stars": False,
    "star_bleeds": False,
    "cosmic_rays": False,
    "bad_columns": False,
}

DEFAULT_FIXED_GAL_CONFIG = {
    "mag": 17.0,
    "hlr": 0.5,
}


def make_sim(
    *,
    rng,
    galaxy_catalog,
    coadd_dim,
    g1,
    g2,
    psf,
    star_catalog=None,
    psf_dim=51,
    dither=False,
    rotate=False,
    bands=['i'],
    epochs_per_band=1,
    noise_factor=1.0,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
):
    """
    Make simulation data

    Parameters
    ----------
    rng: numpy.random.RandomState
        Numpy random state
    galaxy_catalog: catalog
        E.g. WLDeblendGalaxyCatalog or FixedGalaxyCatalog
    coadd_dim: int
        Default 351
    g1: float
        Shear g1 for galaxies
    g2: float
        Shear g2 for galaxies
    psf: GSObject or PowerSpectrumPSF
        The psf object or power spectrum psf
    star_catalog: catalog
        e.g. StarCatalog
    psf_dim: int, optional
        Dimensions of psf image.  Default 51
    dither: bool, optional
        Whether to dither the images at the pixel level, default False
    rotate: bool, optional
        Whether to rotate the images randomly, default False
    bands: list, optional
        Default ['i']
    epochs_per_band: int, optional
        Number of epochs per band
    noise_factor: float, optional
        Factor by which to multiply the noise, default 1
    cosmic_rays: bool
        If True, add cosmic rays
    bad_columns: bool
        If True, add bad columns
    """

    se_dim = get_se_dim(coadd_dim=coadd_dim)

    band_data = {}
    for band in bands:

        survey = get_survey(gal_type=galaxy_catalog.gal_type, band=band)
        noise_per_epoch = survey.noise*np.sqrt(epochs_per_band)*noise_factor

        # go down to coadd depth in this band, not dividing by sqrt(nbands)
        # but we could if we want to be more conservative
        mask_threshold = survey.noise*noise_factor

        objlist, shifts = galaxy_catalog.get_objlist(
            survey=survey,
            g1=g1,
            g2=g2,
        )
        objlist, shifts, bright_objlist, bright_shifts, bright_mags = get_objlist(
            galaxy_catalog=galaxy_catalog,
            survey=survey,
            g1=g1,
            g2=g2,
            star_catalog=star_catalog,
            noise=noise_per_epoch,
        )

        seobs_list = []
        for epoch in range(epochs_per_band):
            seobs = make_seobs(
                rng=rng,
                band=band,
                noise=noise_per_epoch,
                objlist=objlist,
                shifts=shifts,
                dim=se_dim,
                psf=psf,
                psf_dim=psf_dim,
                dither=dither,
                rotate=rotate,
                bright_objlist=bright_objlist,
                bright_shifts=bright_shifts,
                bright_mags=bright_mags,
                mask_threshold=mask_threshold,
                cosmic_rays=cosmic_rays,
                bad_columns=bad_columns,
                star_bleeds=star_bleeds,
            )
            if galaxy_catalog.gal_type == 'wldeblend':
                rescale_wldeblend_images(
                    survey=survey.descwl_survey,
                    image=seobs.image,
                    noise=seobs.noise,
                    weight=seobs.weight,
                )

            # mark high pixels SAT and also set sat value in image for
            # any pixels already marked SAT
            saturate_image_and_mask(
                image=seobs.image.array,
                bmask=seobs.bmask.array,
                sat_val=BAND_SAT_VALS[band],
            )

            seobs_list.append(seobs)

        band_data[band] = seobs_list

    coadd_wcs = make_coadd_wcs(coadd_dim)

    return {
        'band_data': band_data,
        'coadd_wcs': coadd_wcs,
        'psf_dims': [psf_dim]*2,
        'coadd_dims': [coadd_dim]*2,
    }


def get_survey(*, gal_type, band):
    """
    Get a survey object

    Parameters
    ----------
    gal_type: string
        'wldeblend' or 'exp'
    band: string
        e.g. 'r'

    Returns
    -------
    For gal_type 'wldeblend' returns a WLDeblendSurvey object,
    for gal_type 'exp' returns BasicSurvey.
    """
    if gal_type == 'wldeblend':
        survey = WLDeblendSurvey(band=band)
    elif gal_type in ['exp']:
        survey = BasicSurvey(band=band)
    else:
        raise ValueError("bad gal_type: '%s'" % gal_type)

    return survey


def get_objlist(*, galaxy_catalog, survey, g1, g2, star_catalog=None, noise=None):
    """
    get the objlist and shifts, possibly combining the galaxy catalog
    with a star catalog

    Parameters
    ----------
    galaxy_catalog: catalog
        e.g. WLDeblendGalaxyCatalog
    survey: descwl Survey
        For the appropriate band
    g1: float
        Shear for galaxies
    g2: float
        Shear for galaxies
    star_catalog: catalog
        e.g. StarCatalog
    noise: float
        Needed for star catalog

    Returns
    -------
    objlist, shifts
        objlist is a list of galsim GSObject with transformations applied. Shifts
        is an array with fields dx and dy for each object
    """
    objlist, shifts = galaxy_catalog.get_objlist(
        survey=survey,
        g1=g1,
        g2=g2,
    )

    if star_catalog is not None:
        assert noise is not None
        res = star_catalog.get_objlist(
            survey=survey, noise=noise,
        )
        sobjlist, sshifts, bright_objlist, bright_shifts, bright_mags = res

        objlist = objlist + sobjlist

        shifts = np.hstack((shifts, sshifts))
    else:
        bright_objlist = None
        bright_shifts = None
        bright_mags = None

    return objlist, shifts, bright_objlist, bright_shifts, bright_mags


def make_galaxy_catalog(
    *,
    rng,
    gal_type,
    coadd_dim,
    buff,
    layout=None,
    gal_config=None,
):
    """
    rng: numpy.random.RandomState
        Numpy random state
    gal_type: string
        'exp' or 'wldeblend'
    coadd_dim: int
        Dimensions of coadd
    buff: int
        Buffer around the edge where no objects are drawn
    layout: string, optional
        'grid' or 'random'.  Ignored for gal_type "wldeblend", otherwise
        required.
    gal_config: dict or None
        Can be sent for fixed galaxy catalog.  See DEFAULT_FIXED_GAL_CONFIG
        for defaults
    """
    if gal_type == 'wldeblend':
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
        )
    else:

        if layout is None:
            raise ValueError("send layout= for gal_type '%s'" % gal_type)

        gal_config = get_fixed_gal_config(config=gal_config)
        galaxy_catalog = FixedGalaxyCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout=layout,
            mag=gal_config['mag'],
            hlr=gal_config['hlr'],
        )

    return galaxy_catalog


def make_wcs(*, scale, image_origin, world_origin, theta=None):
    """
    make and return a wcs object

    Parameters
    ----------
    scale: float
        Pixel scale
    image_origin: galsim.PositionD
        Image origin position
    world_origin: galsim.CelestialCoord
        Origin on the sky
    theta: float, optional
        Rotation angle

    Returns
    -------
    A galsim wcs object, currently a TanWCS
    """
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
    """
    make a coadd wcs, using the default world origin

    Parameters
    ----------
    coadd_dim: int
        dimensions of the coadd

    Returns
    --------
    A galsim wcs, see make_wcs for return type
    """
    coadd_dims = [coadd_dim]*2
    coadd_cen = (np.array(coadd_dims)-1)/2
    coadd_origin = galsim.PositionD(x=coadd_cen[1], y=coadd_cen[0])
    return make_wcs(
        scale=SCALE,
        image_origin=coadd_origin,
        world_origin=WORLD_ORIGIN,
    )


def get_shifts(
    *,
    rng,
    coadd_dim,
    buff,
    layout,
    nobj=None,
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
    """

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


class FixedPSF(object):
    """
    A simple fixed PSF object

    Parameters
    ----------
    psf: galsim.GSObject
        The psf object
    offset: galsim.PositionD
        Should match the offset of the image thei psf corresponds to
    psf_dim: int
        The dimension of the PSF image that will be created
    wcs: galsim WCS
        E.g. a wcs returned by make_wcs
    """
    def __init__(self, *, psf, offset, psf_dim, wcs):
        self._psf = psf
        self._offset = offset
        self._psf_dim = psf_dim
        self._wcs = wcs

    def __call__(self, *, x, y, center_psf, get_offset=False):
        """
        center_psf is ignored ,just there for compatibility

        Parameters
        ----------
        x: float
            x image position
        y: float
            y image position
        cener_psf: bool
            Center the psf, this is ignored
        get_offset: bool, optional
            If True, return the offset

        Returns
        -------
        A galsim Image, and optionally the offset as a PositonD
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
            if offset is None:
                offset = galsim.PositionD(x=0.0, y=0.0)
            return gsimage, offset
        else:
            return gsimage


def make_seobs(
    *,
    rng,
    band,
    noise,
    objlist,
    shifts,
    dim,
    psf,
    psf_dim,
    dither=False,
    rotate=False,
    bright_objlist=None,
    bright_shifts=None,
    bright_mags=None,
    mask_threshold=None,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
):
    """
    Make an SEObs

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    noise: float
        Gaussian noise level
    objlist: list
        List of GSObj
    shifts: array
        Array with fields dx and dy, which are du, dv offsets
        in sky coords.
    dim: int
        Dimension of image
    psf: GSObject or PowerSpectrumPSF
        the psf
    psf_dim: int
        Dimensions of psf image that will be drawn when psf func is called
    dither: bool
        If set to True, dither randomly by a pixel width
    rotate: bool
        If set to True, rotate the image randomly
    cosmic_rays: bool
        If True, put in cosmic rays
    bad_columns: bool
        If True, put in bad columns
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

    convolved_objects, psf_gsobj = get_convolved_objects(
        objlist=objlist,
        psf=psf,
        shifts=shifts,
        offset=offset,
        se_wcs=se_wcs,
        se_origin=se_origin,
    )

    objects = galsim.Add(convolved_objects)

    # everything gets shifted by the dither offset
    image = objects.drawImage(
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

    bmask = get_bmask(
        image=image,
        rng=rng,
        cosmic_rays=cosmic_rays,
        bad_columns=bad_columns,
    )

    if bright_objlist is not None:
        timage = image.copy()

        assert bright_shifts is not None
        assert bright_mags is not None
        assert mask_threshold is not None

        bright_convolved_objects, _ = get_convolved_objects(
            objlist=bright_objlist,
            psf=psf,
            shifts=bright_shifts,
            offset=offset,
            se_wcs=se_wcs,
            se_origin=se_origin,
        )
        for i, obj in enumerate(bright_convolved_objects):

            obj.drawImage(
                image=timage,
                offset=offset,
                method='phot',
                n_photons=10_000_000,
                maxN=1_000_000,
                rng=galsim.BaseDeviate(rng.randint(0, 2**30)),
            )

            image += timage

            calculate_and_add_bright_star_mask(
                image=timage.array,
                bmask=bmask.array,
                shift=bright_shifts[i],
                wcs=se_wcs,
                origin=se_origin,
                threshold=mask_threshold,
            )

    if star_bleeds and bright_objlist is not None:
        # add bleeds at bright star locations if they are saturated
        add_bleeds(
            image=image,
            origin=se_origin,
            bmask=bmask,
            shifts=bright_shifts,
            mags=bright_mags,
            band=band,
        )

    psf_obj = FixedPSF(psf=psf_gsobj, offset=offset, psf_dim=psf_dim, wcs=se_wcs)

    return SEObs(
        image=image,
        noise=noise_image,
        weight=weight,
        wcs=se_wcs,
        psf_function=psf_obj,
        bmask=bmask,
    )


def add_bleeds(*, image, origin, bmask, shifts, mags, band):
    """
    Add a bleed for each saturated object

    Parameters
    ----------
    image: galsim Image
        Image will be modified to have saturated values in the
        bleed
    origin: galsim.PositionD
        Origin of image in pixels
    bmask: galsim Image
        Mask will be modified to have saturated values in the
        bleed
    shifts: array
        Fields dx and dy.
    mags: list
        List of mags
    band: string
        Filter band

    Returns
    --------
    None
    """

    wcs = image.wcs

    jac_wcs = wcs.jacobian(world_pos=wcs.center)
    max_mag = get_max_mag_with_bleed(band=band)

    for i in range(shifts.size):
        mag = mags[i]
        if mag < max_mag:
            shift_pos = galsim.PositionD(
                x=shifts['dx'][i],
                y=shifts['dy'][i],
            )
            pos = jac_wcs.toImage(shift_pos) + origin

            add_bleed(
                image=image.array,
                bmask=bmask.array,
                pos=pos,
                mag=mag,
                band=band,
            )


def get_bmask(*, image, rng, cosmic_rays, bad_columns):
    """
    get a bitmask for the image, including EDGE and
    optional cosmic rays and bad columns

    Parameters
    ----------
    image: galsim.Image type
        The image
    rng: np.random.RandomState
        The random state object
    cosmic_rays: bool
        Whether to add cosmic rays
    bad_columns: bool
        Whether to add bad columns

    Returns
    -------
    galsim.Image of type int32
    """
    shape = image.array.shape

    mask = generate_basic_mask(shape=shape, edge_width=5)

    if cosmic_rays:

        # bool mask
        c_mask = generate_cosmic_rays(
            shape=shape,
            rng=rng,
            mean_cosmic_rays=1,
        )
        mask[c_mask] |= COSMIC_RAY + SAT

        # wait to do this later
        # image.array[cmask] = BAND_SAT_VALS[band]

    if bad_columns:
        # bool mask
        bc_msk = generate_bad_columns(
            shape=shape,
            rng=rng,
            mean_bad_cols=1,
        )
        mask[bc_msk] |= BAD_COLUMN
        image.array[bc_msk] = 0.0

    return galsim.Image(
        mask,
        bounds=image.bounds,
        wcs=image.wcs,
        dtype=np.int32,
    )


def get_convolved_objects(*, objlist, psf, shifts, offset, se_wcs, se_origin):
    """
    get a list of convolved objects

    Parameters
    ----------
    objlist: list of GSObject
        The list of objects to convolve
    psf: GSObject or PowerSpectrumPSF
        The PSF for convolution
    shifts: list of shifts for each object
        Only used for the spatially variable power spectrum psf
    se_wcs: galsim WCS
        Only used for the spatially variable power specrum psf
    se_origin: galsim.PositionD
        Origin, shifts are relative to this origin

    Returns
    -------
    list of convolved GSObject
    """
    if isinstance(psf, galsim.GSObject):
        convolved_objects = [galsim.Convolve(obj, psf) for obj in objlist]
        psf_gsobj = psf
    else:
        convolved_objects = get_convolved_objlist_variable_psf(
            objlist=objlist,
            shifts=shifts,
            psf=psf,
            wcs=se_wcs,
            origin=se_origin,
        )
        psf_gsobj = psf.getPSF(se_origin)

    return convolved_objects, psf_gsobj


def rescale_wldeblend_images(*, survey, image, noise, weight):
    """
    Rescale wldeblend images noise and weight to our zero point

    Parameters
    ----------
    survey: WLDeblendSurvey
        The survey object
    image: galsim.Image or ndarray
        The image to be scaled
    noise: galsim.Image or ndarray
        The noise image to be scaled
    weight: galsim.Image or ndarray
        The weightimage to be scaled

    Returns
    -------
    None
    """
    fac = get_wldeblend_rescale_fac(survey)
    wfac = 1.0/fac**2

    image *= fac
    noise *= fac
    weight *= wfac


def get_wldeblend_rescale_fac(survey):
    """
    Get the factor to rescale wldeblend images to our zero point

    Parameters
    ----------
    survey: WLDeblendSurvey
        The survey object

    Returns
    -------
    number by which to rescale images
    """
    s_zp = survey.zero_point
    s_et = survey.exposure_time
    return 10.0**(0.4*(ZERO_POINT - 24.0))/s_zp/s_et


def get_convolved_objlist_variable_psf(
    *,
    objlist,
    shifts,
    psf,
    wcs,
    origin,  # pixel origin
):
    """
    Get a list of psf convolved objects for a variable psf

    Parameters
    ----------
    objlist: list
        List of GSObject
    shifts: array
        Array with fields dx and dy, which are du, dv offsets
        in sky coords.
    psf: PowerSpectrumPSF
        See ps_psf
    wcs: galsim wcs
        For the SE image
    origin: galsim.PositionD
        Origin of SE image (with offset included)
    """

    jac_wcs = wcs.jacobian(world_pos=wcs.center)

    new_objlist = []
    for i, obj in enumerate(objlist):
        shift_pos = galsim.PositionD(
            x=shifts['dx'][i],
            y=shifts['dy'][i],
        )
        pos = jac_wcs.toImage(shift_pos) + origin

        psf_gsobj = psf.getPSF(pos)

        obj = galsim.Convolve(obj, psf_gsobj)

        new_objlist.append(obj)

    return new_objlist


def calculate_and_add_bright_star_mask(
    *,
    image,
    bmask,
    shift,
    wcs,
    origin,  # pixel origin
    threshold,
):
    """
    Get a list of psf convolved objects for a variable psf

    Parameters
    ----------
    image: array
        numpy array representing the image
    bmask: array
        numpy array representing the bitmask
    shift: array
        scalar array with fields dx and dy, which are du, dv offsets in sky
        coords.
    wcs: galsim wcs
        For the SE image
    origin: galsim.PositionD
        Origin of SE image (with offset included)
    threshold: float
        The mask will extend to where the profile reaches this value
    """

    jac_wcs = wcs.jacobian(world_pos=wcs.center)

    shift_pos = galsim.PositionD(
        x=shift['dx'],
        y=shift['dy'],
    )
    pos = jac_wcs.toImage(shift_pos) + origin

    radius = calculate_bright_star_mask_radius(
        image=image,
        objrow=pos.y,
        objcol=pos.x,
        threshold=threshold,
    )
    add_bright_star_mask(
        bmask=bmask,
        x=pos.x,
        y=pos.y,
        radius=radius,
        val=BRIGHT,
    )


@njit
def calculate_bright_star_mask_radius(*, image, objrow, objcol, threshold):
    """
    get the radius at which the profile drops to the specified threshold

    Parameters
    ----------
    image: 2d array
        The image
    objrow: float
        The row position of the object center
    objcol: float
        The column position of the object center
    threshold: float
        The mask will extend to where the profile reaches this value

    Returns
    -------
    radius: float
        The radius
    """

    nrows, ncols = image.shape
    radius2 = 0.0

    for row in range(nrows):
        row2 = (objrow - row)**2

        for col in range(ncols):
            col2 = (objcol - col)**2

            tradius2 = row2 + col2
            if tradius2 < radius2:
                # we are already within a previously calculated radius
                continue

            val = image[row, col]
            if val > threshold:
                radius2 = tradius2

    radius = np.sqrt(radius2)
    return radius


def get_sim_config(config=None):
    """
    Get a simulation configuration, with defaults that can
    be over-ridden by the input.  The defaults are in
    DEFAULT_SIM_CONFIG

    Parameters
    ----------
    config: dict, optional
        Dict of options to over ride the defaults

    Returns
    -------
    config dict
    """
    out_config = copy.deepcopy(DEFAULT_SIM_CONFIG)
    sub_configs = ['gal_config']

    if config is not None:
        for key in config:
            if key not in out_config and key not in sub_configs:
                raise ValueError("bad key for sim: '%s'" % key)
        out_config.update(config)
    return out_config


def get_fixed_gal_config(config=None):
    """
    get the configuration for fixed galaxies, with defaults in place

    Parameters
    ----------
    config: dict, optional
        The input config. Over-rides defaults

    Returns
    -------
    the config dict
    """
    out_config = copy.deepcopy(DEFAULT_FIXED_GAL_CONFIG)

    if config is not None:
        for key in config:
            if key not in out_config:
                raise ValueError("bad key for fixed gals: '%s'" % key)
        out_config.update(config)
    return out_config


class WLDeblendSurvey(object):
    """
    wrapper for wldeblend surveys

    Parameters
    ----------
    band: string
        The band, e.g. 'r'
    """
    def __init__(self, *, band):

        pars = descwl.survey.Survey.get_defaults(
            survey_name='LSST',
            filter_band=band,
        )

        pars['survey_name'] = 'LSST'
        pars['filter_band'] = band
        pars['pixel_scale'] = SCALE

        # note in the way we call the descwl package, the image width
        # and height is not actually used
        pars['image_width'] = 10
        pars['image_height'] = 10

        # some versions take in the PSF and will complain if it is not
        # given
        try:
            svy = descwl.survey.Survey(**pars)
        except Exception:
            pars['psf_model'] = None
            svy = descwl.survey.Survey(**pars)

        self.noise = np.sqrt(svy.mean_sky_level)

        self.descwl_survey = svy

    @property
    def filter_band(self):
        """
        get the filter band for this survey

        Returns
        -------
        string filter band, e.g. 'r'
        """
        return self.descwl_survey.filter_band

    def get_flux(self, mag):
        """
        convert mag to flux
        """
        return self.descwl_survey.get_flux(mag)


class BasicSurvey(object):
    """
    represent a simple survey with common interface

    Parameters
    ----------
    band: string
        e.g. 'r'
    """
    def __init__(self, *, band):
        self.band = band
        self.noise = 1.0
        self.filter_band = band

    def get_flux(self, mag):
        """
        get the flux for the input mag using the standard zero point
        """
        return 10**(0.4 * (ZERO_POINT - mag))


class FixedGalaxyCatalog(object):
    """
    Galaxies of fixed galsim type, flux, and size

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        dimensions of the coadd
    buff: int
        Buffer region with no objects, on all sides of image
    layout: string
        The layout of objects, either 'grid' or 'random'
    mag: float
        Magnitude of all objects
    hlr: float
        Half light radius of all objects
    """
    def __init__(self, *, rng, coadd_dim, buff, layout, mag, hlr):
        self.gal_type = 'exp'
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.shifts = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout=layout,
        )

    def get_objlist(self, *, survey, g1, g2):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects]
        """

        flux = survey.get_flux(self.mag)

        num = self.shifts.size
        objlist = [
            self._get_galaxy(i, flux).shear(g1=g1, g2=g2)
            for i in range(num)
        ]

        shifts = self.shifts.copy()
        return objlist, shifts

    def _get_galaxy(self, i, flux):
        """
        get a galaxy object

        Parameters
        ----------
        i: int
            Index of object
        flux: float
            Flux of object

        Returns
        --------
        galsim.GSObject
        """
        return galsim.Exponential(
            half_light_radius=self.hlr,
            flux=flux,
        ).shift(
            dx=self.shifts['dx'][i],
            dy=self.shifts['dy'][i]
        )


class WLDeblendGalaxyCatalog(object):
    """
    Catalog of galaxies from wldeblend

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int
        Buffer region with no objects, on all sides of image
    """
    def __init__(self, *, rng, coadd_dim, buff):
        self.gal_type = 'wldeblend'
        self.rng = rng

        self._wldeblend_cat = read_wldeblend_cat(rng)

        # one square degree catalog, convert to arcmin
        gal_dens = self._wldeblend_cat.size / (60 * 60)
        area = ((coadd_dim - 2*buff)*SCALE/60)**2
        nobj_mean = area * gal_dens
        nobj = rng.poisson(nobj_mean)

        self.shifts = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
            nobj=nobj,
        )

        num = self.shifts.size
        self.indices = self.rng.randint(
            0,
            self._wldeblend_cat.size,
            size=num,
        )

        self.angles = self.rng.uniform(low=0, high=360, size=num)

    def get_objlist(self, *, survey, g1, g2):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object
        g1: float
            The g1 shear to apply to these objects
        g2: float
            The g2 shear to apply to these objects

        Returns
        -------
        [galsim objects]
        """

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        num = self.shifts.size

        band = survey.filter_band

        # object is already shifted, so this results in the scene
        # being sheared
        objlist = [
            self._get_galaxy(builder, band, i).shear(g1=g1, g2=g2)
            for i in range(num)
        ]

        shifts = self.shifts.copy()
        return objlist, shifts

    def _get_galaxy(self, builder, band, i):
        """
        Get a galaxy

        Parameters
        ----------
        builder: descwl.model.GalaxyBuilder
            Builder for this object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object

        Returns
        -------
        galsim.GSObject
        """
        index = self.indices[i]
        dx = self.shifts['dx'][i]
        dy = self.shifts['dy'][i]

        angle = self.angles[i]

        galaxy = builder.from_catalog(
            self._wldeblend_cat[index],
            0,
            0,
            band,
        ).model.rotate(
            angle * galsim.degrees,
        ).shift(
            dx=dx,
            dy=dy,
        )

        return galaxy


def read_wldeblend_cat(rng):
    """
    Read the catalog from the cache, but update the position angles each time

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator

    Returns
    -------
    array with fields
    """
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )

    # not thread safe
    cat = cached_catalog_read(fname)

    cat['pa_disk'] = rng.uniform(
        low=0.0,
        high=360.0,
        size=cat.size,
    )
    cat['pa_bulge'] = cat['pa_disk']
    return cat


class StarCatalog(object):
    """
    Star catalog with variable density

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int
        Buffer region with no objects, on all sides of image
    density: float, optional
        Optional density for catalog, if not sent the density is variable and
        drawn from the expected galactic density
    """
    def __init__(self, *, rng, coadd_dim, buff, density=None):
        self.rng = rng

        self._star_cat = load_sample_stars()

        if density is None:
            density_mean = sample_star_density(
                rng=self.rng,
                min_density=2,
                max_density=100,
            )
        else:
            density_mean = density

        # one square degree catalog, convert to arcmin
        area = ((coadd_dim - 2*buff)*SCALE/60)**2
        nobj_mean = area * density_mean
        nobj = rng.poisson(nobj_mean)

        self.density = nobj/area

        self.shifts = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
            nobj=nobj,
        )

        num = self.shifts.size
        self.indices = self.rng.randint(
            0,
            self._star_cat.size,
            size=num,
        )

    def get_objlist(self, *, survey, noise):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey or BasicSurvey
            The survey object
        noise: float
            The noise level, needed for setting gsparams

        Returns
        -------
        [galsim objects]
        """

        num = self.shifts.size

        band = survey.filter_band
        objlist = []
        shift_ind = []
        bright_objlist = []
        bright_shift_ind = []
        bright_mags = []
        for i in range(num):
            star, mag, isbright = self._get_star(survey, band, i, noise)
            if isbright:
                bright_objlist.append(star)
                bright_shift_ind.append(i)
                bright_mags.append(mag)
            else:
                objlist.append(star)
                shift_ind.append(i)

        # objlist = [
        #     self._get_star(survey, band, i, noise)
        #     for i in range(num)
        # ]

        shifts = self.shifts[shift_ind].copy()
        bright_shifts = self.shifts[bright_shift_ind].copy()
        return objlist, shifts, bright_objlist, bright_shifts, bright_mags

    def _get_star(self, survey, band, i, noise):
        """
        Parameters
        ----------
        survey: WLDeblendSurvey or BasicSurvey
            The survey object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object
        noise: float
            The noise level, needed for setting gsparams

        Returns
        -------
        galsim.GSObject
        """

        index = self.indices[i]
        dx = self.shifts['dx'][i]
        dy = self.shifts['dy'][i]

        mag = get_star_mag(stars=self._star_cat, index=index, band=band)
        flux = survey.get_flux(mag)

        gsparams, isbright = get_star_gsparams(mag, flux, noise)
        star = galsim.Gaussian(
            fwhm=1.0e-4,
            flux=flux,
            gsparams=gsparams,
        ).shift(
            dx=dx,
            dy=dy,
        )

        return star, mag, isbright


def get_star_gsparams(mag, flux, noise):
    """
    Get appropriate gsparams given flux and noise

    Parameters
    ----------
    mag: float
        mag of star
    flux: float
        flux of star
    noise: float
        noise of image

    Returns
    --------
    GSParams, isbright where isbright is true for stars with mag less than 18
    """
    do_thresh = do_acc = False
    if mag < 18:
        do_thresh = True
    if mag < 15:
        do_acc = True

    if do_thresh or do_acc:
        isbright = True

        kw = {}
        if do_thresh:
            folding_threshold = noise/flux
            folding_threshold = np.exp(
                np.floor(np.log(folding_threshold))
            )
            kw['folding_threshold'] = min(folding_threshold, 0.005)

        if do_acc:
            kw['kvalue_accuracy'] = 1.0e-8
            kw['maxk_threshold'] = 1.0e-5

        gsparams = galsim.GSParams(**kw)
    else:
        gsparams = None
        isbright = False

    return gsparams, isbright


def make_psf(*, psf_type):
    """
    Make a fixed PSF

    Parameters
    ----------
    psf_type: string
        'gauss' or 'moffat'

    Returns
    -------
    Gaussian or Moffat
    """
    if psf_type == "gauss":
        psf = galsim.Gaussian(fwhm=PSF_FWHM)
    elif psf_type == "moffat":
        psf = galsim.Moffat(fwhm=PSF_FWHM, beta=MOFFAT_BETA)
    else:
        raise ValueError("bad psf_type '%s'" % psf_type)

    return psf


def make_ps_psf(*, rng, dim):
    """
    get a power spectrum psf

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    dim: int
        Dimensions of image

    Returns
    -------
    PowerSpectrumPSF
    """
    return PowerSpectrumPSF(
        rng=rng,
        im_width=dim,
        buff=dim/2,
        scale=SCALE,
        variation_factor=1,
    )


def get_se_dim(*, coadd_dim):
    """
    get se dim given coadd dim.  The se dim is padded out as
        int(np.ceil(coadd_dim * np.sqrt(2))) + 20

    Parameters
    ----------
    coadd_dim: int
        dimensions of coadd

    Returns
    -------
    integer dimensions of SE image
    """
    return int(np.ceil(coadd_dim * np.sqrt(2))) + 20
