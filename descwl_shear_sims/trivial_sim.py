import os
import copy
import galsim
import numpy as np
import descwl
from .se_obs import SEObs
from .cache_tools import cached_catalog_read
from .ps_psf import PowerSpectrumPSF

PSF_FWHM = 0.8
MOFFAT_BETA = 2.5
SCALE = 0.2
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)

GRID_N_ON_SIDE = 6
RANDOM_DENSITY = 80  # per square arcmin

DEFAULT_TRIVIAL_SIM_CONFIG = {
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
}

DEFAULT_FIXED_GAL_CONFIG = {
    "flux": 150000.0,
    "hlr": 0.5,
}


def make_trivial_sim(
    *,
    rng,
    galaxy_catalog,
    coadd_dim,
    g1,
    g2,
    psf,
    psf_dim=51,
    dither=False,
    rotate=False,
    bands=['i'],
    epochs_per_band=1,
    noise_factor=1.0,
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
    """

    se_dim = get_se_dim(coadd_dim=coadd_dim)

    band_data = {}
    for band in bands:

        survey = get_survey(gal_type=galaxy_catalog.gal_type, band=band)
        noise_per_epoch = survey.noise*np.sqrt(epochs_per_band)*noise_factor

        # all_obj = galaxy_catalog.get_all_obj(survey=survey, g1=g1, g2=g2)
        objlist, shifts = galaxy_catalog.get_objlist(
            survey=survey,
            g1=g1,
            g2=g2,
        )

        seobs_list = []
        for epoch in range(epochs_per_band):
            seobs = make_seobs(
                rng=rng,
                noise=noise_per_epoch,
                objlist=objlist,
                shifts=shifts,
                dim=se_dim,
                psf=psf,
                psf_dim=psf_dim,
                dither=dither,
                rotate=rotate,
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
    if gal_type == 'wldeblend':
        survey = WLDeblendSurvey(band=band)
    elif gal_type in ['exp']:
        survey = TrivialSurvey()
    else:
        raise ValueError("bad gal_type: '%s'" % gal_type)

    return survey


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
            flux=gal_config['flux'],
            hlr=gal_config['hlr'],
        )

    return galaxy_catalog


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
            if offset is None:
                offset = galsim.PositionD(x=0.0, y=0.0)
            return gsimage, offset
        else:
            return gsimage


def make_seobs(
    *,
    rng,
    noise,
    objlist,
    shifts,
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

    bmask = galsim.Image(
        np.zeros(dims, dtype='i4'),
        bounds=image.bounds,
        wcs=image.wcs,
        dtype=np.int32,
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


def get_trivial_sim_config(config=None):
    out_config = copy.deepcopy(DEFAULT_TRIVIAL_SIM_CONFIG)
    sub_configs = ['gal_config']

    if config is not None:
        for key in config:
            if key not in out_config and key not in sub_configs:
                raise ValueError("bad key for sim: '%s'" % key)
        out_config.update(config)
    return out_config


def get_fixed_gal_config(config=None):
    out_config = copy.deepcopy(DEFAULT_FIXED_GAL_CONFIG)

    if config is not None:
        for key in config:
            if key not in out_config:
                raise ValueError("bad key for fixed gals: '%s'" % key)
        out_config.update(config)
    return out_config


class WLDeblendSurvey(object):
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


class TrivialSurvey(object):
    def __init__(self):
        self.noise = 1.0


class FixedGalaxyCatalog(object):
    """
    Galaxies of fixed galsim type, flux, and size

    Same for all bands
    """
    def __init__(self, *, rng, coadd_dim, buff, layout, flux, hlr):
        self.gal_type = 'exp'
        self.flux = flux
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

        num = self.shifts.size
        objlist = [
            self._get_galaxy(i).shear(g1=g1, g2=g2)
            for i in range(num)
        ]

        shifts = self.shifts.copy()
        return objlist, shifts

    def get_all_obj(self, *, survey, g1, g2):
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

        num = self.shifts.size
        objlist = [self._get_galaxy(i) for i in range(num)]

        return galsim.Add(
            objlist,
        ).shear(
            g1=g1,
            g2=g2,
        )

    def _get_galaxy(self, i):
        return galsim.Exponential(
            half_light_radius=self.hlr,
            flux=self.flux,
        ).shift(
            dx=self.shifts['dx'][i],
            dy=self.shifts['dy'][i]
        )


class WLDeblendGalaxyCatalog(object):
    """
    Galaxies from wldeblend
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

        band = survey.descwl_survey.filter_band
        objlist = [
            self._get_galaxy(builder, band, i).shear(g1=g1, g2=g2)
            for i in range(num)
        ]

        shifts = self.shifts.copy()
        return objlist, shifts

    def get_all_obj(self, *, survey, g1, g2):
        """
        get a combined set of galsim objects

        Returns
        -------
        equivalent of galsim.Add(list_of_objs)
        """

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        num = self.shifts.size

        band = survey.descwl_survey.filter_band
        objlist = [self._get_galaxy(builder, band, i) for i in range(num)]

        return galsim.Add(
            objlist,
        ).shear(
            g1=g1,
            g2=g2,
        )

    def _get_galaxy(self, builder, band, i):

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
    we get it from the cache, but update the position angles
    each time
    """
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )

    cat = cached_catalog_read(fname)

    cat['pa_disk'] = rng.uniform(
        low=0.0,
        high=360.0,
        size=cat.size,
    )
    cat['pa_bulge'] = cat['pa_disk']
    return cat


def make_psf(*, psf_type):
    if psf_type == "gauss":
        psf = galsim.Gaussian(fwhm=PSF_FWHM)
    elif psf_type == "moffat":
        psf = galsim.Moffat(fwhm=PSF_FWHM, beta=MOFFAT_BETA)
    else:
        raise ValueError("bad psf_type '%s'" % psf_type)

    return psf


def make_ps_psf(*, rng, dim):
    return PowerSpectrumPSF(
        rng=rng,
        im_width=dim,
        buff=dim/2,
        scale=SCALE,
        variation_factor=1,
    )


def get_se_dim(*, coadd_dim):
    """
    get se dim given coadd dim
    """
    return int(np.ceil(coadd_dim * np.sqrt(2))) + 20
