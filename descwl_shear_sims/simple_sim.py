"""Class for making a mulit-band, multi-epoch sim w/ galsim.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
and cleaned up a fair bit.
"""
import os
import logging
import copy

import galsim
import numpy as np
from collections import OrderedDict

from .se_obs import SEObs
from .render_sim import append_wcs_info_and_render_objs_with_psf_shear
from .gen_tanwcs import gen_tanwcs
from .gen_sip_wcs import gen_sip_wcs
from .randsphere import randsphere, randcap
from .gen_masks import (
    generate_basic_mask,
    generate_cosmic_rays,
    generate_bad_columns,
)
from .gen_star_masks import add_bright_star_mask
from .star_bleeds import add_bleed

from .stars import (
    sample_star,
    sample_fixed_star,
    load_sample_stars,
    sample_star_density,
)


from .ps_psf import PowerSpectrumPSF

# default mask bits from the stack
from .lsst_bits import BAD_COLUMN, COSMIC_RAY, SAT, BRIGHT
from .saturation import BAND_SAT_VALS, saturate_image_and_mask
from .cache_tools import cached_catalog_read
from .sim_constants import ZERO_POINT

LOGGER = logging.getLogger(__name__)

GALS_KWS_DEFAULTS = {
    'exp': {'half_light_radius': 0.5, 'mag': 17.75},
    'wldeblend': {'make_round': False, 'noise_factor': 1.0},
}
STARS_KWS_DEFAULTS = {
    'density': 1,
    'mag': 19.0,
    'subtract_bright': False,
}
SAMPLE_DENSITY_KEYS = ('min_density', 'max_density')

PSF_KWS_DEFAULTS = {
    'gauss': {'fwhm': 0.8, 'g1': 0.0, 'g2': 0.0},
    'moffat': {'fwhm': 0.8, 'g1': 0.0, 'g2': 0.0, 'beta': 2.5},
    'ps': {},  # the defaults for the ps PSF are in the class
}


class Sim(object):
    """A multi-epoch, multi-band simulation for testing weak lensing shear
    measurements.

    Parameters
    ----------
    rng : np.random.RandomState, int or None
        An RNG to use for drawing the objects. If an int or None is passed,
        the input is used to initialize a new `np.random.RandomState` object.
    epochs_per_band : int, optional
        The number of single epoch images **per band**. Default is 10.
    noise_per_band : float or list of floats, optional
        The total noise for a single band. Can be different per band. Default
        is 0.6 units which roughly approximates the 10-year LSST depth for an
        image with a zero-point of 30 in the i-band.
    bands : list of str, optional
        A list of bands to simulate. Default is ('r', 'i', 'z').
    g1 : float, optional
        The simulated shear for the 1-axis. Default is 0.02.
    g2 : float, optional
        The simulated shear for the 2-axis. Default is 0.
    shear_scene : bool, optional
        Whether or not to shear the full scene. Default is True.
    scale : float, optional
        The overall pixel scale of the images. Default is 0.2 arcsec.
    coadd_dim : int, optional
        The total dimension of the coadd image. Default is 351.  An odd default
        is chosen, along with that of the single epoch images, so that the
        grids will match perfectly in the case of no dithers/rotations etc.
        However, odd dimensions are not required.
    psf_dim: int, optional
        Dimensions of the psf image.  Default is 53
    buff : int, optional
        The width of the buffer region in the coadd image where no objects are
        drawn. Default is 50.
    cap_radius:  float
        Draw positions from a spherical cap of this opening angle in arcmin
        rather than restricting positions to a small region within the coadd
    edge_width:  int
        Width of boundary to be marked as EDGE in the bitmask, default 5
    se_dim: int
        Dimensions of the single epoch image.  If None (the default) the
        size is chosen to encompass the coadd for small dithers.
    layout_type : str, optional
        A string indicating how to layout the objects in the image. Possible options
        are

            'random' : randomly place the objects in the image
            'grid' : place the objects on a grid in the image

        Default is 'random'.
    layout_kws : dict, optional
        A dictionary giving options for the specific layout type. These vary per
        layout type and are

            'random' - no options
            'grid' - any of the following
                'dim' : int, required
                    An integer giving the dimension of the grid on one axis.

    gals: bool
        If set to True, draw galaxies. Default True.
    gals_type : str, optional
        A string indicating what kind of galaxy to simulate. Possible options are

            'exp' : simple exponential disks
            'wldeblend' : use the LSSTDESC WeakLensingDeblending package

        Default is 'exp'.
    gals_kws : dict, optional
        A dictionary of options for constructing galaxies. These vary per
        galaxy type. Possible entries per galaxy type are

            all types - any of the following
                'density' : float
                    The density of galaxies in number per arcminute^2.
                    Default is 80.

            'exp' - any of the keywords to `galsim.Exponential`
                Default is 'half_light_radius=0.5'.

            'wldeblend' - any of the following keywords
                'catalog' : str
                    A path to the catalog to draw from. If this keyword is not
                    given, you need to have the one square degree catsim catalog
                    in the current working directory or in the directory given by
                    the environment variable 'CATSIM_DIR'.
                'make_round': bool
                    If True, make the objects round
                'noise_factor': float
                    Factor by which to scale the noise, default 1

    psf_type : str, optional
        A string indicating the kind of PSF. Possible options are

            'gauss' : a Gaussian PSF
            'moffat': a Moffat PSF
            'ps' : an approximate spatially varying PSF constructed with a
                power spectrum model

        Default is 'gauss'.
    psf_kws : dict, optional
        A dictionary of options for constructing the PSF. These vary per
        psf type. Possible entries per psf type are

            'gauss' - any of the following keywords
                'fwhm': float, optional
                    The FWHM of the Gaussian. Default is 0.8.
                'g1': float, optional
                    The shape of the PSF on the 1-axis. Default is 0.
                'g2': float, optional
                    The shape of the PSF on the 2-axis. Default is 0.
            'moffat' - any of the following keywords
                'fwhm': float, optional
                    The FWHM of the Gaussian. Default is 0.8.
                'beta': float, optional
                    The beta of the moffat, default 2.5
                'g1': float, optional
                    The shape of the PSF on the 1-axis. Default is 0.
                'g2': float, optional
                    The shape of the PSF on the 2-axis. Default is 0.


            'ps' - any of the following keywords
                trunc : float
                    The truncation scale for the shape/magnification power spectrum
                    used to generate the PSF variation. Default is 1.
                noise_level : float, optional
                    If not `None`, generate a noise field to add to the PSF images with
                    desired noise. A value of 1e-2 generates a PSF image with an
                    effective signal-to-noise of ~250. Default is none
                variation_factor : float, optional
                    This factor is used internally to scale the overall variance in the
                    PSF shape power spectra and the change in the PSF size across the
                    image. Setting this factor greater than 1 results in more variation
                    and less than 1 results in less variation. Default is 10.
                median_seeing : float, optional
                    The approximate median seeing for the PSF. Default is 0.8.

    wcs_kws : dict, optional
        A dictionary of options for constructing the WCS. These are any of

            type: str
                Type of the wcs.  Options are
                    'tan' for a pure tangent plane
                    'tan-sip' to include distortioins in sip convention
                Default is 'tan'
            position_angle_range : 2-tuple of floats
                The range of position angles to select from for rotating the image
                WCS coordinares. In degrees.
            dither_range : 2-tuple of floats
                The lowest and highest dither in uv coordinate pixels.
            scale_frac_std : float
                The fractional variance in the generated image pixel scale.
            shear_std : float
                The standard deviation of the Gaussian shear put into the WCS.
            ra : float, optional
                The ra of the center of the image in world coordinates.
            dec : float, optional
                The ra of the center of the image in world coordinates.

        The default values will generate a WCS that is at `(ra, dec) = (180, 0)`
        with `position_angle_range=(0,0)`, `scale_frac_std=0`, `dither_range=(0,0)`,
        and `shear_std=0`.
    cosmic_rays : bool, optional
        If `True` then add cosmic rays, otherwise do not. Default is `False`.
    cosmic_rays_kws : dict, optional
        A dictionary of options for generating cosmic rays

            mean_cosmic_rays : int, optional
                The mean number of cosmic rays.
            min_length : int, optional
                The minimum length of the track.
            max_length : int, optional
                The maximum length of the track.

        See descwl_shear_sims.gen_masks.generate_cosmic_rays for the defaults.
        The imput value for `mean_cosmic_rays` is scaled by the area of
        the SE image relative to the final coadd area without buffers.
    bad_columns : bool, optional
        If `True` then add bad columns, otherwise do not. Default is `False`.
    bad_columns_kws : dict, optional
        A dictionary of options for generating bad columns

            mean_bad_cols : float, optional
                The mean of the Poisson distribution for the total number of
                bad columns to generate.
            gap_prob : float
                The probability that the bad column has a gap in it.
            min_gap_frac : float
                The minimum fraction of the image that the gap spans.
            max_gap_frac : floatn
                The maximum fraction of the image that the gap spans.

        See descwl_shear_sims.gen_masks.generate_bad_columns for the defaults.
    saturate: bool
        If True, saturate values above a threshold.  Default is False.  If
        star_bleeds is True, then saturate is also set to True
    stars: bool, optional
        If True, draw stars in the sim. Default is False.
    stars_type : str, optional
        A string indicating the kinds of stars to draw. Should be one of

            'fixed' - draw stars with a fixed brightness of magnitude 19
            'sample' - sample stars from a catalog derived from more detailed
                       simulations

        Default is 'fixed'.
    stars_kws: dict, optional
        A dictionary of options for generating stars

            density: float
                number per square arcmin, default 1.  For stars_type 'sample',
                can also be a dict with min_density and max_density
            mag: float
                magnitude for fixed stars, default 19.0
            min_mag: float
                Minimum alowed mag in any band.  Used for stars_type 'sample'
            subtract_bright: bool
                If True, subtract off the bright stars that are saturated.

    star_bleeds: bool, optional
        If `True` then add bleeds. Default is `False`.

    trim_stamps: bool, optional
        If True, trim the stamps copied into the images.  This is to prevent
        the star images from getting too large and galsim raising exceptions.
        Default True.

    Methods
    -------
    gen_sim()
        Generate a simulation.

    Properties
    ----------
    mean_nobj: int
        Get the expected mean number of objects.  For non-grids we
        draw a poisson deviate with this mean
    mean_density: float
        Get the mean expected density of objects per square arcminute.
    star_density: float
        Get the mean expected density of stars used in the simulation
    gal_density: float
        Get the mean expected density of galaxies used in the simulation
    """
    def __init__(
        self, *,
        rng,
        epochs_per_band=10,
        noise_per_band=0.6,
        bands=('r', 'i', 'z'),
        g1=0.02,
        g2=0.0,
        shear_scene=True,
        scale=0.2,
        coadd_dim=351,
        psf_dim=53,
        buff=50,
        cap_radius=None,
        edge_width=5,
        se_dim=None,
        layout_type='random',
        layout_kws=None,
        gals=True,
        gals_type='exp',
        gals_kws=None,
        psf_type='gauss',
        psf_kws=None,
        wcs_kws=None,
        cosmic_rays=False,
        cosmic_rays_kws=None,
        bad_columns=False,
        bad_columns_kws=None,
        saturate=False,
        stars=False,
        stars_type='fixed',
        stars_kws=None,
        star_bleeds=False,
        trim_stamps=True,
    ):
        self._rng = (
            rng
            if isinstance(rng, np.random.RandomState)
            else np.random.RandomState(seed=rng))

        ########################################
        # we set these here so they are seeded once - no calls to the RNG
        # should preceed these calls
        self._noise_rng = np.random.RandomState(
            seed=self._rng.randint(1, 2**32-1))
        self._galsim_rng = galsim.BaseDeviate(
            seed=self._rng.randint(low=1, high=2**32-1))

        ########################################
        # rendering
        self.saturate = saturate  # will be forced True if star_bleeds is True
        self.trim_stamps = trim_stamps

        ########################################
        # band structure
        self.epochs_per_band = epochs_per_band
        self.bands = bands
        self.n_bands = len(bands)
        self.noise_per_band = np.array(noise_per_band) * np.ones(self.n_bands)
        self.noise_per_epoch = self.noise_per_band * np.sqrt(self.epochs_per_band)

        ########################################
        # shears
        self.g1 = g1
        self.g2 = g2
        self.shear_scene = shear_scene

        ########################################
        # defects & masking
        self.cosmic_rays = cosmic_rays
        self.cosmic_rays_kws = cosmic_rays_kws or {}
        self.bad_columns = bad_columns
        self.bad_columns_kws = bad_columns_kws or {}
        self.edge_width = edge_width
        assert edge_width >= 2, 'edge width must be >= 2'

        ########################################
        # WCS and image sizes
        self._setup_wcs(
            scale=scale,
            coadd_dim=coadd_dim,
            buff=buff,
            cap_radius=cap_radius,
            wcs_kws=wcs_kws,
            se_dim=se_dim,
        )

        ########################################
        # PSF
        self.psf_type = psf_type
        self.psf_kws = copy.deepcopy(PSF_KWS_DEFAULTS[self.psf_type])
        if psf_kws is not None:
            self.psf_kws.update(copy.deepcopy(psf_kws))

        # used when not doing wldeblend
        self._default_flux_funcs = {b: get_flux for b in self.bands}

        # info about coadd PSF image
        self.psf_dim = int(psf_dim)
        self._psf_cen = (self.psf_dim - 1)/2

        ########################################
        # galaxies
        self.gals = gals
        self.gals_type = gals_type
        self.gals_kws = copy.deepcopy(GALS_KWS_DEFAULTS[self.gals_type])
        if gals_kws:
            self.gals_kws.update(copy.deepcopy(gals_kws))
        if gals_type == 'exp':
            check_keys(
                self.gals_kws,
                ('half_light_radius', 'mag', 'density'),
                'gals_kws',
            )
        else:
            check_keys(
                self.gals_kws,
                ('catalog', 'make_round', 'noise_factor'),
                'gals_kws',
            )

        if self.gals_type == 'exp':
            self._fixed_gal_mag = self.gals_kws['mag']

        self._gal_dens = self.gals_kws.get('density', 80)

        # now we call any extra init for wldeblend
        # this call will reset some things above
        if self.gals_type == 'wldeblend':
            self._extra_init_for_wldeblend()

        ######################
        # stars
        self._setup_stars(
            stars=stars,
            stars_type=stars_type,
            stars_kws=stars_kws,
        )
        self.star_bleeds = star_bleeds
        if self.star_bleeds:
            self.saturate = True

        ####################################
        # grids
        self.layout_type = layout_type
        self.layout_kws = layout_kws or {}
        # reset nobj to the number in a grid if we are using one
        if self.layout_type == 'grid':
            self._obj_grid_ind = 0
            self._nobj = self.layout_kws.get('dim')**2

            # re-define the density parameters for the grid
            star_frac = self._star_dens/(self._gal_dens + self._star_dens)
            density = self._nobj/self.area_sqr_arcmin

            self._gal_dens = density*(1-star_frac)
            self._star_dens = density*star_frac
        else:
            self._nobj = (
                int(self._gal_dens * self.area_sqr_arcmin)
                + int(self._star_dens * self.area_sqr_arcmin)
            )

        # we only allow the sim class to be used once
        # so that method outputs can be cached safely as needed
        # this attribute gets set to True after it is used
        self.called = False

        LOGGER.info('simulating bands: %s', self.bands)

    @property
    def mean_nobj(self):
        """
        the mean number of objects expected.  For non-grids we draw a poisson
        deviate with this mean.
        """
        return self._nobj

    @property
    def mean_density(self):
        """
        the density of stars
        """
        return self._nobj/self.area_sqr_arcmin

    @property
    def star_density(self):
        """
        the density of stars
        """
        return self._star_dens

    @property
    def gal_density(self):
        """
        the density of stars
        """
        return self._gal_dens

    @property
    def object_data(self):
        """
        get the object data

        Returns
        -------
        list of dict, on element per object.  Each dict is keyed by band
        """
        return self._object_data

    def _setup_wcs(
            self, *,
            scale, coadd_dim, buff, cap_radius, wcs_kws, se_dim,
    ):
        self.scale = scale
        self.coadd_dim = coadd_dim
        self.buff = buff
        self.cap_radius = cap_radius
        if self.cap_radius is not None:
            self.buff = 0

        self._set_wcs_kws(wcs_kws)

        # the SE image could be rotated, so we make it big enough to cover the
        # whole coadd region plus we make sure it is odd
        default_se_dim = (
            int(np.ceil(self.coadd_dim * np.sqrt(2))) + 10 + 2*self.edge_width
        )
        if default_se_dim % 2 == 0:
            default_se_dim += 1

        if se_dim is not None:
            self.se_dim = int(se_dim)
            assert self.se_dim >= default_se_dim
        else:
            # the SE image could be rotated, so we make it big enough to cover the
            # whole coadd region plus we make sure it is odd
            self.se_dim = default_se_dim

        self._se_cen = (self.se_dim - 1) / 2

        self._world_ra = self.wcs_kws.get('ra', 180)
        self._world_dec = self.wcs_kws.get('dec', 0)
        self._world_origin = galsim.CelestialCoord(
            ra=self._world_ra * galsim.degrees,
            dec=self._world_dec * galsim.degrees)
        self._se_origin = galsim.PositionD(x=self._se_cen, y=self._se_cen)

        # coadd WCS to determine where we should draw locations
        # for objects in the sky
        self._coadd_cen = (self.coadd_dim - 1) / 2
        self._coadd_origin = galsim.PositionD(x=self._coadd_cen, y=self._coadd_cen)
        self.coadd_wcs = galsim.TanWCS(
            affine=galsim.AffineTransform(
                self.scale, 0, 0, self.scale,
                origin=galsim.PositionD(x=self._coadd_cen, y=self._coadd_cen),
                world_origin=galsim.PositionD(x=0, y=0)
            ),
            world_origin=self._world_origin,
            units=galsim.arcsec
        )
        self._coadd_jac = self.coadd_wcs.jacobian(
            world_pos=self._world_origin).getMatrix()

        if self.cap_radius is None:
            self._ra_range, self._dec_range = self._get_patch_ranges()

            self.area_sqr_arcmin = (
                # factor of 60 to arcmin
                (self._ra_range[1] - self._ra_range[0]) * 60 *
                # factor of 180 * 60 / pi to go from radians to arcmin
                180 * 60 / np.pi * (np.sin(self._dec_range[1] / 180.0 * np.pi) -
                                    np.sin(self._dec_range[0] / 180.0 * np.pi)))
        else:
            # area of spherical cap on the unit sphere with opening angle
            # cap_radius in arcmin
            cap_rad_radians = np.radians(self.cap_radius/60)
            area_radians = 2 * np.pi * (1 - np.cos(cap_rad_radians))
            self.area_sqr_arcmin = area_radians * (60*180/np.pi)**2

        LOGGER.info('area is %f arcmin**2', self.area_sqr_arcmin)

    def _set_wcs_kws(self, wcs_kws):
        self.wcs_kws = wcs_kws or {}
        if 'type' not in self.wcs_kws:
            self.wcs_kws['type'] = 'tan'

        if self.wcs_kws['type'] == 'tan':
            self.wcs_func = gen_tanwcs
        elif self.wcs_kws['type'] == 'tan-sip':
            self.wcs_func = gen_sip_wcs
        else:
            raise ValueError("bad wcs type '%s'" % self.wcs_kws['type'])

    def _setup_stars(
        self, *,
        stars,
        stars_kws,
        stars_type,
    ):
        self.stars = stars
        self.stars_type = stars_type
        self.stars_kws = copy.deepcopy(STARS_KWS_DEFAULTS)
        if stars_kws is not None:
            self.stars_kws.update(copy.deepcopy(stars_kws))

        if self.stars:
            assert self.stars_type in ('sample', 'fixed')
            check_keys(
                self.stars_kws,
                ('density', 'mag', 'min_mag', 'subtract_bright'),
                'stars_kws',
            )
            if ('density' in self.stars_kws and
                    isinstance(self.stars_kws['density'], dict)):
                check_keys(
                    self.stars_kws['density'],
                    SAMPLE_DENSITY_KEYS,
                    'stars_kws["density"]',
                )
                # make sure both are there
                assert (
                    len(self.stars_kws['density'].keys())
                    == len(SAMPLE_DENSITY_KEYS)
                )

            if isinstance(self.stars_kws['density'], dict):
                ddict = self.stars_kws['density']
                self._star_dens = sample_star_density(
                    rng=self._rng,
                    min_density=ddict['min_density'],
                    max_density=ddict['max_density'],
                )
            else:
                self._star_dens = self.stars_kws['density']

            if self.stars_type == 'sample':
                self._example_stars = load_sample_stars()
            else:
                self._example_stars = None
        else:
            self._star_dens = 0.0
            self._example_stars = None

    def _extra_init_for_wldeblend(self):
        # guard the import here
        import descwl
        from .galaxy_builder import RoundGalaxyBuilder

        # make sure to find the proper catalog
        if self.gals:
            if 'catalog' not in self.gals_kws:
                fname = os.path.join(
                    os.environ.get('CATSIM_DIR', '.'),
                    'OneDegSq.fits')
            else:
                fname = self.gals_kws['catalog']

            self._wldeblend_cat = cached_catalog_read(fname)
            self._wldeblend_cat['pa_disk'] = self._rng.uniform(
                low=0.0, high=360.0, size=self._wldeblend_cat.size)
            self._wldeblend_cat['pa_bulge'] = self._wldeblend_cat['pa_disk']

        self._surveys = {}
        self._builders = {}
        self._survey_flux_funcs = {}
        noises = []
        for band in self.bands:
            # make the survey and code to build galaxies from it
            pars = descwl.survey.Survey.get_defaults(
                survey_name='LSST',
                filter_band=band)

            pars['survey_name'] = 'LSST'
            pars['filter_band'] = band
            pars['pixel_scale'] = self.scale

            # note in the way we call the descwl package, the image width
            # and height is not actually used
            pars['image_width'] = self.coadd_dim
            pars['image_height'] = self.coadd_dim

            # some versions take in the PSF and will complain if it is not
            # given
            try:
                _svy = descwl.survey.Survey(**pars)
            except Exception:
                pars['psf_model'] = None
                _svy = descwl.survey.Survey(**pars)

            self._surveys[band] = _svy
            self._survey_flux_funcs[band] = _svy.get_flux

            if self.gals_kws['make_round']:
                self._builders[band] = RoundGalaxyBuilder(
                    survey=self._surveys[band],
                    no_disk=False,
                    no_bulge=False,
                    no_agn=False,
                    verbose_model=False,
                )
            else:
                self._builders[band] = descwl.model.GalaxyBuilder(
                    survey=self._surveys[band],
                    no_disk=False,
                    no_bulge=False,
                    no_agn=False,
                    verbose_model=False,
                )

            noises.append(np.sqrt(self._surveys[band].mean_sky_level))

        self.noise_per_band = np.array(noises)*self.gals_kws['noise_factor']
        self.noise_per_epoch = self.noise_per_band * np.sqrt(self.epochs_per_band)

        # when we sample from the catalog, we need to pull the right number
        # of objects. Since the default catalog is one square degree
        # and we fill a fraction of the image, we need to set the
        # base source density `ngal`. This is in units of number per
        # square arcminute.
        if self.gals:
            self._gal_dens = self._wldeblend_cat.size / (60 * 60)
        else:
            self._gal_dens = 0.0

        LOGGER.info('catalog density: %f per sqr arcmin', self._gal_dens)

    def _get_patch_ranges(self):
        ra = []
        dec = []
        edges = [self.buff, self.coadd_dim - self.buff]
        for x in edges:
            for y in edges:
                sky = self.coadd_wcs.toWorld(galsim.PositionD(x=x, y=y))
                ra.append(sky.ra.deg)
                dec.append(sky.dec.deg)

        # make sure ra_range bounds self._world_ra
        ra_range = np.array([np.min(ra), np.max(ra)])
        if ra_range[1] < self._world_ra:
            # max is min and min needs to go up by 360
            ra_range = np.array([ra_range[1], ra_range[0] + 360])
        elif ra_range[0] > self._world_ra:
            # min is max and max needs to go down by 360
            ra_range = np.array([ra_range[1] - 360, ra_range[0]])
        return ra_range, np.array([np.min(dec), np.max(dec)])

    def gen_sim(self):
        """Generate a simulation.

        Note that this method can only be called once.

        Returns
        -------
        sim : OrderedDict
            A dictionary keyed on band which contains a list of SEObs for
            each epoch in the band.
        """

        if self.called:
            raise RuntimeError("A `Sim` object can only be called once!")
        self.called = True

        self._object_data = self._generate_objects()

        self._set_gsparams(self._object_data)

        band_data = OrderedDict()
        for band_ind, band in enumerate(self.bands):
            band_objs = [o[band] for o in self._object_data]

            wcs_objects = self._get_wcs_for_band(band)

            (
                psf_funcs_galsim,
                psf_funcs_rendered,
            ) = self._get_psf_funcs_for_band(band)

            band_data[band] = []
            for epoch, wcs, psf_galsim, psf_func in zip(
                    range(self.epochs_per_band), wcs_objects,
                    psf_funcs_galsim, psf_funcs_rendered
            ):
                se_image = append_wcs_info_and_render_objs_with_psf_shear(
                    objs=band_objs,
                    psf_function=psf_galsim,
                    wcs=wcs,
                    img_dim=self.se_dim,
                    method='auto',
                    g1=self.g1,
                    g2=self.g2,
                    shear_scene=self.shear_scene,
                    threshold=self.noise_per_band[band_ind],
                    trim_stamps=self.trim_stamps,
                )

                se_image += self._generate_noise_image(band_ind)

                # make galsim image with same wcs as se_image but
                # with pure random noise
                noise_example = self._generate_noise_image(band_ind)
                noise_image = se_image.copy()
                noise_image.array[:, :] = noise_example

                se_weight = (
                    se_image.copy() * 0
                    + 1.0 / self.noise_per_epoch[band_ind]**2
                )

                if self.gals_type == 'wldeblend':
                    # put the images on our common ZERO_POINT before
                    # checking for saturation
                    self._rescale_wldeblend(
                        image=se_image,
                        noise=noise_image,
                        weight=se_weight,
                        band=band,
                    )

                # put this after to make sure bad cols are totally dark
                bmask, se_image = self._generate_mask_plane(
                    se_image=se_image,
                    wcs=wcs,
                    objs=band_objs,
                    band=band,
                    epoch=epoch,
                )

                band_data[band].append(
                    SEObs(
                        image=se_image,
                        noise=noise_image,
                        weight=se_weight,
                        wcs=wcs,
                        psf_function=psf_func,
                        bmask=bmask,
                    )
                )

        if self.stars:
            add_bright_star_masks(self._object_data, band_data)
            if self.stars_kws['subtract_bright']:

                if self.gals_type == 'wldeblend':
                    factors = self._get_wldeblend_rescale_factors()
                else:
                    factors = [None]*len(self.bands)

                subtract_bright_stars(self._object_data, band_data, factors)

        return band_data

    def _set_gsparams(self, all_objs):
        """Function due to M Jarvis. Attempts to set the folding threshold
        so that we render will into the noise in the final coadded image.
        """
        # these are approximate noises in the final coadded image
        noises = {}
        for band_ind, band in enumerate(self.bands):
            sky_noise_per_pixel = (
                self.noise_per_band[band_ind]
                / np.sqrt(self.n_bands)
            )
            noises[band] = sky_noise_per_pixel

        b1 = self.bands[0]
        for odata in all_objs:
            do_thresh = do_acc = False
            if odata[b1]['type'] == 'star':
                if any(odata[band]['mag'] < 18 for band in odata):
                    do_thresh = True

                if any(odata[band]['mag'] < 15 for band in odata):
                    do_acc = True

            if do_thresh or do_acc:
                for band in self.bands:
                    kw = {}
                    band_data = odata[band]

                    obj = band_data['obj']

                    if do_thresh:
                        sky_noise_per_pixel = noises[band]

                        folding_threshold = sky_noise_per_pixel / obj.flux
                        folding_threshold = np.exp(
                            np.floor(np.log(folding_threshold))
                        )

                        kw['folding_threshold'] = min(folding_threshold, 0.005)

                    if do_acc:
                        kw['kvalue_accuracy'] = 1.0e-8
                        kw['maxk_threshold'] = 1.0e-5

                    gsp = galsim.GSParams(**kw)
                    band_data['obj'] = obj.withGSParams(gsp)

    def _get_wldeblend_rescale_factors(self):
        factors = []
        for band in self.bands:
            fac = self._get_wldeblend_rescale_fac(band)
            factors.append(fac)
        return factors

    def _get_wldeblend_rescale_fac(self, band):
        survey = self._surveys[band]

        s_zp = survey.zero_point
        s_et = survey.exposure_time
        return 10.0**(0.4*(ZERO_POINT - 24.0))/s_zp/s_et

    def _rescale_wldeblend(self, *, image, noise, weight, band):
        """
        Take out the exposure time
        """

        fac = self._get_wldeblend_rescale_fac(band)
        wfac = 1.0/fac**2

        image *= fac
        noise *= fac
        weight *= wfac

    def _generate_mask_plane(self, *, se_image, wcs, objs, band, epoch):
        """
        set masks for edges, cosmics, bad columns and saturated stars/bleeds

        also clip high values and set mask bits

        make sure the image is on the right zero point before calling this code
        """

        # saturation occurs in the single epoch image.  Note before calling
        # this method we should have rescaled wldeblend images, and we have put
        # that into the sat vals
        sat_val = BAND_SAT_VALS[band]
        shape = se_image.array.shape
        bmask = generate_basic_mask(shape=shape, edge_width=self.edge_width)

        if self.saturate:
            saturate_image_and_mask(
                image=se_image.array,
                mask=bmask,
                sat_val=sat_val,
            )

        area_factor = (
            (self.coadd_dim - 2 * self.buff)**2
            / self.se_dim**2)

        if self.cosmic_rays:
            defaults = {
                'mean_cosmic_rays': 1,
            }
            defaults.update(self.cosmic_rays_kws)
            defaults['mean_cosmic_rays'] = (
                defaults['mean_cosmic_rays']
                / area_factor
            )
            msk = generate_cosmic_rays(
                shape=shape,
                rng=self._rng,
                **defaults,
            )
            bmask[msk] |= COSMIC_RAY
            se_image.array[msk] = sat_val

        if self.bad_columns:
            defaults = {
                'mean_bad_cols': 1,
            }
            defaults.update(self.bad_columns_kws)
            defaults['mean_bad_cols'] = (
                defaults['mean_bad_cols']
                / np.sqrt(area_factor)
            )
            msk = generate_bad_columns(
                shape=shape,
                rng=self._rng,
                **defaults,
            )
            bmask[msk] |= BAD_COLUMN
            se_image.array[msk] = 0.0

        if self.stars:
            for odata in objs:
                if (odata['overlaps'][epoch]):
                    assert len(odata['pos']) == len(odata['overlaps'])

                    pos = odata['pos'][epoch]
                    row, col = int(pos.y), int(pos.x)
                    if (bmask[row, col] & SAT) != 0:
                        odata['is_bright'] = True

        if self.stars and self.star_bleeds:
            # special stars that get bleeds.  They are marked
            # saturated, but note there can be saturated pixels
            # even without this set (see saturate_image_and_mask)
            # TODO rename saturated key to has_bleed or something
            for odata in objs:
                if (
                    odata['overlaps'][epoch] and
                    odata['type'] == 'star' and
                    odata['is_bright']
                ):
                    pos = odata['pos'][epoch]

                    add_bleed(
                        image=se_image.array,
                        bmask=bmask,
                        pos=pos,
                        mag=odata['mag'],
                        band=band,
                    )

        bmask_image = galsim.Image(
            bmask,
            bounds=se_image.bounds,
            wcs=se_image.wcs,
            dtype=np.int32,
        )
        return bmask_image, se_image

    def _generate_noise_image(self, band_ind):
        """
        generate a random noise field for the given band

        Parameters
        ----------
        band_ind: int
            The band index

        Returns
        -------
        numpy array
        """

        return (
            self._noise_rng.normal(size=(self.se_dim, self.se_dim)) *
            self.noise_per_epoch[band_ind]
        )

    def _generate_objects(self):
        """Generate all objects in all bands.

        Returns
        -------
        object_data : list of OrderedDicts
            A list the length of the number of objects with an OrderedDict
            for each object holding each objects galsim representation in each band,
            the object type, and it's offset in u,v in the image.
        """

        object_data = []

        if not self.gals and not self.stars:
            return object_data

        nobj = self._get_nobj()
        LOGGER.info('drawing %d objects for a %f square arcmin patch',
                    nobj, self.area_sqr_arcmin)

        total_density = (self._star_dens + self._gal_dens)
        if total_density > 0:
            gal_frac = self._gal_dens / (self._star_dens + self._gal_dens)
        else:
            gal_frac = 0

        for i in range(nobj):
            # unsheared offset from center of uv image
            du, dv = self._get_dudv()
            dudv = galsim.PositionD(x=du, y=dv)

            if self.gals and self.stars:
                ran_u = self._rng.uniform()
                _type_to_draw = 'gal' if ran_u < gal_frac else 'star'
            elif not self.stars and self.gals:
                _type_to_draw = 'gal'
            elif not self.gals and self.stars:
                _type_to_draw = 'star'
            else:
                raise ValueError(
                    "We could not determine a type of object to draw! "
                    "Check the input settings for `stars`, `galaxies`, "
                    "and their densities!"
                )

            if _type_to_draw == 'gal':
                # get the galaxy
                if self.gals_type == 'exp':
                    obj_data = self._get_gal_exp()
                elif self.gals_type == 'wldeblend':
                    obj_data = self._get_gal_wldeblend()
                else:
                    raise ValueError(
                        'gals_type "%s" not valid!' % self.gals_type
                    )
            else:
                # get the star as an dict by band
                obj_data = self._get_star()
                if not self._keep_star(obj_data):
                    continue

            for band in self.bands:
                obj_data[band]['dudv'] = dudv
            object_data.append(obj_data)

        return object_data

    def _keep_star(self, star):
        """
        remove stars under certain conditions
        Note this no longer checks saturated

        the conditions are:
          - too bright
        """
        keep = True

        min_mag = self.stars_kws.get('min_mag', None)
        if min_mag is not None:
            if any((star[band]['mag'] < min_mag for band in star)):
                keep = False

        return keep

    def _get_dudv(self):
        """Return an offset from the center of the image in the (u, v) plane
        of the coadd."""
        if self.layout_type == 'grid':
            gdim = self.layout_kws.get('dim')
            # half of the width of center of the patch that has objects
            # object locations should be in [-pos_width, +pos_width] below
            # we are making a grid in (x,y) as opposed to ra-dec so the spacing
            # is even
            frac = 1.0 - self.buff * 2 / self.coadd_dim
            _pos_width = self.coadd_dim * frac * 0.5
            yind, xind = np.unravel_index(
                self._obj_grid_ind, (gdim, gdim))
            dg = _pos_width * 2 / gdim
            self._obj_grid_ind += 1
            dpos = galsim.PositionD(
                y=yind * dg + dg/2 - _pos_width,
                x=xind * dg + dg/2 - _pos_width)
        else:
            if self.cap_radius is None:
                ra, dec = randsphere(
                    self._rng, 1, ra_range=self._ra_range, dec_range=self._dec_range)
            else:
                ra, dec = randcap(
                    rng=self._rng,
                    nrand=1,
                    ra=self._world_ra,
                    dec=self._world_dec,
                    radius=self.cap_radius/60,
                )

            wpos = galsim.CelestialCoord(
                ra=ra[0] * galsim.degrees, dec=dec[0] * galsim.degrees)
            ipos = self.coadd_wcs.toImage(wpos)
            dpos = ipos - self._coadd_origin

        dudv = np.dot(self._coadd_jac, np.array([dpos.x, dpos.y]))
        return dudv

    def _get_nobj(self):
        # grids have a fixed number of objects, otherwise we use a varying number
        if self.layout_type == 'grid':
            return self._nobj
        else:
            return self._rng.poisson(self._nobj)

    def _get_gal_exp(self):
        """Return an OrderedDict keyed on band with the galsim object for
        a given exp gal."""
        # flux = 10**(0.4 * (ZERO_POINT - self._fixed_gal_mag))
        flux = get_flux(self._fixed_gal_mag)

        use_kwargs = copy.deepcopy(self.gals_kws)
        use_kwargs.pop('mag', None)
        use_kwargs.pop('density', None)

        _gal = OrderedDict()
        for band in self.bands:
            obj = galsim.Exponential(
                **use_kwargs
            ).withFlux(flux)
            _gal[band] = {
                'obj': obj,
                'type': 'galaxy',
                'is_bright': False,
            }

        return _gal

    def _get_gal_wldeblend(self):
        """Return an OrderedDict keyed on band with the galsim object for
        a given exp gal."""

        _gal = OrderedDict()

        rind = self._rng.choice(self._wldeblend_cat.size)
        angle = self._rng.uniform() * 360

        for band in self.bands:
            obj = self._builders[band].from_catalog(
                self._wldeblend_cat[rind], 0, 0,
                self._surveys[band].filter_band,
            ).model.rotate(
                angle * galsim.degrees,
            )
            _gal[band] = {
                'obj': obj,
                'type': 'galaxy',
                'is_bright': False,
            }

        return _gal

    def _get_star(self):
        """

        Returns
        -------
        An OrderedDict keyed on band with data of the form
            {'obj': Gaussian, 'type': 'star'}
        """

        if self.gals_type == 'wldeblend':
            flux_funcs = self._survey_flux_funcs
        else:
            flux_funcs = self._default_flux_funcs

        if self.stars_type == 'sample':
            star = sample_star(
                rng=self._rng,
                star_data=self._example_stars,
                flux_funcs=flux_funcs,
                bands=self.bands,
            )
        else:
            star = sample_fixed_star(
                rng=self._rng,
                mag=self.stars_kws['mag'],
                bands=self.bands,
                flux_funcs=flux_funcs,
            )

        return star

    def _get_wcs_for_band(self, band):
        """
        return list of all wcs, one for each epoch in this band
        """
        if not hasattr(self, '_band_wcs_objs'):
            self._gen_all_wcs()

        return self._band_wcs_objs[band]

    def _gen_all_wcs(self):
        """
        generate a wcs for each band and epoch.  The result is stored
        in a dict of lists

        self._band_wcs_objs[band][epoch]
        """
        wcs_kws = dict(
            position_angle_range=self.wcs_kws.get('position_angle_range', (0, 0)),
            dither_range=self.wcs_kws.get('dither_range', (0, 0)),
            scale_frac_std=self.wcs_kws.get('scale_frac_std', 0),
            shear_std=self.wcs_kws.get('shear_std', 0),
            scale=self.scale,
            world_origin=self._world_origin,
            origin=self._se_origin,
            rng=self._rng,
        )

        self._band_wcs_objs = {}
        for band in self.bands:
            self._band_wcs_objs[band] = []
            for i in range(self.epochs_per_band):
                twcs = self.wcs_func(**wcs_kws)
                self._band_wcs_objs[band].append(twcs)

    def _init_ps_psfs(self):
        if not hasattr(self, '_ps_psf_objs'):
            self._ps_psf_objs = {}
            for band in self.bands:
                self._ps_psf_objs[band] = []
                for _ in range(self.epochs_per_band):
                    self._ps_psf_objs[band].append(
                        PowerSpectrumPSF(
                            rng=self._rng,
                            im_width=self.se_dim,
                            buff=self.se_dim/2,
                            scale=self.scale,
                            **self.psf_kws,
                        )
                    )

    def _get_psf_funcs_for_band(self, band):
        psf_funcs_galsim = []
        psf_funcs = []
        for epoch in range(self.epochs_per_band):
            pgf, prf = self._get_psf_funcs_for_band_epoch(band, epoch)
            psf_funcs_galsim.append(pgf)
            psf_funcs.append(prf)
        return psf_funcs_galsim, psf_funcs

    def _get_psf_funcs_for_band_epoch(self, band, epoch):
        # using closures here to capture some local state
        # we could make separate objects, but this seems lighter and easier

        if self.psf_type == 'ps':
            self._init_ps_psfs()

        se_wcs = self._get_wcs_for_band(band)[epoch]

        def _psf_galsim_func(*, x, y):
            if self.psf_type == 'gauss':
                kws = copy.deepcopy(self.psf_kws)
                g1 = kws.pop('g1')
                g2 = kws.pop('g2')
                psf = (
                    galsim.Gaussian(**kws)
                    .shear(g1=g1, g2=g2)
                    .withFlux(1.0)
                )

            elif self.psf_type == 'moffat':
                kws = copy.deepcopy(self.psf_kws)
                g1 = kws.pop('g1')
                g2 = kws.pop('g2')
                psf = (
                    galsim.Moffat(**kws)
                    .shear(g1=g1, g2=g2)
                    .withFlux(1.0)
                )
            elif self.psf_type == 'ps':
                pos = galsim.PositionD(x=x, y=y)
                psf = self._ps_psf_objs[band][epoch].getPSF(pos)
            else:
                raise ValueError('psf_type "%s" not valid!' % self.psf_type)

            return psf

        def _psf_render_func(*, x, y, center_psf, get_offset=False):
            image_pos = galsim.PositionD(x=x, y=y)
            psf = _psf_galsim_func(x=x, y=y)
            if center_psf:
                offset = None
            else:
                offset = galsim.PositionD(
                    x=x-int(x+0.5),
                    y=y-int(y+0.5),
                )
            gsimage = psf.drawImage(
                nx=self.psf_dim,
                ny=self.psf_dim,
                offset=offset,
                wcs=se_wcs.local(image_pos=image_pos),
            )
            if get_offset:
                return gsimage, offset
            else:
                return gsimage

        return _psf_galsim_func, _psf_render_func


def add_bright_star_masks(obj_data, band_data):
    """
    add a bright star mask in all bands
    """

    bands = list(band_data.keys())
    for odata in obj_data:
        if odata[bands[0]]['type'] == 'star':

            if any(odata[band]['is_bright'] for band in odata):

                for band in band_data:

                    nepoch = len(band_data[band])
                    for epoch in range(nepoch):
                        radius = odata[band]['radius'][epoch]

                        if radius > 0:
                            pos = odata[band]['pos'][epoch]
                            bmask = band_data[band][epoch].bmask.array
                            add_bright_star_mask(
                                mask=bmask, x=pos.x, y=pos.y,
                                radius=radius,
                                val=BRIGHT,
                            )


def subtract_bright_stars(obj_data, band_data, factors):
    """
    subtract bright stars in all bands
    """

    bands = list(band_data.keys())
    for odata in obj_data:
        if odata[bands[0]]['type'] == 'star':

            if any(odata[band]['is_bright'] for band in odata):

                for iband, band in enumerate(band_data):
                    fac = factors[iband]

                    nepoch = len(band_data[band])
                    for epoch in range(nepoch):
                        if odata[band]['overlaps'][epoch]:
                            overlap = odata[band]['overlap'][epoch]
                            stamp = odata[band]['stamp'][epoch]

                            if fac is not None:
                                stamp = stamp*fac

                            image = band_data[band][epoch].image
                            image[overlap] -= stamp[overlap]


def get_flux(mag):
    return 10**(0.4 * (ZERO_POINT - mag))


def check_keys(dct, allowed_keys, name):
    for key in dct.keys():
        if key not in allowed_keys:
            raise ValueError("'%s' is not a valid key for %s" % (key, name))
