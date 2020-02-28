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
from .render_sim import render_objs_with_psf_shear
from .gen_tanwcs import gen_tanwcs
from .randsphere import randsphere, randcap
from .gen_masks import (
    generate_basic_mask,
    generate_cosmic_rays,
    generate_bad_columns,
)
from .gen_star_masks import StarMaskPDFs, add_star_and_bleed
from .stars import (
    sample_star,
    sample_fixed_star,
    load_sample_stars,
)

from .ps_psf import PowerSpectrumPSF

# default mask bits from the stack
from .lsst_bits import BAD_COLUMN, COSMIC_RAY, SAT_VAL
from .cache_tools import cached_catalog_read

LOGGER = logging.getLogger(__name__)

GAL_KWS_DEFAULTS = {
    'exp': {'half_light_radius': 0.5},
    'wldeblend': {'ngals_factor': 1.0},
}
STARS_KWS_DEFAULTS = {
    'type': 'fixed',
    'density': 1,
}
SAT_STARS_KWS_DEFAULTS = {
    # density of sat starsper square arcmin when star type is fixed
    # not used for star type sample, instead the MAG_SAT is used
    'density': 0.0555,  # per square arcmin, 200 per sq degree
}
PSF_KWS_DEFAULTS = {
    'gauss': {'fwhm': 0.8, 'g1': 0.0, 'g2': 0.0},
    'ps': {},  # the defaults for the ps PSF are in the class
}

# TODO get a realistic value
EXP_GAL_MAG = 18.0


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
        is 180 units.
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
    ngals : float, optional
        The number of objects to simulate per arcminute^2. Default is 80.
    grid_gals : bool, optional
        If True, `ngals` will be interpreted as the number of
        galaxies to put on a grid per side.
    gal_type : str, optional
        A string indicating what kind of galaxy to simulate. Possible options are

            'exp' : simple exponential disks
            'wldeblend' : use the LSSTDESC WeakLensingDeblending package

        Default is 'exp'.
    gal_kws : dict, optional
        A dictionary of options for constructing galaxies. These vary per
        galaxy type. Possible entries per galaxy type are

            'exp' - any of the keywords to `galsim.Exponential`
                Default is 'half_light_radius=0.5'.

            'wldeblend' - any of the following keywords
                'catalog' : str
                    A path to the catalog to draw from. If this keyword is not
                    given, you need to have the one square degree catsim catalog
                    in the current working directory or in the directory given by
                    the environment variable 'CATSIM_DIR'.
                'ngals_factor' : float
                    A factor by which to cut down the catsim catalog. The
                    Default for this is 1.0 in order to better match LSST
                    depths and number densities.

    psf_type : str, optional
        A string indicating the kind of PSF. Possible options are

            'gauss' : a Gaussian PSF
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
            widths : n-tuple of ints, optional
                The possible widths of the bad columns.
            p : n-tuple of floats, optional
                The frequency of each of the bad column widths.
            min_length_frac : n-tuple of floats, optional
                The minimum fraction of the image the bad column spans. There should be
                one entry per bad column width in `widths`.
            max_length_frac : n-tuple of floats, optional
                The maximum fraction of the image the bad column spans. There should be
                one entry per bad column width in `widths`.
            gap_prob : n-tuple of floats, optional
                The probability that the bad column has a gap in it. There should be
                one entry per bad column width in `widths`.
            min_gap_frac : n-tuple of floats, optional
                The minimum fraction of the image that the gap spans. There should be
                one entry per bad column width in `widths`.
            max_gap_frac : n-tuple of floats, optional
                The maximum fraction of the image that the gap spans. There should be
                one entry per bad column width in `widths`.

        See descwl_shear_sims.gen_masks.generate_bad_columns for the defaults.

    stars: bool, optional
    stars_kws: dict, optional
        A dictionary of options for generating stars

            type: type of star, for now just "fixed" is allowed, which
                means a fixed brightness at mag 19 with the same mean
                number as the number of galaxies
            density: number per square arcmin, default 20

    sat_stars: bool, optional
        If `True` then add star and bleed trail masks. Default is `False`.
    sat_stars_kws : dict, optional
        A dictionary of options for generating star and bleed trail masks

            density: float
                Number of saturated stars per square arcmin, default 0.055
            radmean: float
                Mean star mask radius in pixels for log normal distribution,
                default 3
            radstd: float
                Radius standard deviation in pixels for log normal distribution,
                default 5
            radmin: float
                Minimum radius in pixels of star masks.  The log normal values
                will be clipped to more than this value.  Default 3
            radmax: float
                Maximum radius in pixels of star masks.  The log normal values
                will be clipped to less than this value. Default 500
            bleed_length_fac: float
                The bleed length is this factor times the *diameter* of the
                circular star mask, default 2


    Methods
    -------
    gen_sim()
        Generate a simulation.
    """
    def __init__(
        self, *,
        rng,
        epochs_per_band=10,
        noise_per_band=100,
        bands=('r', 'i', 'z'),
        g1=0.02,
        g2=0.0,
        shear_scene=True,
        scale=0.2,
        coadd_dim=351,
        buff=50,
        cap_radius=None,
        edge_width=5,
        se_dim=None,
        ngals=80,
        grid_gals=False,
        gal_type='exp',
        gal_kws=None,
        psf_type='gauss',
        psf_kws=None,
        wcs_kws=None,
        cosmic_rays=False,
        cosmic_rays_kws=None,
        bad_columns=False,
        bad_columns_kws=None,
        stars=False,
        stars_kws=None,
        sat_stars=False,
        sat_stars_kws=None,
    ):
        self._rng = (
            rng
            if isinstance(rng, np.random.RandomState)
            else np.random.RandomState(seed=rng))

        # we set these here so they are seeded once - no calls to the RNG
        # should preceed these calls
        self._noise_rng = np.random.RandomState(
            seed=self._rng.randint(1, 2**32-1))
        self._galsim_rng = galsim.BaseDeviate(
            seed=self._rng.randint(low=1, high=2**32-1))

        self.epochs_per_band = epochs_per_band
        self.bands = bands
        self.n_bands = len(bands)
        self.noise_per_band = np.array(noise_per_band) * np.ones(self.n_bands)
        self.noise_per_epoch = self.noise_per_band * np.sqrt(self.epochs_per_band)

        self.g1 = g1
        self.g2 = g2
        self.shear_scene = shear_scene

        self.scale = scale
        self.coadd_dim = coadd_dim
        self.buff = buff
        self.cap_radius = cap_radius
        if self.cap_radius is not None:
            self.buff = 0

        self.edge_width = edge_width
        assert edge_width >= 2, 'edge width must be >= 2'

        self.ngals = ngals
        self.grid_gals = grid_gals
        self.gal_type = gal_type

        self.gal_kws = gal_kws or {}
        gal_kws_defaults = copy.deepcopy(GAL_KWS_DEFAULTS[self.gal_type])
        gal_kws_defaults.update(self.gal_kws)
        self._final_gal_kws = gal_kws_defaults

        self.psf_type = psf_type
        self.psf_kws = copy.deepcopy(PSF_KWS_DEFAULTS[self.psf_type])
        if psf_kws is not None:
            self.psf_kws.update(psf_kws)

        self.wcs_kws = wcs_kws or {}

        self.cosmic_rays = cosmic_rays
        self.cosmic_rays_kws = cosmic_rays_kws or {}
        self.bad_columns = bad_columns
        self.bad_columns_kws = bad_columns_kws or {}

        # the SE image could be rotated, so we make it big enough to cover the
        # whole coadd region plus we make sure it is odd
        default_se_dim = (
            int(np.ceil(self.coadd_dim * np.sqrt(2))) + 10 + 2*edge_width
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

        ######################################
        # wcs info
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

        # info about coadd PSF image
        self.psf_dim = 53
        self._psf_cen = (self.psf_dim - 1)/2

        # now we call any extra init for wldeblend
        # this call will reset some things above
        self._ngals_factor = 1.0
        if self.gal_type == 'wldeblend':
            self._extra_init_for_wldeblend()

        # reset nobj to the number in a grid if we are using one
        if self.grid_gals:
            self._gal_grid_ind = 0
            self._nobj = self.ngals * self.ngals
        else:
            self._nobj = int(
                self.ngals
                * self.area_sqr_arcmin
                * self._ngals_factor
            )

        self._setup_stars(
            stars=stars,
            stars_kws=stars_kws,
            sat_stars=sat_stars,
            sat_stars_kws=sat_stars_kws,
        )

        # we only allow the sim class to be used once
        # so that method outputs can be cached safely as needed
        # this attribute gets set to True after it is used
        self.called = False

        LOGGER.info('simulating bands: %s', self.bands)

    def _setup_stars(self,
                     *,
                     stars,
                     stars_kws,
                     sat_stars,
                     sat_stars_kws):

        if stars and self.grid_gals:
            raise ValueError('no grid gals when there are stars')

        self.stars = stars
        self.sat_stars = sat_stars

        self.stars_kws = {}
        self.stars_kws.update(copy.deepcopy(STARS_KWS_DEFAULTS))

        if stars_kws is not None:
            self.stars_kws.update(copy.deepcopy(stars_kws))

        assert self.stars_kws['type'] in ('sample', 'fixed')
        if self.stars_kws['type'] == 'sample':
            assert self.gal_type == 'wldeblend', (
                'gal type must be wldeblend for star type sample',
            )

        # currently fixed average number of stars in every
        # image

        self._nstars = int(
            self.stars_kws['density'] * self.area_sqr_arcmin,
        )

        use_sat_kws = copy.deepcopy(SAT_STARS_KWS_DEFAULTS)

        if sat_stars_kws is not None:
            use_sat_kws.update(copy.deepcopy(sat_stars_kws))

        # density per square arcmin. Pop it because it is not
        # used by the pdf

        sat_density = use_sat_kws.pop('density', 0.0)

        if sat_stars:
            self.star_mask_pdf = StarMaskPDFs(
                rng=self._rng,
                **use_sat_kws
            )

            if self.stars_kws['type'] == 'fixed':
                self.sat_stars_frac = sat_density/self.stars_kws['density']
            else:
                self._example_stars = load_sample_stars()
        else:
            self.sat_stars_frac = 0.0
            self._example_stars = None

    def _get_nstars(self):
        if self.stars:
            return self._rng.poisson(self._nstars)
        else:
            return 0

    def _extra_init_for_wldeblend(self):
        # guard the import here
        import descwl

        # make sure to find the proper catalog
        if 'catalog' not in self._final_gal_kws:
            fname = os.path.join(
                os.environ.get('CATSIM_DIR', '.'),
                'OneDegSq.fits')
        else:
            fname = self._final_gal_kws['catalog']

        self._wldeblend_cat = cached_catalog_read(fname)
        self._wldeblend_cat['pa_disk'] = self._rng.uniform(
            low=0.0, high=360.0, size=self._wldeblend_cat.size)
        self._wldeblend_cat['pa_bulge'] = self._wldeblend_cat['pa_disk']

        self._surveys = {}
        self._builders = {}
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
            self._builders[band] = descwl.model.GalaxyBuilder(
                survey=self._surveys[band],
                no_disk=False,
                no_bulge=False,
                no_agn=False,
                verbose_model=False)

            noises.append(np.sqrt(self._surveys[band].mean_sky_level))

        self.noise_per_band = np.array(noises)
        self.noise_per_epoch = self.noise_per_band * np.sqrt(self.epochs_per_band)

        # when we sample from the catalog, we need to pull the right number
        # of objects. Since the default catalog is one square degree
        # and we fill a fraction of the image, we need to set the
        # base source density `ngal`. This is in units of number per
        # square arcminute.
        self.ngals = self._wldeblend_cat.size / (60 * 60)
        LOGGER.info('catalog density: %f per sqr arcmin', self.ngals)

        # we use a factor to make sure the depth matches that in
        # the real data
        self._ngals_factor = self._final_gal_kws['ngals_factor']

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

        all_gal_data, uv_offsets_gals = self._generate_gals()
        all_star_data, uv_offsets_stars = self._generate_stars()

        all_data = all_gal_data + all_star_data
        uv_offsets = uv_offsets_gals + uv_offsets_stars

        band_data = OrderedDict()
        for band_ind, band in enumerate(self.bands):
            band_objs = [o[band] for o in all_data]

            wcs_objects = self._get_wcs_for_band(band)

            psf_funcs_galsim, psf_funcs_rendered = \
                self._get_psf_funcs_for_band(band)

            band_data[band] = []
            for epoch, wcs, psf_galsim, psf_func in zip(
                    range(self.epochs_per_band), wcs_objects,
                    psf_funcs_galsim, psf_funcs_rendered):

                se_image, overlap_info = render_objs_with_psf_shear(
                    objs=band_objs,
                    uv_offsets=uv_offsets,
                    psf_function=psf_galsim,
                    wcs=wcs,
                    img_dim=self.se_dim,
                    method='auto',
                    g1=self.g1,
                    g2=self.g2,
                    shear_scene=self.shear_scene,
                )

                se_image += self._generate_noise_image(band_ind)

                # put this after to make sure bad cols are totally dark
                bmask, se_image = self._generate_mask_plane(
                    se_image=se_image,
                    wcs=wcs,
                    objs=band_objs,
                    overlap_info=overlap_info,
                )

                # make galsim image with same wcs as se_image but
                # with pure random noise
                noise_example = self._generate_noise_image(band_ind)
                noise_image = se_image.copy()
                noise_image.array[:, :] = noise_example

                se_weight = (
                    se_image.copy() * 0
                    + 1.0 / self.noise_per_epoch[band_ind]**2
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

        return band_data

    def _generate_mask_plane(self, *, se_image, wcs, objs, overlap_info):
        """
        set masks for edges, cosmics, bad columns and saturated stars/bleeds
        """

        shape = se_image.array.shape
        bmask = generate_basic_mask(shape=shape, edge_width=self.edge_width)

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
            se_image.array[msk] = SAT_VAL

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

        if self.stars and self.sat_stars:
            for obj_data, info in zip(objs, overlap_info):
                if (info['overlaps'] and
                        obj_data['type'] == 'star' and
                        obj_data['saturated']):

                    pos = info['pos']

                    sat_data = obj_data['sat_data']
                    add_star_and_bleed(
                        mask=bmask,
                        image=se_image.array,
                        x=pos.x,
                        y=pos.y,
                        radius=sat_data['radius'],
                        bleed_width=sat_data['bleed_width'],
                        bleed_length=sat_data['bleed_length'],
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

    def _generate_gals(self):
        """Generate all galaxies in all bands.

        Returns
        -------
        all_band_obj : list of OrderedDicts
            A list the length of the number of galaxies with an OrderedDict
            for each object holding each galaxies galsim object in each band.
        uv_offsets : list of galsim.PositionD
            A list of the uv-plane offsets in the world coordinates for each
            galaxy.
        """

        nobj = self._get_nobj()

        LOGGER.info('drawing %d galaxies for a %f square arcmin patch',
                    nobj, self.area_sqr_arcmin)

        all_data = []
        uv_offsets = []
        for i in range(nobj):
            # unsheared offset from center of uv image
            du, dv = self._get_dudv()
            duv = galsim.PositionD(x=du, y=dv)

            # get the galaxy
            if self.gal_type == 'exp':
                gal_data = self._get_gal_exp()
            elif self.gal_type == 'wldeblend':
                gal_data = self._get_gal_wldeblend()
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            all_data.append(gal_data)
            uv_offsets.append(duv)

        return all_data, uv_offsets

    def _generate_stars(self):
        """Generate all stars in all bands.

        Returns
        -------
        all_band_obj : list of OrderedDicts
            A list the length of the number of stars with an OrderedDict
            for each object holding each stars galsim object in each band.
        uv_offsets : list of galsim.PositionD
            A list of the uv-plane offsets in the world coordinates for each
            star.
        """

        nstars = self._get_nstars()

        LOGGER.info('drawing %d stars for a %f square arcmin patch',
                    nstars, self.area_sqr_arcmin)

        all_data = []
        uv_offsets = []
        for i in range(nstars):
            # unsheared offset from center of uv image
            du, dv = self._get_dudv()
            duv = galsim.PositionD(x=du, y=dv)

            # get the star as an dict by band
            star_data = self._get_star()

            all_data.append(star_data)
            uv_offsets.append(duv)

        return all_data, uv_offsets

    def _get_dudv(self):
        """Return an offset from the center of the image in the (u, v) plane
        of the coadd."""
        if self.grid_gals:
            # half of the width of center of the patch that has objects
            # object locations should be in [-pos_width, +pos_width] below
            # we are making a grid in (u,v) as opposed to ra-dec so the spacing
            # is even
            frac = 1.0 - self.buff * 2 / self.coadd_dim
            _pos_width = self.coadd_dim * frac * 0.5 * self.scale
            yind, xind = np.unravel_index(
                self._gal_grid_ind, (self.ngals, self.ngals))
            dg = _pos_width * 2 / self.ngals
            self._gal_grid_ind += 1
            return (
                yind * dg + dg/2 - _pos_width,
                xind * dg + dg/2 - _pos_width)
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
        if self.grid_gals:
            return self._nobj
        else:
            return self._rng.poisson(self._nobj)

    def _get_gal_exp(self):
        """Return an OrderedDict keyed on band with the galsim object for
        a given exp gal."""
        flux = 10**(0.4 * (30 - EXP_GAL_MAG))

        _gal = OrderedDict()
        for band in self.bands:
            obj = galsim.Exponential(
                **self._final_gal_kws
            ).withFlux(flux)
            _gal[band] = {'obj': obj, 'type': 'galaxy'}

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
            _gal[band] = {'obj': obj, 'type': 'galaxy'}

        return _gal

    def _get_star(self):
        """

        Returns
        -------
        An OrderedDict keyed on band with data of theform
        {'obj': Gaussian, 'type': 'star'}
        """

        if self.stars_kws['type'] == 'sample':
            star = sample_star(
                rng=self._rng,
                star_data=self._example_stars,
                surveys=self._surveys,
                bands=self.bands,
                sat_stars=self.sat_stars,
                star_mask_pdf=self.star_mask_pdf,
            )
        else:
            star = sample_fixed_star(
                rng=self._rng,
                bands=self.bands,
                sat_stars=self.sat_stars,
                sat_stars_frac=self.sat_stars_frac,
                star_mask_pdf=self.star_mask_pdf,
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
                twcs = gen_tanwcs(**wcs_kws)
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
                return psf
            elif self.psf_type == 'ps':
                return self._ps_psf_objs[band][epoch].getPSF(galsim.PositionD(x=x, y=y))
            else:
                raise ValueError('psf_type "%s" not valid!' % self.psf_type)

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
