"""Class for making a mulit-band, multi-epoch sim w/ galsim.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
and cleaned up a fair bit.
"""
import os
import logging
import copy
import functools

import fitsio
import galsim
import numpy as np
from collections import OrderedDict

from .se_obs import SEObs
from .render_sim import render_objs_with_psf_shear
from .gen_tanwcs import gen_tanwcs
from .randsphere import randsphere
from .gen_masks import generate_cosmic_rays, generate_bad_columns

LOGGER = logging.getLogger(__name__)

GAL_KWS_DEFAULTS = {
    'exp': {'half_light_radius': 0.5},
    'wldeblend': {'ngals_factor': 1.0},
}

# default mask bits
COSMIC_RAY = np.int64(2**0)
BAD_COLUMN = np.int64(2**1)


@functools.lru_cache(maxsize=8)
def _cached_catalog_read(fname):
    return fitsio.read(fname)


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
        The total dimension of the coadd image. Default is 350.
    buff : int, optional
        The width of the buffer region in the coadd image where no objects are
        drawn. Default is 50.
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

        Default is 'gauss'.
    psf_kws : dict, optional
        A dictionary of options for constructing the PSF. These vary per
        psf type. Possible entries per psf type are 'fwhm' plus

            'gauss' - no extra options

        Default is 'fwhm=0.8'.
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

    Methods
    -------
    gen_sim()
        Generate a simulation.
    """
    def __init__(
        self, *,
        rng,
        epochs_per_band=10,
        noise_per_band=180,
        bands=('r', 'i', 'z'),
        g1=0.02,
        g2=0.0,
        shear_scene=True,
        scale=0.2,
        coadd_dim=350,
        buff=50,
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

        self.ngals = ngals
        self.grid_gals = grid_gals
        self.gal_type = gal_type

        self.gal_kws = gal_kws or {}
        gal_kws_defaults = copy.deepcopy(GAL_KWS_DEFAULTS[self.gal_type])
        gal_kws_defaults.update(self.gal_kws)
        self._final_gal_kws = gal_kws_defaults

        self.psf_type = psf_type
        self.psf_kws = psf_kws or {'fwhm': 0.8}
        self.wcs_kws = wcs_kws or {}

        self.cosmic_rays = cosmic_rays
        self.cosmic_rays_kws = cosmic_rays_kws or {}
        self.bad_columns = bad_columns
        self.bad_columns_kws = bad_columns_kws or {}

        # the SE image could be rotated, so we make it big enough to cover the
        # whole coadd region plus we make sure it is odd
        self.se_dim = int(np.ceil(self.coadd_dim * np.sqrt(2))) + 10
        if self.se_dim % 2 == 0:
            self.se_dim = self.se_dim + 1
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
        self._ra_range, self._dec_range = self._get_patch_ranges()

        self.area_sqr_arcmin = (
            # factor of 60 to arcmin
            (self._ra_range[1] - self._ra_range[0]) * 60 *
            # factor of 180 * 60 / pi to go from radians to arcmin
            180 * 60 / np.pi * (np.sin(self._dec_range[1] / 180.0 * np.pi) -
                                np.sin(self._dec_range[0] / 180.0 * np.pi)))
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
                * self._ngals_factor)

        # we only allow the sim class to be used once
        # so that method outputs can be cached safely as needed
        # this attribute gets set to True after it is used
        self.called = False

        LOGGER.info('simulating bands: %s', self.bands)

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

        self._wldeblend_cat = _cached_catalog_read(fname)
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

        all_objects, uv_offsets = self._generate_all_objects()

        band_data = OrderedDict()
        for band_ind, band in enumerate(self.bands):
            band_objects = [o[band] for o in all_objects]
            wcs_objects = self._get_wcs_for_band(band)
            psf_funcs_galsim, psf_funcs_rendered = self._get_psf_funcs_for_band(band)

            band_data[band] = []
            for epoch, wcs, psf_galsim, psf_func in zip(
                    range(self.epochs_per_band), wcs_objects,
                    psf_funcs_galsim, psf_funcs_rendered):

                se_image = render_objs_with_psf_shear(
                        objs=band_objects,
                        psf_function=psf_galsim,
                        uv_offsets=uv_offsets,
                        wcs=wcs,
                        img_dim=self.se_dim,
                        method='auto',
                        g1=self.g1,
                        g2=self.g2,
                        shear_scene=self.shear_scene)

                se_image += self._generate_noise_image(band_ind)

                # put this after to make sure bad cols are totally dark
                bmask, se_image = self._generate_mask_plane(se_image)

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

    def _generate_mask_plane(self, se_image):
        shape = se_image.array.shape
        bmask = np.zeros(se_image.array.shape, dtype=np.int64)

        area_factor = (
            (self.coadd_dim - 2 * self.buff)**2
            / self.se_dim**2)

        if self.cosmic_rays:
            defaults = {
                'mean_cosmic_rays': 1,
            }
            defaults.update(self.cosmic_rays_kws)
            defaults['mean_cosmic_rays'] = int(
                defaults['mean_cosmic_rays']
                / area_factor
            )
            msk = generate_cosmic_rays(
                shape=shape,
                rng=self._rng,
                **defaults,
            )
            bmask[msk] |= COSMIC_RAY
            se_image.array[msk] = 1e18

        if self.bad_columns:
            defaults = {
                'mean_bad_cols': 1,
            }
            defaults.update(self.bad_columns_kws)
            defaults['mean_bad_cols'] = int(
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

        return (
            galsim.Image(
                bmask,
                bounds=se_image.bounds,
                wcs=se_image.wcs,
                dtype=np.int32),
            se_image,
        )

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

    def _generate_all_objects(self):
        """Generate all objects in all bands.

        Returns
        -------
        all_band_obj : list of OrderedDicts
            A list the length of the number of galaxies with an OrderedDict
            for each object holding each galaxies galsim object in each band.
        uv_offsets : list of galsim.PositionD
            A list of the uv-plane offsets in the world coordinates for each
            galaxy.
        """
        all_band_obj = []
        uv_offsets = []

        nobj = self._get_nobj()

        LOGGER.info('drawing %d objects for a %f square arcmin patch',
                    nobj, self.area_sqr_arcmin)

        for i in range(nobj):
            # unsheared offset from center of uv image
            du, dv = self._get_dudv()
            duv = galsim.PositionD(x=du, y=dv)

            # get the galaxy
            if self.gal_type == 'exp':
                gals = self._get_gal_exp()
            elif self.gal_type == 'wldeblend':
                gals = self._get_gal_wldeblend()
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            all_band_obj.append(gals)
            uv_offsets.append(duv)

        return all_band_obj, uv_offsets

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
            ra, dec = randsphere(
                self._rng, 1, ra_range=self._ra_range, dec_range=self._dec_range)
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
        flux = 10**(0.4 * (30 - 18))

        _gal = OrderedDict()
        for band in self.bands:
            obj = galsim.Exponential(
                **self._final_gal_kws
            ).withFlux(flux)
            _gal[band] = obj

        return _gal

    def _get_gal_wldeblend(self):
        """Return an OrderedDict keyed on band with the galsim object for
        a given exp gal."""

        _gal = OrderedDict()

        rind = self._rng.choice(self._wldeblend_cat.size)
        angle = self._rng.uniform() * 360

        for band in self.bands:
            _gal[band] = self._builders[band].from_catalog(
                self._wldeblend_cat[rind], 0, 0,
                self._surveys[band].filter_band).model.rotate(
                    angle * galsim.degrees)

        return _gal

    def _get_wcs_for_band(self, band):
        if not hasattr(self, '_band_wcs_objs'):
            wcs_kws = dict(
                position_angle_range=self.wcs_kws.get('position_angle_range', (0, 0)),
                dither_range=self.wcs_kws.get('dither_range', (0, 0)),
                scale_frac_std=self.wcs_kws.get('scale_frac_std', 0),
                shear_std=self.wcs_kws.get('shear_std', 0),
                scale=self.scale,
                world_origin=self._world_origin,
                origin=self._se_origin,
                rng=self._rng)

            self._band_wcs_objs = {}
            for band in self.bands:
                self._band_wcs_objs[band] = []
                for _ in range(self.epochs_per_band):
                    self._band_wcs_objs[band].append(gen_tanwcs(**wcs_kws))

        return self._band_wcs_objs[band]

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
        se_wcs = self._get_wcs_for_band(band)[epoch]

        def _psf_galsim_func(*, x, y):
            if self.psf_type == 'gauss':
                psf = galsim.Gaussian(**self.psf_kws).withFlux(1.0)
                return psf
            else:
                raise ValueError('psf_type "%s" not valid!' % self.psf_type)

        def _psf_render_func(*, x, y):
            image_pos = galsim.PositionD(x=x, y=y)
            psf = _psf_galsim_func(x=x, y=y)
            return psf.drawImage(
                nx=self.psf_dim,
                ny=self.psf_dim,
                wcs=se_wcs.local(image_pos=image_pos))

        return _psf_galsim_func, _psf_render_func
