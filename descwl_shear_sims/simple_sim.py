"""Class for making a mulit-band, multi-epoch sim w/ galsim.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
and cleaned up a fair bit.
"""
import logging

import galsim
import numpy as np
from collections import OrderedDict

from .se_obs import SEObs
from .render_sim import render_objs_with_psf_shear
from .util import randsphere

LOGGER = logging.getLogger(__name__)

RA_CEN = 100.0
DEC_CEN = 45.0


class SimpleSim(object):
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
        The total dimension of the coadd image. Default is 300.
    buff : int, optional
        The width of the buffer region in the coadd image where no objects are
        drawn. Default is 25.
    ngals_per_arcmin2 : float, optional
        The number of objects to simulate per arcminute^2. Default is 80.
    grid_gals : bool, optional
        If True, `ngal` will be interpreted as the number of galaxies to put on a
        grid per side.
    gal_type : str, optional
        A string indicating what kind of galaxy to simulate. Possible options are

            'exp' : simple exponential disks

        Default is 'exp'.
    gal_kws : dict, optional
        A dictionary of options for constructing galaxies. These vary per
        galaxy type. Possible entries per galaxy type are

            'exp' - any of the keywords to `galsim.Exponential`

        Default is 'half_light_radius=0.5'.
    psf_type : str, optional
        A string indicating the kind of PSF. Possible options are

            'gauss' : a Gaussian PSF

        Default is 'gauss'.
    psf_kws : dict, optional
        A dictionary of options for constructing the PSF. These vary per
        psf type. Possible entries per psf type are 'fwhm' plus

            'gauss' - no extra options

        Default is 'fwhm=0.8'.

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
            coadd_dim=300,
            buff=25,
            ngals_per_arcmin2=80,
            grid_gals=False,
            gal_type='exp',
            gal_kws=None,
            psf_type='gauss',
            psf_kws=None):

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

        self.ngals_per_arcmin2 = ngals_per_arcmin2
        self.grid_gals = grid_gals
        self.gal_type = gal_type
        self.gal_kws = gal_kws or {'half_light_radius': 0.5}
        self.psf_type = psf_type
        self.psf_kws = psf_kws or {'fwhm': 0.8}

        self.area_sqr_arcmin = ((self.coadd_dim - 2*self.buff) * scale / 60)**2

        # the SE image could be rotated, so we make it big enough to cover the
        # whole coadd region plus we make sure it is odd
        self.se_dim = int(np.ceil(self.coadd_dim * np.sqrt(2))) + 10
        if self.se_dim % 2 == 0:
            self.se_dim = self.se_dim + 1

        # wcs info
        self._coadd_cen = (self.coadd_dim - 1)/2
        self._coadd_uv_cen = galsim.PositionD(
            x=self._coadd_cen * self.scale,
            y=self._coadd_cen * self.scale)

        # frac of a single dimension that is used for drawing objects
        frac = 1.0 - self.buff * 2 / self.coadd_dim

        # half of the width of center of the patch that has objects
        # object locations should be in [-pos_width, +pos_width] below
        self._pos_width = self.coadd_dim * frac * 0.5 * self.scale

        # used later to draw objects
        self._shear_mat = galsim.Shear(g1=self.g1, g2=self.g2).getMatrix()

        # reset nobj to the number in a grid if we are using one
        if self.grid_gals:
            self._gal_grid_ind = 0
            self._nobj = self.ngal * self.ngal
        else:
            self._nobj = int(
                self.ngals_per_arcmin2 * self.area_sqr_arcmin)

        # we only allow the sim class to be used once
        # so that method outputs can be cached safely as needed
        # this attribute gets set to True after it is used
        self.called = False

        LOGGER.info('simulating %d bands', self.n_bands)

        # info about coadd PSF image
        self.psf_dim = 53
        self._psf_cen = (self.psf_dim - 1)/2

        self._setup_se_ranges()

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
                        uv_cen=self._coadd_uv_cen,
                        wcs=wcs,
                        img_dim=self.se_dim,
                        method='auto',
                        g1=self.g1,
                        g2=self.g2,
                        shear_scene=self.shear_scene)

                se_image += self._generate_noise_image(band_ind)

                # make galsim image with same wcs as se_image but
                # with pure random noise
                noise_example = self._generate_noise_image(band_ind)
                noise_image = se_image.copy()
                noise_image.array[:, :] = noise_example

                se_weight = se_image.copy() * 0 + 1.0 / self.noise_per_band[band_ind]**2

                band_data[band].append(
                    SEObs(
                        image=se_image,
                        noise=noise_image,
                        weight=se_weight,
                        wcs=wcs,
                        psf_function=psf_func,
                    )
                )

        return band_data

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
            self.noise_per_band[band_ind]
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
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            all_band_obj.append(gals)
            uv_offsets.append(duv)

        return all_band_obj, uv_offsets

    def _get_dudv(self):
        """Return an offset from the center of the image in the (u, v) plane
        of the coadd."""
        if self.grid_gals:
            yind, xind = np.unravel_index(
                self._gal_grid_ind, (self.ngal, self.ngal))
            dg = self._pos_width * 2 / self.ngal
            self._gal_grid_ind += 1
            return (
                yind * dg + dg/2 - self._pos_width,
                xind * dg + dg/2 - self._pos_width)
        else:
            return self._rng.uniform(
                low=-self._pos_width,
                high=self._pos_width,
                size=2)

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
                **self.gal_kws
            ).withFlux(flux)
            _gal[band] = obj

        return _gal

    '''
    def _get_wcs_for_band(self, band):
        # only a simple pixel scale right now
        se_cen = (self.se_dim - 1) / 2
        wcs = galsim.AffineTransform(
            dudx=self.scale,
            dudy=0,
            dvdx=0,
            dvdy=self.scale,
            world_origin=self._coadd_uv_cen,
            origin=galsim.PositionD(x=se_cen, y=se_cen),
        )
        return [wcs] * self.epochs_per_band
    '''

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

    def _setup_se_ranges(self):
        """
        set up position range for input single epoch images

        Currently the offsets are uniform in both directions, with excursions
        equal to half the coadd size
        """

        self.coadd_ra, self.coadd_dec = 10, 0
        cosdec = np.cos(np.radians(self.coadd_dec))

        offset = self.coadd_dim/2*self.scale/3600
        self.ra_range = [
            self.coadd_ra - offset*cosdec,
            self.coadd_ra + offset*cosdec,
        ]
        self.dec_range = [
            self.coadd_dec - offset,
            self.coadd_dec + offset,
        ]

    def _get_se_image_pos(self):
        """
        random position within ra, dec ranges
        """
        rav, decv = randsphere(
            self._rng,
            1,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
        )
        return galsim.CelestialCoord(
            rav[0]*galsim.degrees,
            decv[0]*galsim.degrees,
        )

    def _get_theta(self):
        """
        random rotation
        """
        return self._rng.uniform(low=0, high=np.pi*2)

    def _get_random_wcs(self):
        """
        get a TanWCS for random offsets and random rotation
        """
        world_pos = self._get_se_image_pos()
        theta = self._get_theta()

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s), (s, c)))
        vec = np.array([[self.scale, 0], [0, self.scale]])

        cd = np.dot(rot, vec)

        se_cen = (self.se_dim - 1) / 2
        se_origin = galsim.PositionD(se_cen, se_cen)

        affine = galsim.AffineTransform(
            cd[0, 0],
            cd[0, 1],
            cd[1, 0],
            cd[1, 1],
            origin=se_origin,
        )

        return galsim.TanWCS(affine, world_origin=world_pos)

    def _get_wcs_for_band(self, band):
        """
        get wcs for all epochs
        """
        wcs_list = []

        for i in range(self.epochs_per_band):
            wcs = self._get_random_wcs()
            wcs_list.append(wcs)

        return wcs_list
