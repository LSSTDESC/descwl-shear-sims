"""
Copied from https://github.com/beckermr/metadetect-sims under BSD
"""

import numpy as np
import galsim
import galsim.lensing_ps
import galsim.table
import galsim.utilities


class PowerSpectrumPSF(object):
    """Produce a spatially varying Moffat PSF according to the power spectrum
    given by Heymans et al. (2012).

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance.
    im_width : float
        The width of the image in pixels.
    buff : int
        An extra buffer of pixels for things near the edge.
    scale : float
        The pixel scale of the image
    trunc : float
        The truncation scale for the shape/magnification power spectrum
        used to generate the PSF variation.
    noise_level : float, optional
        If not `None`, generate a noise field to add to the PSF images with
        desired noise. A value of 1e-2 generates a PSF image with an
        effective signal-to-noise of ~250.
    variation_factor : float, optional
        This factor is used internally to scale the overall variance in the
        PSF shape power spectra and the change in the PSF size across the
        image. Setting this factor greater than 1 results in more variation
        and less than 1 results in less variation.
    median_seeing : float, optional
        The approximate median seeing for the PSF.

    Methods
    -------
    getPSF(pos)
        Get a PSF model at a given position.
    """
    def __init__(self, *,
                 rng, im_width, buff, scale, trunc=1,
                 noise_level=None, variation_factor=10,
                 median_seeing=0.8):
        self._rng = rng
        self._im_cen = (im_width - 1)/2
        self._scale = scale
        self._tot_width = im_width + 2 * buff
        self._x_scale = 2.0 / self._tot_width / scale
        self._noise_level = noise_level
        self._buff = buff
        self._variation_factor = variation_factor
        self._median_seeing = median_seeing

        # set the power spectrum and PSF params
        # Heymans et al, 2012 found L0 ~= 3 arcmin, given as 180 arcsec here.
        def _pf(k):
            return (k**2 + (1./180)**2)**(-11./6.) * np.exp(-(k*trunc)**2)
        self._ps = galsim.PowerSpectrum(
            e_power_function=_pf,
            b_power_function=_pf)
        ng = 128
        gs = max(self._tot_width * self._scale / ng, 1)
        self.ng = ng
        self.gs = gs
        seed = self._rng.randint(1, 2**30)
        self._ps.buildGrid(
            grid_spacing=gs,
            ngrid=ng,
            get_convergence=True,
            variance=(0.01 * variation_factor)**2,
            rng=galsim.BaseDeviate(seed))

        # cache the galsim LookupTable2D objects by hand to speed computations
        g1_grid, g2_grid, mu_grid = galsim.lensing_ps.theoryToObserved(
            self._ps.im_g1.array, self._ps.im_g2.array,
            self._ps.im_kappa.array)

        self._lut_g1 = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid, g1_grid.T,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5))
        self._lut_g2 = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid, g2_grid.T,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5))
        self._lut_mu = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid, mu_grid.T - 1,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5))

        self._g1_mean = self._rng.normal() * 0.01 * variation_factor
        self._g2_mean = self._rng.normal() * 0.01 * variation_factor

        if self._noise_level is not None and self._noise_level > 0:
            self._noise_field = self._rng.normal(
                size=(im_width + buff + 37, im_width + buff + 37)
            ) * noise_level

        def _getlogmnsigma(mean, sigma):
            logmean = np.log(mean) - 0.5*np.log(1 + sigma**2/mean**2)
            logvar = np.log(1 + sigma**2/mean**2)
            logsigma = np.sqrt(logvar)
            return logmean, logsigma

        lm, ls = _getlogmnsigma(self._median_seeing, 0.1)
        self._fwhm_central = np.exp(self._rng.normal() * ls + lm)

    def _get_lensing(self, pos):
        pos_x, pos_y = galsim.utilities._convertPositions(
            pos, galsim.arcsec, '_get_lensing')
        return (
            self._lut_g1(pos_x, pos_y),
            self._lut_g2(pos_x, pos_y),
            self._lut_mu(pos_x, pos_y)+1)

    def _get_atm(self, x, y):
        xs = (x + 1 - self._im_cen) * self._scale
        ys = (y + 1 - self._im_cen) * self._scale
        g1, g2, mu = self._get_lensing((xs, ys))

        if g1*g1 + g2*g2 >= 1.0:
            norm = np.sqrt(g1*g1 + g2*g2) / 0.5
            g1 /= norm
            g2 /= norm

        fwhm = self._fwhm_central / np.power(mu, 0.75)

        psf = galsim.Moffat(
            beta=2.5,
            fwhm=fwhm
        ).shear(
            g1=g1 + self._g1_mean, g2=g2 + self._g2_mean
        )

        return psf

    def getPSF(self, pos):  # noqa: N802
        """Get a PSF model at a given position.

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism object.
        """
        psf = self._get_atm(pos.x, pos.y)

        if self._noise_level is not None and self._noise_level > 0:
            xll = int(pos.x + self._buff - 16)
            yll = int(pos.y + self._buff - 16)
            assert xll >= 0 and xll+33 <= self._noise_field.shape[1]
            assert yll >= 0 and yll+33 <= self._noise_field.shape[0]

            stamp = self._noise_field[yll:yll+33, xll:xll+33].copy()
            psf += galsim.InterpolatedImage(
                galsim.ImageD(stamp, scale=self._scale),
                normalization="sb")

        return psf.withFlux(1.0)
