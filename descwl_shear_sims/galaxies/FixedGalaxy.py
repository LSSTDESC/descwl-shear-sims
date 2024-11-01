import copy

import galsim
import numpy as np
from galsim import DeVaucouleurs, Exponential

from ..constants import SCALE
from ..layout import Layout

DEFAULT_FIXED_GAL_CONFIG = {
    "mag": 17.0,
    "hlr": 0.5,
    "morph": "exp",
}


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


class FixedGalaxyCatalog(object):
    """
    Galaxies of fixed galsim type, flux, and size and shape.

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    layout: string | Layout, optional
        The layout of objects, either 'grid' or 'random'
    coadd_dim: int, optional
        dimensions of the coadd
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    pixel_scale: float, optional
        pixel scale in arcsec
    """
    def __init__(
        self, *,
        rng,
        mag,
        hlr,
        morph='exp',
        layout=None,
        coadd_dim=None,
        buff=0,
        pixel_scale=SCALE,
    ):
        self.gal_type = 'fixed'
        self.morph = morph
        self.mag = mag
        self.hlr = hlr

        if isinstance(layout, str):
            self.layout = Layout(layout, coadd_dim, buff, pixel_scale)
        else:
            assert isinstance(layout, Layout)
            self.layout = layout
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
        )

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indexes

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        indexes = None
        flux = survey.get_flux(self.mag)

        sarray = self.shifts_array
        objlist = []
        shifts = []
        redshifts = None
        for i in range(len(self)):
            objlist.append(self._get_galaxy(flux))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

    def _get_galaxy(self, flux):
        """
        get a galaxy object

        Parameters
        ----------
        flux: float
            Flux of object

        Returns
        --------
        galsim.GSObject
        """

        if self.morph == 'exp':
            gal = _generate_exp(hlr=self.hlr, flux=flux)
        elif self.morph == 'dev':
            gal = _generate_dev(hlr=self.hlr, flux=flux)
        elif self.morph == 'bd':
            gal = _generate_bd(hlr=self.hlr, flux=flux)
        elif self.morph == 'bdk':
            gal = _generate_bdk(hlr=self.hlr, flux=flux)
        else:
            raise ValueError(f"bad gal type '{self.morph}'")

        return gal


class GalaxyCatalog(FixedGalaxyCatalog):
    """
    Galaxies of fixed galsim type, but varying properties.

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    layout: string
        The layout of objects, either 'grid' or 'random'
    coadd_dim: int
        dimensions of the coadd
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    pixel_scale: float
        pixel scale in arcsec
    """
    def __init__(
        self, *,
        rng,
        mag,
        hlr,
        morph='exp',
        layout=None,
        coadd_dim=None,
        buff=0,
        pixel_scale=SCALE,
    ):
        super().__init__(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            pixel_scale=pixel_scale,
            layout=layout,
            mag=mag,
            hlr=hlr,
            morph=morph,
        )
        self.gal_type = 'varying'

        # we use this to ensure the same galaxies are generated in different
        # bands
        self.morph_seed = rng.randint(0, 2**31)
        self.gs_morph_seed = rng.randint(0, 2**31)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indexes

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        self._morph_rng = np.random.RandomState(self.morph_seed)
        self._gs_morph_rng = galsim.BaseDeviate(seed=self.gs_morph_seed)
        return super().get_objlist(survey=survey)

    def _get_galaxy(self, flux):
        """
        get a galaxy object

        Parameters
        ----------
        flux: float
            Flux of object

        Returns
        --------
        galsim.GSObject
        """

        if self.morph == 'exp':
            gal = _generate_exp(
                hlr=self.hlr, flux=flux, vary=True, rng=self._morph_rng,
            )
        elif self.morph == 'dev':
            gal = _generate_dev(
                hlr=self.hlr, flux=flux, vary=True, rng=self._morph_rng,
            )
        elif self.morph == 'bd':
            gal = _generate_bd(
                hlr=self.hlr, flux=flux,
                vary=True, rng=self._morph_rng,
            )
        elif self.morph == 'bdk':
            gal = _generate_bdk(
                hlr=self.hlr, flux=flux,
                vary=True,
                rng=self._morph_rng, gsrng=self._gs_morph_rng,
            )
        else:
            raise ValueError(f"bad morph '{self.morph}'")

        return gal


def _generate_exp(hlr, flux, vary=False, rng=None):
    gal = Exponential(half_light_radius=hlr, flux=flux)

    if vary:
        g1, g2 = _generate_g1g2(rng)
        gal = gal.shear(g1=g1, g2=g2)

    return gal


def _generate_dev(hlr, flux, vary=False, rng=None):
    gal = DeVaucouleurs(half_light_radius=hlr, flux=flux)
    if vary:
        g1, g2 = _generate_g1g2(rng)
        gal = gal.shear(g1=g1, g2=g2)

    return gal


def _generate_bd(
    hlr, flux,
    vary=False,
    rng=None,
    max_bulge_shift_frac=0.1,  # fraction of hlr
    max_bulge_rot=np.pi/4,
):

    if vary:
        bulge_frac = _generate_bulge_frac(rng)
    else:
        bulge_frac = 0.5

    disk_frac = (1.0 - bulge_frac)

    bulge = DeVaucouleurs(half_light_radius=hlr, flux=flux * bulge_frac)
    disk = Exponential(half_light_radius=hlr, flux=flux * disk_frac)

    if vary:
        bulge = _shift_bulge(rng, bulge, hlr, max_bulge_shift_frac)

    if vary:
        g1disk, g2disk = _generate_g1g2(rng)

        g1bulge, g2bulge = g1disk, g2disk
        if vary:
            g1bulge, g2bulge = _rotate_bulge(rng, max_bulge_rot, g1bulge, g2bulge)

        bulge = bulge.shear(g1=g1bulge, g2=g2bulge)
        disk = disk.shear(g1=g1disk, g2=g2disk)

    return galsim.Add(bulge, disk)


def _generate_bdk(
    hlr, flux,
    vary=False,
    rng=None,
    gsrng=None,
    knots_hlr_frac=0.25,
    max_knots_disk_frac=0.1,  # fraction of disk light
    max_bulge_shift_frac=0.1,  # fraction of hlr
    max_bulge_rot=np.pi/4,
):

    if vary:
        bulge_frac = _generate_bulge_frac(rng)
    else:
        bulge_frac = 0.5

    all_disk_frac = (1.0 - bulge_frac)

    knots_hlr = knots_hlr_frac * hlr
    if vary:
        knots_sub_frac = _generate_knots_sub_frac(rng, max_knots_disk_frac)
    else:
        knots_sub_frac = max_knots_disk_frac

    disk_frac = (1 - knots_sub_frac) * all_disk_frac
    knots_frac = knots_sub_frac * all_disk_frac

    bulge = DeVaucouleurs(half_light_radius=hlr, flux=flux * bulge_frac)
    disk = Exponential(half_light_radius=hlr, flux=flux * disk_frac)

    if gsrng is None:
        # fixed galaxy, so fix the rng
        gsrng = galsim.BaseDeviate(123)

    knots = galsim.RandomKnots(
        npoints=10,
        half_light_radius=knots_hlr,
        flux=flux * knots_frac,
        rng=gsrng,
    )

    if vary:
        bulge = _shift_bulge(rng, bulge, hlr, max_bulge_shift_frac)

    if vary:
        g1disk, g2disk = _generate_g1g2(rng)

        g1bulge, g2bulge = g1disk, g2disk
        if vary:
            g1bulge, g2bulge = _rotate_bulge(rng, max_bulge_rot, g1bulge, g2bulge)

        bulge = bulge.shear(g1=g1bulge, g2=g2bulge)
        disk = disk.shear(g1=g1disk, g2=g2disk)
        knots = knots.shear(g1=g1disk, g2=g2disk)

    return galsim.Add(bulge, disk, knots)


def _generate_bulge_frac(rng):
    assert rng is not None, 'send rng to generate bulge fraction'
    return rng.uniform(low=0.0, high=1.0)


def _generate_g1g2(rng, std=0.2):
    assert rng is not None, 'send rng to vary shape'
    while True:
        g1, g2 = rng.normal(scale=std, size=2)
        g = np.sqrt(g1**2 + g2**2)
        if abs(g) < 0.9999:
            break

    return g1, g2


def _generate_bulge_shift(rng, hlr, max_bulge_shift_frac):
    bulge_shift = rng.uniform(low=0.0, high=max_bulge_shift_frac*hlr)
    bulge_shift_angle = rng.uniform(low=0, high=2*np.pi)
    bulge_shiftx = bulge_shift * np.cos(bulge_shift_angle)
    bulge_shifty = bulge_shift * np.sin(bulge_shift_angle)

    return bulge_shiftx, bulge_shifty


def _shift_bulge(rng, bulge, hlr, max_bulge_shift_frac):
    bulge_shiftx, bulge_shifty = _generate_bulge_shift(
        rng, hlr, max_bulge_shift_frac,
    )
    return bulge.shift(bulge_shiftx, bulge_shifty)


def _rotate_bulge(rng, max_bulge_rot, g1, g2):
    assert rng is not None, 'send rng to rotate bulge'
    bulge_rot = rng.uniform(low=-max_bulge_rot, high=max_bulge_rot/4)
    return _rotate_shape(g1, g2, bulge_rot)


def _rotate_shape(g1, g2, theta_radians):
    twotheta = 2.0 * theta_radians

    cos2angle = np.cos(twotheta)
    sin2angle = np.sin(twotheta)
    g1rot = g1 * cos2angle + g2 * sin2angle
    g2rot = -g1 * sin2angle + g2 * cos2angle

    return g1rot, g2rot


def _generate_knots_sub_frac(rng, max_knots_disk_frac):
    assert rng is not None, 'send rng to generate knots sub frac'
    return rng.uniform(low=0.0, high=max_knots_disk_frac)


class FixedPairGalaxyCatalog(FixedGalaxyCatalog):
    """
    A pair of galaxies of fixed galsim type, flux, and size

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    sep: float
        Separation of pair in arcsec
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    """
    def __init__(self, *, rng, mag, hlr, sep, morph='exp'):
        self.gal_type = 'fixed'
        self.morph = morph
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.layout = Layout("pair")
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
            sep=sep,
        )


class PairGalaxyCatalog(GalaxyCatalog):
    """
    A pair of galaxies of fixed galsim type, flux, and size

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    sep: float
        Separation of pair in arcsec
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    """
    def __init__(self, *, rng, mag, hlr, sep, morph='exp'):
        self.gal_type = 'varying'
        self.morph = morph
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.morph_seed = rng.randint(0, 2**31)
        self.gs_morph_seed = rng.randint(0, 2**31)

        self.layout = Layout("pair")
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
            sep=sep,
        )
