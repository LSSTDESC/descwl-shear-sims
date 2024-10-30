import numpy as np
import os
import copy
import galsim
import galsim.roman as roman
from galsim import DeVaucouleurs
from galsim import Exponential
import descwl
import pyarrow as pa
from .layout import Layout
from .constants import SCALE, ZERO_POINT
from .cache_tools import cached_catalog_read
import h5py
import time

DEFAULT_FIXED_GAL_CONFIG = {
    "mag": 17.0,
    "hlr": 0.5,
    "morph": "exp",
}


def make_galaxy_catalog(
    *,
    rng,
    gal_type,
    coadd_dim=None,
    buff=0,
    pixel_scale=SCALE,
    layout=None,
    gal_config=None,
    sep=None,
):
    """
    rng: numpy.random.RandomState
        Numpy random state
    gal_type: string
        'fixed', 'varying', 'wldeblend' or 'roman_ou2024'
    coadd_dim: int
        Dimensions of coadd
    buff: int, optional
        Buffer around the edge where no objects are drawn.  Ignored for
        layout 'grid'.  Default 0.
    pixel_scale: float
        pixel scale in arcsec
    layout: string, optional
        'grid' or 'random'.  Ignored for gal_type "wldeblend", otherwise
        required.
    gal_config: dict or None
        Can be sent for fixed galaxy catalog.  See DEFAULT_FIXED_GAL_CONFIG
        for defaults mag, hlr and morph
    sep: float, optional
        Separation of pair in arcsec for layout='pair'
    """

    if layout is None and gal_type == 'wldeblend':
        layout = 'random'

    if isinstance(layout, str):
        layout = Layout(
            layout_name=layout,
            coadd_dim=coadd_dim,
            buff=buff,
            pixel_scale=pixel_scale,
        )
    else:
        assert isinstance(layout, Layout)

    if layout.layout_name == 'pair':
        if sep is None:
            raise ValueError(
                f'send sep= for gal_type {gal_type} and layout {layout}'
            )
        gal_config = get_fixed_gal_config(config=gal_config)

        if gal_type in ['fixed', 'exp']:  # TODO remove exp
            cls = FixedPairGalaxyCatalog
        else:
            cls = PairGalaxyCatalog

        galaxy_catalog = cls(
            rng=rng,
            mag=gal_config['mag'],
            hlr=gal_config['hlr'],
            morph=gal_config['morph'],
            sep=sep,
        )

    else:
        if gal_type == 'wldeblend':
            galaxy_catalog = WLDeblendGalaxyCatalog(
                rng=rng,
                layout=layout,
            )
        elif gal_type in ['fixed', 'varying', 'exp']:  # TODO remove exp
            gal_config = get_fixed_gal_config(config=gal_config)

            if gal_type == 'fixed':
                cls = FixedGalaxyCatalog
            else:
                cls = GalaxyCatalog

            galaxy_catalog = cls(
                rng=rng,
                mag=gal_config['mag'],
                hlr=gal_config['hlr'],
                morph=gal_config['morph'],
                layout=layout,
            )

        else:
            raise ValueError(f'bad gal_type "{gal_type}"')
    return galaxy_catalog


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


class WLDeblendGalaxyCatalog(object):
    """
    Catalog of galaxies from wldeblend

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    layout: str|Layout, optional
    coadd_dim: int, optional
        Dimensions of the coadd
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    pixel_scale: float, optional
        pixel scale
    select_observable: list[str] | str
        A list of observables to apply selection
    select_lower_limit: list | ndarray
        lower limits of the slection cuts
    select_upper_limit: list | ndarray
        upper limits of the slection cuts
    """
    def __init__(
        self,
        *,
        rng,
        layout='random',
        coadd_dim=None,
        buff=None,
        pixel_scale=SCALE,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ):
        self.gal_type = 'wldeblend'
        self.rng = rng

        self._wldeblend_cat = read_wldeblend_cat(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

        # one square degree catalog, convert to arcmin
        density = self._wldeblend_cat.size / (60 * 60)
        if isinstance(layout, str):
            self.layout = Layout(layout, coadd_dim, buff, pixel_scale)
        else:
            assert isinstance(layout, Layout)
            self.layout = layout
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
            density=density,
        )

        # randomly sample from the catalog
        num = len(self)
        self.indices = self.rng.randint(
            0,
            self._wldeblend_cat.size,
            size=num,
        )
        # do a random rotation for each galaxy
        self.angles = self.rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indexes

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        band = survey.filter_band

        sarray = self.shifts_array
        indexes = []
        objlist = []
        shifts = []
        redshifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(builder, band, i))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))
            index = self.indices[i]
            indexes.append(index)
            redshifts.append(self._wldeblend_cat[index]["redshift"])

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

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

        angle = self.angles[i]

        galaxy = builder.from_catalog(
            self._wldeblend_cat[index],
            0,
            0,
            band,
        ).model.rotate(
            angle * galsim.degrees,
        )

        return galaxy


def read_wldeblend_cat(
    select_observable=None,
    select_lower_limit=None,
    select_upper_limit=None,
):
    """
    Read the catalog from the cache, but update the position angles each time

    Parameters
    ----------
    select_observable: list[str] | str
        A list of observables to apply selection
    select_lower_limit: list[float] | ndarray[float]
        lower limits of the slection cuts
    select_upper_limit: list[float] | ndarray[float]
        upper limits of the slection cuts

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
    if select_observable is not None:
        select_observable = np.atleast_1d(select_observable)
        if not set(select_observable) < set(cat.dtype.names):
            raise ValueError("Selection observables not in the catalog columns")
        mask = np.ones(len(cat)).astype(bool)
        if select_lower_limit is not None:
            select_lower_limit = np.atleast_1d(select_lower_limit)
            assert len(select_observable) == len(select_lower_limit)
            for nn, ll in zip(select_observable, select_lower_limit):
                mask = mask & (cat[nn] > ll)
        if select_upper_limit is not None:
            select_upper_limit = np.atleast_1d(select_upper_limit)
            assert len(select_observable) == len(select_upper_limit)
            for nn, ul in zip(select_observable, select_upper_limit):
                mask = mask & (cat[nn] <= ul)
        cat = cat[mask]
    return cat

class OpenUniverse2024RubinRomanCatalog(object):
    """
    Catalog of galaxies from OpenUniverse2024 Rubin-Roman input galaxies
    https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html

    Parameters
    ---------
    rng: np.random.RandomState
        The random number generator
    layout: str|Layout, optional
    coadd_dim: int, optional
        Dimensions of the coadd
    buff: int, optinal
        Buffer region with no objects, on all sides of image.
        Ignored for layout 'grid'. Default 0.
    pixel_scale: float, optional 
        pixel scale
    select_observable: list[str] | str
        A list of observables (data columns) to apply selection
    select_lower_limit: list | ndarray
        lower limits of the selection cuts
    select_upper_limit: list | ndarray
        upper limits of the selection cuts
    """

    def __init__(
        self,
        *,
        rng,
        layout='random',
        coadd_dim=None,
        buff=None,
        pixel_scale=SCALE,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ):
        self.gal_type = 'rubinroman'
        self.rng = rng

        self._ou2024rubinroman_cat, self.sed_cat, self.mask = read_ou2024rubinroman_cat(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )
        self.gal_ids = np.array(self._ou2024rubinroman_cat['galaxy_id'])
 
        # The catalog corresponds to an nside=32 healpix pixel.
        area_tot_arcmin = 60.**2 * (180./np.pi)**2 * 4. * np.pi / (12.*32.**2)
        density = len(self._ou2024rubinroman_cat) / area_tot_arcmin
        if isinstance(layout, str):
            self.layout = Layout(layout, coadd_dim, buff, pixel_scale)
        else:
            assert isinstance(layout, Layout)
            self.layout = layout
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
            density=density,
        )

        # Randomly sample from the catalog
        num = len(self)
        self.indices = self.rng.randint(
            0,
            len(self._ou2024rubinroman_cat),
            size=num,
        )
        # do a random rotation for each galaxy
        self.angles = self.rng.uniform(low=0, high=360, size=num)
    
    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indices

        Parameters
        ----------
        survey: object with survey parameters

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        if survey.name=="roman":
            filters = roman.getBandpasses(AB_zeropoint=True)
            self.bandpass = filters[survey.band]
        if survey.name=="lsst":
            fname_bandpass = os.path.join(
                os.environ.get('CATSIM_DIR', '.'),
                f'LSST_{survey.band}.dat',
            )
            #### zeropoint for LSST is hard-coded for now
            zeropoint = {"u":12.627880975839947,"g":14.486360566315488,
                "r":14.324886825965237,"i":13.998598224053055,
                "z":13.612736383992512,"y":12.785119907991263}
            self.bandpass = galsim.Bandpass(fname_bandpass, wave_type='nm').withZeropoint(zeropoint[survey.band])
    
        sarray = self.shifts_array
        indexes = []
        objlist = []
        shifts = []
        redshifts = []
        for i in range(len(self)):
            print(len(self),i)
            objlist.append(self._get_galaxy(survey, i))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))
            index = self.indices[i]
            indexes.append(index)
            redshifts.append(self._ou2024rubinroman_cat["redshift"][index])

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

    def _get_galaxy(self, survey, i):
        """
        Get a galaxy

        Parameters
        ----------
        survey: object with survey parameters
            see surveys.py
        i: int
            Index of galaxies
        
        Returns
        -------
        galsim.GSObject
        """
        index = self.indices[i]
        angle = self.angles[i]

        # From the SED data, read the SED for the galaxy.
        # Note that how SEDs are stored can be a bit confusing.
        gal_id = str(self.gal_ids[index])
        f_sed = self.sed_cat["galaxy"][gal_id[:9]][gal_id][()]
        wave_list = self.sed_cat["meta"]["wave_list"][()]
        galaxy = _generate_rubinroman_galaxies(
            self.rng,
            survey,
            self._ou2024rubinroman_cat.slice(index,1),
            f_sed,
            wave_list,
            self.bandpass,
        )

        return galaxy

def _generate_rubinroman_galaxies(rng, survey, entry, f_sed, wave_list, bandpass, build_from_sed=False):
    """
    Generate a GSObject from an entry
    from the OpenUniverse2024 Rubin-Roman catalog
    
    Parameters
    ---------
    rng: random number generator
    survey: object with survey parameters
    entry: pyarrow table (len=1)
        Galaxy properties, sliced from the pyarrow table
    f_sed: list[float]
        Flux density values of the SED
    wave_list: list[float]
        List of wavelengths corresponding to f_sed
    bandpass: galsim bandpass object
        Bandpass corresponding to this simulation
    build_from_sed: bool
        whether want to build galaxies out of SED (slow)
        if False, it generate a random bulge fraction

    Returns
    -------
    A galsim galaxy object: GSObject 
    """

    band = survey.band
    sname = survey.name

    bulge_hlr = np.array(entry['spheroidHalfLightRadiusArcsec'])[0]
    disk_hlr = np.array(entry['diskHalfLightRadiusArcsec'])[0]
    disk_shear1, disk_shear2 = np.array(entry['diskEllipticity1'])[0], np.array(entry['diskEllipticity2'])[0]
    bulge_shear1, bulge_shear2 = np.array(entry['spheroidEllipticity1'])[0], np.array(entry['spheroidEllipticity2'])[0]
    gal_id = np.array(entry['galaxy_id'])[0]
    mag = np.array(entry[sname+'_mag_'+band])[0]
    flux = survey.get_flux(mag)
    gal_id_str = str(gal_id)
    
    if build_from_sed:
        # set up bulge and disk for sed
        bulge_lookup = galsim.LookupTable( x = wave_list, f = f_sed[0])
        disk_lookup = galsim.LookupTable( x = wave_list, f = f_sed[1])
        knots_lookup = galsim.LookupTable( x = wave_list, f = f_sed[2])
        bulge_sed = galsim.SED(bulge_lookup, wave_type = 'Ang', flux_type='fnu', redshift = redshift)
        disk_sed = galsim.SED(disk_lookup, wave_type = 'Ang', flux_type='fnu', redshift = redshift)
        knots_sed = galsim.SED(knots_lookup, wave_type = 'Ang', flux_type='fnu', redshift = redshift)

        # light profile
        bulge = galsim.Sersic(4, half_light_radius=bulge_hlr).shear(g1 =bulge_shear1 , g2 = bulge_shear2)
        disk = galsim.Sersic(1, half_light_radius=disk_hlr).shear(g1 =disk_shear1 , g2 = disk_shear2)

        # Make galaxy object. Note that we are not drawing with knots, 
        # so we add the knots' SED to the
        # disk SED to preserve the correct flux.
        gal = bulge*bulge_sed + disk*(disk_sed + knots_sed)
        gal = gal.withFlux(flux, bandpass)
    else:
        bulge_frac = _generate_bulge_frac(rng)
        bulge = galsim.Sersic(4, half_light_radius=bulge_hlr, flux=flux*bulge_frac).shear(g1 =bulge_shear1 , g2 = bulge_shear2)
        disk = galsim.Sersic(1, half_light_radius=disk_hlr, flux=flux*(1.-bulge_frac)).shear(g1 =disk_shear1 , g2 = disk_shear2)
        gal = bulge + disk
        gal= gal.withFlux(flux)

    return gal


def read_ou2024rubinroman_cat(
    select_observable=None,
    select_lower_limit=None,
    select_upper_limit=None,
):
    """
    Read the catalog from the cache, but update the position angles each time

    Parameters
    ----------
    select_observable: list[str] | str
        A list of observables to apply selection
    select_lower_limit: list[float] | ndarray[float]
        lower limits of the slection cuts
    select_upper_limit: list[float] | ndarray[float]
        upper limits of the slection cuts

    Returns
    -------
    array with fields
    """
    ### galaxy catalog
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'galaxy_10307.parquet',
    )
    ### flux catalog (same length)
    fname_flux = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'galaxy_flux_10307.parquet',
    )
    fname_sed = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'galaxy_sed_10307.hdf5',
    )
    cat = cached_catalog_read(fname,format='parquet')
    cat_flux = cached_catalog_read(fname_flux,format='parquet')
    cat_sed = cached_catalog_read(fname_sed,format='h5py')

    ### Merge the flux catalog to the galaxy catalog
    for nn in cat_flux.column_names:
        if nn != 'galaxy_id':
            cat = cat.append_column(nn, cat_flux[nn])

    ### Got this from TianQing ==> need to properly calculate them in the future
    zeropoint = {"u":12.627880975839947,"g":14.486360566315488,
            "r":14.324886825965237,"i":13.998598224053055,
            "z":13.612736383992512,"y":12.785119907991263}
    roman_filters = roman.getBandpasses(AB_zeropoint=True)
    for band in ['W146','R062','Z087','Y106','J129','H158','F184','K213']:
        bandpass = roman_filters[band]
        zeropoint[band] = bandpass.zeropoint

    for filter_name in zeropoint.keys():
        if filter_name in 'ugrizy':
            flux_filter = cat['lsst_flux_'+filter_name]
            mag_filter = pa.compute.add(pa.compute.multiply(-2.5,pa.compute.log10(flux_filter)),zeropoint[band])
            cat = cat.append_column('lsst_mag_'+filter_name, mag_filter)
        else:
            flux_filter = cat['roman_flux_'+filter_name]
            mag_filter = pa.compute.add(pa.compute.multiply(-2.5,pa.compute.log10(flux_filter)),zeropoint[band])
            cat = cat.append_column('roman_mag_'+filter_name, mag_filter)

    if select_observable is not None:
        select_observable = np.atleast_1d(select_observable)
        if not set(select_observable) < set(cat.column_names):
            raise ValueError("Selection observables not in the catalog columns")
        mask = np.ones(len(cat)).astype(bool)
        if select_lower_limit is not None:
            select_lower_limit = np.atleast_1d(select_lower_limit)
            assert len(select_observable) == len(select_lower_limit)
            for nn, ll in zip(select_observable, select_lower_limit):
                mask = mask & (cat[nn] > ll)
        if select_upper_limit is not None:
            select_upper_limit = np.atleast_1d(select_upper_limit)
            assert len(select_observable) == len(select_upper_limit)
            for nn, ul in zip(select_observable, select_upper_limit):
                mask = mask & (cat[nn] <= ul)
        cat = cat[mask]
    else:
        mask = None
    return cat, cat_sed, mask
