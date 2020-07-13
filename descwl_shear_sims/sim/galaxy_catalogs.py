import os
import copy
import galsim
import descwl

from .shifts import get_shifts
from .constants import SCALE
from ..cache_tools import cached_catalog_read


DEFAULT_FIXED_GAL_CONFIG = {
    "mag": 17.0,
    "hlr": 0.5,
}


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
