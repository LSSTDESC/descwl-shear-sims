import os
import copy
import galsim
import descwl

from .shifts import get_shifts, get_pair_shifts
from .constants import SCALE
from .cache_tools import cached_catalog_read


DEFAULT_FIXED_GAL_CONFIG = {
    "mag": 17.0,
    "hlr": 0.5,
}


def make_galaxy_catalog(
    *,
    rng,
    gal_type,
    coadd_dim=None,
    buff=None,
    layout=None,
    gal_config=None,
    sep=None,
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
    sep: float, optional
        Separation of pair in arcsec for layout='pair'
    """
    if layout == 'pair':
        if sep is None:
            raise ValueError(
                f'send sep= for gal_type {gal_type} and layout {layout}'
            )
        gal_config = get_fixed_gal_config(config=gal_config)
        galaxy_catalog = FixedPairGalaxyCatalog(
            rng=rng,
            mag=gal_config['mag'],
            hlr=gal_config['hlr'],
            sep=sep,
        )
    else:
        if coadd_dim is None:
            raise ValueError(
                f'send coadd_dim= for gal_type {gal_type} and layout {layout}'
            )
        if buff is None:
            raise ValueError(
                f'send buff= for gal_type {gal_type} and layout {layout}'
            )

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
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    """
    def __init__(self, *, rng, coadd_dim, buff, layout, mag, hlr):
        self.gal_type = 'exp'
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.shifts_array = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout=layout,
        )

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects], [shifts]
        """

        flux = survey.get_flux(self.mag)

        sarray = self.shifts_array
        objlist = []
        shifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(flux))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))

        return objlist, shifts

    def _get_galaxy(self, flux):
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
        )


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
    """
    def __init__(self, *, rng, mag, hlr, sep):
        self.gal_type = 'exp'
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.shifts_array = get_pair_shifts(
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

        self.shifts_array = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
            nobj=nobj,
        )

        num = len(self)
        self.indices = self.rng.randint(
            0,
            self._wldeblend_cat.size,
            size=num,
        )

        self.angles = self.rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object

        Returns
        -------
        [galsim objects], [shifts]
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
        objlist = []
        shifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(builder, band, i))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))

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
    return cat
