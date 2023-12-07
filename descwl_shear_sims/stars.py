import os
from copy import deepcopy
import functools
import numpy as np
import fitsio
import galsim

from .constants import SCALE
from .cache_tools import cached_catalog_read
from .shifts import get_shifts

DEFAULT_MIN_STAR_DENSITY = 2    # unit: per square arcmin
DEFAULT_MAX_STAR_DENSITY = 100
DEFAULT_DENSITY = None

DEFAULT_STAR_CONFIG = {
    'min_density': DEFAULT_MIN_STAR_DENSITY,
    'max_density': DEFAULT_MAX_STAR_DENSITY,
    'density': DEFAULT_DENSITY,  # overrides sampling
}


def get_star_config(config=None):
    """
    get the configuration for a star catalog, with defaults in place

    Parameters
    ----------
    config: dict, optional
        The input config. Over-rides defaults

    Returns
    -------
    the config dict
    """
    out_config = deepcopy(DEFAULT_STAR_CONFIG)

    if config is not None:
        for key in config:
            if key not in out_config:
                raise ValueError("bad key for stars: '%s'" % key)
        out_config.update(config)

    return out_config


def make_star_catalog(rng, coadd_dim, buff=0, star_config=None, layout='random'):
    """
    Creat a StarCatalog

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int, optional
        Buffer around the edge where no objects are drawn. Default 0.
    star_config: dict
        Entries can be 'min_density', 'max_density' for sampling
        and 'density' to pick an exact density

    Returns
    ------
    StarCatalog
    """

    star_config = get_star_config(config=star_config)
    return StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=star_config['density'],
        min_density=star_config['min_density'],
        max_density=star_config['max_density'],
        layout=layout,
    )


class StarCatalog(object):
    """
    Star catalog with variable density

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int, optional
        Buffer around the edge where no objects are drawn. Default 0.
    pixel_scale: float
        pixel scale
    min_density: int, optional
        Set the minimum density to sample (ignored if density= is sent)
    max_density: int, optional
        Set the maximum density to sample (ignored if density= is sent)
    density: float, optional
        Optional density for catalog, if not sent the density is variable and
        drawn from the expected galactic density
    layout: string
        'random' or 'random_disk'
    """
    def __init__(
        self, *,
        rng,
        coadd_dim,
        buff=0,
        pixel_scale=SCALE,
        min_density=DEFAULT_MIN_STAR_DENSITY,
        max_density=DEFAULT_MAX_STAR_DENSITY,
        density=DEFAULT_DENSITY,
        layout='random',
    ):
        self.rng = rng

        self._star_cat = load_sample_stars()

        if density is None:
            density_mean = sample_star_density(
                rng=self.rng,
                min_density=min_density,
                max_density=max_density,
            )
        else:
            density_mean = density

        if layout == 'random':
            # this layout is random in a square
            area = ((coadd_dim - 2*buff)*pixel_scale/60)**2
        elif layout == 'random_disk':
            # this layout is random in a circle
            radius = (coadd_dim/2. - buff)*pixel_scale/60  # unit: arcmin
            area = np.pi*radius**2
            del radius
        else:
            raise ValueError("layout can only be 'random' or 'random_disk' \
                    for wldeblend")

        nobj_mean = area * density_mean
        nobj = rng.poisson(nobj_mean)
        self.density = nobj/area

        self.shifts_array = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            pixel_scale=pixel_scale,
            layout=layout,
            nobj=nobj,
        )

        num = len(self)
        self.indices = self.rng.randint(
            0,
            self._star_cat.size,
            size=num,
        )

    def __len__(self):
        return self.shifts_array.size

    def get_objlist(self, *, survey, noise):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey or BasicSurvey
            The survey object
        noise: float
            The noise level, needed for setting gsparams

        Returns
        -------
        [galsim objects], [shifts],
        [bright_objlist], [bright_shifts], [bright_mags]
        """

        sarray = self.shifts_array

        band = survey.filter_band
        objlist = []
        shifts = []

        bright_objlist = []
        bright_shifts = []
        bright_mags = []

        for i in range(len(self)):
            pos = galsim.PositionD(sarray['dx'][i], sarray['dy'][i])
            star, mag, isbright = self._get_star(survey, band, i, noise)
            if isbright:
                bright_objlist.append(star)
                bright_shifts.append(pos)
                bright_mags.append(mag)
            else:
                objlist.append(star)
                shifts.append(pos)

        return objlist, shifts, bright_objlist, bright_shifts, bright_mags

    def _get_star(self, survey, band, i, noise):
        """
        Parameters
        ----------
        survey: WLDeblendSurvey or BasicSurvey
            The survey object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object
        noise: float
            The noise level, needed for setting gsparams

        Returns
        -------
        galsim.GSObject
        """

        index = self.indices[i]

        mag = get_star_mag(stars=self._star_cat, index=index, band=band)
        flux = survey.get_flux(mag)

        gsparams, isbright = get_star_gsparams(mag, flux, noise)
        star = galsim.Gaussian(
            fwhm=1.0e-4,
            flux=flux,
            gsparams=gsparams,
        )

        return star, mag, isbright


def get_star_gsparams(mag, flux, noise):
    """
    Get appropriate gsparams given flux and noise

    Parameters
    ----------
    mag: float
        mag of star
    flux: float
        flux of star
    noise: float
        noise of image

    Returns
    --------
    GSParams, isbright where isbright is true for stars with mag less than 18
    """
    do_thresh = do_acc = False
    if mag < 18:
        do_thresh = True
    if mag < 15:
        do_acc = True

    if do_thresh or do_acc:
        isbright = True

        kw = {}
        if do_thresh:

            # this is designed to quantize the folding_threshold values,
            # so that there are fewer objects in the GalSim C++ cache.
            # With continuous values of folding_threshold, there would be
            # a moderately largish overhead for each object.

            folding_threshold = noise/flux
            folding_threshold = np.exp(
                np.floor(np.log(folding_threshold))
            )
            kw['folding_threshold'] = min(folding_threshold, 0.005)

        if do_acc:
            kw['kvalue_accuracy'] = 1.0e-8
            kw['maxk_threshold'] = 1.0e-5

        gsparams = galsim.GSParams(**kw)
    else:
        gsparams = None
        isbright = False

    return gsparams, isbright


def get_star_mag(*, stars, index, band):
    magname = '%s_ab' % band
    return stars[magname][index]


def sample_star_density(*, rng, min_density, max_density):
    """
    sample from the set of example densities
    """
    densities = load_sample_star_densities(
        min_density=min_density,
        max_density=max_density,
    )
    ind = rng.choice(densities.size)

    return densities[ind]


@functools.lru_cache(maxsize=1)
def load_sample_star_densities(*, min_density, max_density):
    assert 'CATSIM_DIR' in os.environ
    fname = os.path.join(
        os.environ['CATSIM_DIR'],
        'stellar_density_lsst.fits.gz',
    )
    with fitsio.FITS(fname) as fits:
        densities = fits[1]['I'].read().ravel()

    w, = np.where(
        (densities > min_density) &
        (densities < max_density)
    )
    return densities[w]


def load_sample_stars():
    assert 'CATSIM_DIR' in os.environ
    fname = os.path.join(
        os.environ['CATSIM_DIR'],
        'stars_med_june2018.fits',
    )
    return cached_catalog_read(fname)
