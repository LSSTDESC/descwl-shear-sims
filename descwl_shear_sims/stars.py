import os
import functools
import numpy as np
import fitsio
from collections import OrderedDict
import galsim
from copy import deepcopy

from .cache_tools import cached_catalog_read
from .saturation import BAND_STAR_MAG_SAT


def sample_fixed_star(*,
                      rng,
                      mag,
                      bands,
                      flux_funcs):
    """
    Returns
    -------
    An OrderedDict keyed on band with data of theform
    {'obj': Gaussian, 'type': 'star'}
    """

    star = OrderedDict()
    for band in bands:

        flux = flux_funcs[band](mag)
        obj = galsim.Gaussian(fwhm=1.0e-4).withFlux(flux)

        star[band] = {
            'type': 'star',
            'obj': obj,
            'type': 'star',
            'mag': mag,
            'is_bright': False,
        }

    return star


def sample_star(*,
                rng,
                star_data,
                flux_funcs,
                bands):
    """
    sample a star from the input example star data
    """
    # same star index for all bands
    star_ind = rng.choice(star_data.size)

    star = OrderedDict()

    for band in bands:
        bstar = {
            'type': 'star',
            'is_bright': False,
        }
        bstar['mag'] = get_star_mag(stars=star_data, index=star_ind, band=band)
        bstar['flux'] = flux_funcs[band](bstar['mag'])

        bstar['obj'] = galsim.Gaussian(
            fwhm=1.0e-4,
        ).withFlux(
            flux=bstar['flux'],
        )

        star[band] = bstar

    return star


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
