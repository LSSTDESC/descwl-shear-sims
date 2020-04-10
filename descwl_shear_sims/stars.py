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
                      sat_stars,
                      sat_stars_frac,
                      star_mask_pdf,
                      flux_funcs):
    """
    Returns
    -------
    An OrderedDict keyed on band with data of theform
    {'obj': Gaussian, 'type': 'star'}
    """

    saturated = False
    sat_data = None
    if sat_stars:
        if rng.uniform() < sat_stars_frac:
            saturated = True
            sat_data = star_mask_pdf.sample()

    star = OrderedDict()
    for band in bands:

        flux = flux_funcs[band](mag)
        obj = galsim.Gaussian(fwhm=1.0e-4).withFlux(flux)

        star[band] = {
            'type': 'star',
            'obj': obj,
            'type': 'star',
            'mag': mag,
            'saturated': saturated,
            'sat_data': deepcopy(sat_data),
        }

    return star


def sample_star(*,
                rng,
                star_data,
                flux_funcs,
                bands,
                sat_stars,
                star_mask_pdf=None):

    # same star index for all bands
    star_ind = rng.choice(star_data.size)

    star = OrderedDict()

    for band in bands:
        bstar = {'type': 'star'}
        bstar['mag'] = get_star_mag(stars=star_data, index=star_ind, band=band)
        bstar['flux'] = flux_funcs[band](bstar['mag'])

        bstar['obj'] = galsim.Gaussian(
            fwhm=1.0e-4,
        ).withFlux(
            flux=bstar['flux'],
        )

        bstar['saturated'] = is_saturated(mag=bstar['mag'], band=band)
        star[band] = bstar

    if sat_stars:
        set_sat_data(star=star, star_mask_pdf=star_mask_pdf)

    return star


def set_sat_data(*, star, star_mask_pdf):
    sat_data = star_mask_pdf.sample()

    for band, bstar in star.items():
        bstar['saturated'] = is_saturated(mag=bstar['mag'], band=band)
        if bstar['saturated']:
            bstar['sat_data'] = deepcopy(sat_data)


def any_saturated(*, star):
    return any(star[band]['saturated'] for band in star)


def get_star_mag(*, stars, index, band):
    magname = '%s_ab' % band
    return stars[magname][index]


def is_saturated(*, mag, band):
    if mag < BAND_STAR_MAG_SAT[band]:
        return True
    else:
        return False


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
