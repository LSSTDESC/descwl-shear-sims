import os
from collections import OrderedDict
import galsim
from copy import deepcopy

from .cache_tools import cached_catalog_read

FIXED_STAR_MAG = 19.0

# From the LSST science book
# mag to saturate for 30 second exposures, need this for the
# longer exposures, so for now just add one TODO
MAG_SAT = {
    'u': 14.7+1,
    'g': 15.7+1,
    'r': 15.8+1,
    'i': 15.8+1,
    'z': 15.3+1,
}


def sample_fixed_star(*,
                      rng,
                      bands,
                      sat_stars,
                      sat_stars_frac,
                      star_mask_pdf):
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

    flux = 10**(0.4 * (30 - FIXED_STAR_MAG))

    star = OrderedDict()
    for band in bands:
        obj = galsim.Gaussian(fwhm=1.0e-4).withFlux(flux)

        star[band] = {
            'type': 'star',
            'obj': obj,
            'type': 'star',
            'saturated': saturated,
            'sat_data': deepcopy(sat_data),
        }

    return star


def sample_star(*,
                rng,
                star_data,
                surveys,
                bands,
                sat_stars,
                star_mask_pdf=None):

    # same star index for all bands
    star_ind = rng.choice(star_data.size)

    star = OrderedDict()

    for band in bands:
        bstar = {'type': 'star'}
        bstar['mag'] = get_star_mag(stars=star_data, index=star_ind, band=band)
        bstar['flux'] = surveys[band].get_flux(bstar['mag'])

        bstar['obj'] = galsim.Gaussian(
            fwhm=1.0e-4,
        ).withFlux(
            flux=bstar['flux'],
        )

        # we might reset this below
        bstar['saturated'] = False
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


def get_star_mag(*, stars, index, band):
    magname = '%s_ab' % band
    return stars[magname][index]


def is_saturated(*, mag, band):
    if mag < MAG_SAT[band]:
        return True
    else:
        return False


def load_sample_stars():
    assert 'CATSIM_DIR' in os.environ
    fname = os.path.join(
        os.environ['CATSIM_DIR'],
        'stars_med_june2018.fits',
    )
    return cached_catalog_read(fname)
