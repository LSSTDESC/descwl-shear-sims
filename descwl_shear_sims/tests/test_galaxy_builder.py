import os
import numpy as np
import galsim
import pytest

from ..galaxy_builder import RoundGalaxyBuilder
from ..cache_tools import cached_catalog_read


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_galaxy_builder_smoke():
    import descwl

    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )

    seed = 55
    rng = np.random.RandomState(seed)

    cat = cached_catalog_read(fname)
    band = 'r'
    scale = 0.20

    pars = descwl.survey.Survey.get_defaults(
        survey_name='LSST',
        filter_band=band,
    )
    pars['survey_name'] = 'LSST'
    pars['filter_band'] = band
    pars['pixel_scale'] = scale

    survey = descwl.survey.Survey(**pars)

    builder = RoundGalaxyBuilder(
        survey=survey,
        no_disk=False,
        no_bulge=False,
        no_agn=False,
        verbose_model=False,
    )

    i = rng.randint(0, cat.size)
    builder.from_catalog(
        cat[i], 0, 0,
        survey.filter_band,
    )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_galaxy_builder():
    import descwl

    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )

    seed = 55
    rng = np.random.RandomState(seed)

    cat = cached_catalog_read(fname)
    band = 'r'
    pars = descwl.survey.Survey.get_defaults(
        survey_name='LSST',
        filter_band=band,
    )

    survey = descwl.survey.Survey(**pars)

    builder = RoundGalaxyBuilder(
        survey=survey,
        no_disk=False,
        no_bulge=False,
        no_agn=False,
        verbose_model=False,
    )

    psf = galsim.Gaussian(fwhm=0.7)

    i = rng.randint(0, cat.size)
    obj0 = builder.from_catalog(
        cat[i], 0, 0,
        survey.filter_band,
    ).model
    obj = galsim.Convolve(obj0, psf)

    im = obj.drawImage(
        nx=53, ny=53, scale=0.2,
    ).array

    cen = (np.array(im.shape)-1)/2

    rows, cols = np.mgrid[
        0:im.shape[0],
        0:im.shape[1],
    ]

    rows = rows - cen[0]
    cols = cols - cen[1]

    imsum = im.sum()
    irr = (im*rows**2).sum()/imsum
    irc = (im*rows*cols).sum()/imsum
    icc = (im*cols**2).sum()/imsum

    e1 = (irr - icc)/(irr + icc)
    e2 = 2*irc/(irr + icc)

    assert abs(e1) < 0.001
    assert abs(e2) < 0.001
