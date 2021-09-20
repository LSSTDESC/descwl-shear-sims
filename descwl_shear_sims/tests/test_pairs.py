import pytest
import numpy as np
from ..galaxies import make_galaxy_catalog, DEFAULT_FIXED_GAL_CONFIG
from ..psfs import make_fixed_psf

from ..sim import make_sim
from ..constants import ZERO_POINT


@pytest.mark.parametrize('dither,rotate', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_pairs_smoke(dither, rotate):
    """
    test sim can run
    """
    seed = 74321
    rng = np.random.RandomState(seed)

    sep = 4.0  # arcseconds

    coadd_dim = 100
    bands = ['i']

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type='exp',
        layout='pair',
        sep=sep,
    )

    psf = make_fixed_psf(psf_type='gauss')
    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        bands=bands,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=dither,
        rotate=rotate,
    )

    image = sim_data['band_data']['i'][0].image.array
    imsum = image.sum()

    avg_flux = imsum / 2
    mag = ZERO_POINT - 2.5*np.log10(avg_flux)
    assert abs(mag - DEFAULT_FIXED_GAL_CONFIG['mag']) < 0.005
