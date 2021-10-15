import pytest
import numpy as np
from ..galaxies import make_galaxy_catalog
from ..psfs import make_fixed_psf

from ..sim import make_sim


@pytest.mark.parametrize('morph', ('exp', 'dev', 'bd', 'bdk'))
def test_pairs_smoke(morph):
    """
    test sim can run
    """
    seed = 74321

    for trial in (1, 2):
        rng = np.random.RandomState(seed)

        sep = 4.0  # arcseconds

        coadd_dim = 100
        buff = 10
        bands = ['i']

        gal_config = {
            'hlr': 1.0,
            'mag': 22,
            'morph': morph,
        }

        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            gal_type='fixed',
            gal_config=gal_config,
            layout='random',
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
        )

        if trial == 1:
            image = sim_data['band_data']['i'][0].image.array
        else:
            new_image = sim_data['band_data']['i'][0].image.array

            assert np.all(image == new_image)
