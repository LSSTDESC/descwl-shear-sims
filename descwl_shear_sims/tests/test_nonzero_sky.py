import numpy as np
from ..galaxies import make_galaxy_catalog
from ..psfs import make_fixed_psf

from ..sim import make_sim


def test_nonzero_sky():
    """
    make an image with very faint galaxies so we just have
    the sky and noise
    """
    seed = 8811

    rng = np.random.RandomState(seed)
    sky_n_sigma = -3  # sky over-subtracted by 3 sigma

    coadd_dim = 351
    psf_dim = 51
    bands = ['r', 'i', 'z']
    ntrial = 3

    for i in range(ntrial):
        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type='fixed',
            coadd_dim=coadd_dim,
            buff=0,
            layout='random',
            gal_config={'mag': 37.0, 'hlr': 0.5},
        )

        psf = make_fixed_psf(psf_type="gauss")
        data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            psf_dim=psf_dim,
            bands=bands,
            g1=0.02,
            g2=0.00,
            psf=psf,
            sky_n_sigma=sky_n_sigma,
        )

        for band in bands:
            calexp = data['band_data'][band][0]
            image = calexp.image.array
            var = calexp.variance.array

            # from matplotlib import pyplot as plt
            # plt.imshow(image)
            # plt.show()

            mn = image.mean()
            err = image.std() / np.sqrt(image.size)

            sigma = np.sqrt(var[10, 10])

            expected_mean = sky_n_sigma * sigma

            assert abs(mn - expected_mean) < 4 * err
