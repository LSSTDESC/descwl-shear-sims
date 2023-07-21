import os
import pytest
import numpy as np
import lsst.afw.image as afw_image
import lsst.afw.geom as afw_geom

from descwl_shear_sims.galaxies import make_galaxy_catalog, DEFAULT_FIXED_GAL_CONFIG
from descwl_shear_sims.stars import StarCatalog, make_star_catalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf

from descwl_shear_sims.sim import make_sim, get_se_dim

def test_sim_se_dim():
    """
    test sim can run
    """
    seed = 74321
    rng = np.random.RandomState(seed)

    coadd_dim = 351
    se_dim = 351
    psf_dim = 51
    bands = ["i"]
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=30,
        layout="grid",
    )

    psf = make_fixed_psf(psf_type="gauss")
    data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        psf_dim=psf_dim,
        bands=bands,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )
    return data['band_data']['i'][0]

a = test_sim_se_dim()
data = a.getImage().getArray()
