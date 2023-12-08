import os
import pytest
import numpy as np
import astropy.io.fits as pyfits
import lsst.afw.image as afw_image
import lsst.afw.geom as afw_geom

from descwl_shear_sims.sim import make_sim
from descwl_shear_sims import __test_dir__
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.stars import StarCatalog, make_star_catalog
from descwl_shear_sims.psfs import make_fixed_psf

from descwl_shear_sims.shear import ShearConstant

shear_obj = ShearConstant(g1=0.02, g2=0.)

@pytest.mark.parametrize('gal_type', ['wldeblend', 'fixed'])
def test_sim_consistency(gal_type):
    seed = 7421
    coadd_dim = 201
    buff = 30
    rng = np.random.RandomState(seed)

    # gal_type = "fixed"

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type=gal_type,
        coadd_dim=coadd_dim,
        buff=buff,
        layout="random",
    )

    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=100,
    )

    psf = make_fixed_psf(psf_type="moffat")

    # tests that we actually get saturation are in test_star_masks_and_bleeds
    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
    )

    data = _["band_data"]["i"][0].getMaskedImage().getImage().getArray()
    fname = os.path.join(__test_dir__, "image_%s_232d7f1.fits" % gal_type)
    data_ref = pyfits.getdata(fname)
    assert np.max(np.abs(data-data_ref)) < 1e-9
    return

if __name__ == '__main__':
    test_sim_consistency("wldeblend")
    test_sim_consistency("fixed")
