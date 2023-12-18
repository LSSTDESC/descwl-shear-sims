#!/usr/bin/env python
"""
This is the unit test to make sure that the simulated images are centerred at
(dim-1)/2.
"""
import pytest
import galsim
import numpy as np
from ..sim import make_sim
from ..galaxies import make_galaxy_catalog
from ..psfs import make_fixed_psf  # for making a power spectrum PSF
from ..shear import ShearConstant

shear_obj = ShearConstant(g1=0.02, g2=0.)


@pytest.mark.parametrize("ran_seed", [0, 1, 2])
def test_sim_center(ran_seed):
    """Tests to make sure the center of a simulated galaxy is at image center
    if offset is zero.
    """
    rng = np.random.RandomState(ran_seed)
    args = {
        "rotate": False,
        "dither": False,
        "cosmic_rays": False,
        "bad_columns": False,
        "star_bleeds": False,
        "star_catalog": None,
    }
    band_list = ["i"]
    # make simulation that has zero offset since buff = coadd_dim / 2
    coadd_dim = 60
    buff = 30
    psf = make_fixed_psf(psf_type="moffat")

    # galaxy catalog; you can make your own
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="random_disk",
    )

    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
        bands=band_list,
        noise_factor=0.0,
        theta0=0.0,
        **args
    )
    img_array = sim_data["band_data"][band_list[0]][0].getImage().getArray()
    out_dim = img_array.shape[0]
    assert out_dim == coadd_dim + 10, (
        "The length of the output image should \
            be %d"
        % out_dim
    )
    img = galsim.ImageF(img_array)
    moments = galsim.hsm.FindAdaptiveMom(img)
    offset = galsim.BoundsD(moments.image_bounds).center - moments.moments_centroid
    np.testing.assert_almost_equal(offset.x, 0.0, 5)
    np.testing.assert_almost_equal(offset.y, 0.0, 5)

    return


if __name__ == "__main__":
    test_sim_center(0)
