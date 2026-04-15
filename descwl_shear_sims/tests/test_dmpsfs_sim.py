import numpy as np
import galsim

from ..psfs import PowerSpectrumPSF
from ..sim import make_sim
from ..galaxies import make_galaxy_catalog
from ..shear import ShearConstant


class TestPSF:
    """
    Something that meets the interface getPSF
    """
    def getPSF(self, psf):
        return galsim.Gaussian(fwhm=1)


def test_dmpsf_smoke():
    rng = np.random.RandomState(seed=10)

    shear_obj = ShearConstant(g1=0.02, g2=0.)
    psf = TestPSF()

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=300,
        buff=30,
        layout="grid",
    )

    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=300,
        psf_dim=51,
        bands=['r'],
        shear_obj=shear_obj,
        psf=psf,
    )


def test_dmpsf_ps_smoke():
    rng = np.random.RandomState(seed=10)

    shear_obj = ShearConstant(g1=0.02, g2=0.)
    psf = PowerSpectrumPSF(
        rng=rng,
        im_width=120,
        buff=20,
        scale=0.2,
        trunc=10,
    )

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=300,
        buff=30,
        layout="grid",
    )

    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=300,
        psf_dim=51,
        bands=['r'],
        shear_obj=shear_obj,
        psf=psf,
    )
