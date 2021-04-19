import numpy as np
import galsim
import lsst.afw.image as afw_image
import lsst.geom as geom
from ..sim import FixedDMPSF

WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)
SCALE = 0.263


def make_wcs(dim):

    dims = [dim]*2
    cen = (np.array(dims)-1)/2
    image_origin = galsim.PositionD(x=cen[1], y=cen[0])

    mat = np.array(
        [[SCALE, 0.0],
         [0.0, SCALE]],
    )

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=WORLD_ORIGIN,
        units=galsim.arcsec,
    )


def test_fixed_dmpsf_smoke():
    dim = 20
    masked_image = afw_image.MaskedImageF(dim, dim)
    exp = afw_image.ExposureF(masked_image)

    gspsf = galsim.Gaussian(fwhm=0.9)
    psf_dim = 15
    wcs = make_wcs(dim)

    fpsf = FixedDMPSF(gspsf=gspsf, psf_dim=psf_dim, wcs=wcs)
    exp.setPsf(fpsf)

    psf = exp.getPsf()

    x = 8.5
    y = 10.1
    pos = geom.Point2D(x=x, y=y)
    gs_pos = galsim.PositionD(x=x, y=y)

    # this one is always centered
    msim = psf.computeKernelImage(pos)
    assert msim.array.shape == (psf_dim, psf_dim)

    gsim = gspsf.drawImage(
        nx=psf_dim, ny=psf_dim,
        wcs=wcs.local(image_pos=gs_pos),
    )

    assert np.allclose(msim.array, gsim.array)
