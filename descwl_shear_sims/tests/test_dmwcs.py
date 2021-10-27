import numpy as np
import galsim
import lsst.afw.image as afw_image
import lsst.geom as geom
from ..wcs import make_dm_wcs
from ._wcs import make_sim_wcs


def test_dmwcs():
    dim = 20

    masked_image = afw_image.MaskedImageF(dim, dim)
    exp = afw_image.ExposureF(masked_image)

    galsim_wcs = make_sim_wcs(dim)
    tdm_wcs = make_dm_wcs(galsim_wcs)

    exp.setWcs(tdm_wcs)

    dm_wcs = exp.getWcs()

    dm_cd = dm_wcs.getCdMatrix()
    assert np.all(dm_cd == galsim_wcs.cd)

    x = 8.5
    y = 10.1

    # galsim uses 1 offset
    gs_pos = galsim.PositionD(x=x, y=y)
    # DM uses zero offest
    pos = geom.Point2D(x=x - 1, y=y - 1)

    skypos = dm_wcs.pixelToSky(pos)

    print(type(skypos))
    print(skypos)
    print(skypos.getRa(), skypos.getDec())
    print(type(skypos.getRa()))

    print()

    gs_skypos = galsim_wcs.toWorld(gs_pos)

    print(type(gs_skypos))
    print(gs_skypos)
    print(gs_skypos.ra, gs_skypos.dec)
    # stop

    assert np.allclose(
        skypos.getRa().asRadians(), gs_skypos.ra / galsim.radians,
    )
    assert np.allclose(
        skypos.getDec().asRadians(), gs_skypos.dec / galsim.radians,
    )

    impos = dm_wcs.skyToPixel(skypos)
    assert np.allclose(impos.x, x - 1)
    assert np.allclose(impos.y, y - 1)
