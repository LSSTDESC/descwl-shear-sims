import numpy as np
import galsim
import lsst.afw.image as afw_image
import lsst.geom as geom
from ..wcs import make_dm_wcs, make_coadd_dm_wcs_simple
from ._wcs import make_sim_wcs, SCALE


def test_dmwcs():
    dim = 20

    masked_image = afw_image.MaskedImageF(dim, dim)
    exp = afw_image.ExposureF(masked_image)

    galsim_wcs = make_sim_wcs(dim)
    tdm_wcs = make_dm_wcs(galsim_wcs)

    exp.setWcs(tdm_wcs)

    dm_wcs = exp.getWcs()

    dm_cd = dm_wcs.getCdMatrix()
    print("dm wcs cd matrix", dm_cd)
    print("galsim wcs cd matrix", galsim_wcs.cd)
    assert np.all(dm_cd == galsim_wcs.cd)

    x = 8.5
    y = 10.1

    # galsim uses 1 offset
    gs_pos = galsim.PositionD(x=x, y=y)
    # DM uses zero offest
    pos = geom.Point2D(x=x - 1, y=y - 1)

    skypos = dm_wcs.pixelToSky(pos)

    print("type of dm skypos", type(skypos))
    print("dm skypos", skypos)
    print("dm skypos ra and dec", skypos.getRa(), skypos.getDec())
    print("type of dm skypos ra", type(skypos.getRa()))

    gs_skypos = galsim_wcs.toWorld(gs_pos)

    print("type of galsim skypos", type(gs_skypos))
    print("galsim skypos", gs_skypos)
    print("galsim skypos ra and dec", gs_skypos.ra, gs_skypos.dec)
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


def test_coadd_dmwcs_simple():
    coadd_dim = 20

    masked_image = afw_image.MaskedImageF(coadd_dim, coadd_dim)
    exp = afw_image.ExposureF(masked_image)

    galsim_wcs = make_sim_wcs(coadd_dim)
    print("input pixel scale", SCALE)
    tcoadd_dm_wcs_simple, coadd_bbox = make_coadd_dm_wcs_simple(coadd_dim, SCALE)

    exp.setWcs(tcoadd_dm_wcs_simple)

    dm_coadd_wcs_simple = exp.getWcs()

    dm_coadd_simple_cd = dm_coadd_wcs_simple.getCdMatrix()
    print("dm wcs cd matrix", dm_coadd_simple_cd)
    print("galsim wcs cd matrix", galsim_wcs.cd)
    assert np.all(dm_coadd_simple_cd == galsim_wcs.cd)

    x = 8.5
    y = 10.1

    gs_pos = galsim.PositionD(x=x, y=y)

    pos = geom.Point2D(x=x - 1, y=y - 1)
    skypos = dm_coadd_wcs_simple.pixelToSky(pos)

    print("type of dm skypos", type(skypos))
    print("dm skypos", skypos)
    print("dm skypos ra and dec", skypos.getRa(), skypos.getDec())
    print("type of dm skypos ra", type(skypos.getRa()))

    gs_skypos = galsim_wcs.toWorld(gs_pos)

    print("type of galsim skypos", type(gs_skypos))
    print("galsim skypos", gs_skypos)
    print("galsim sky position ra dec", gs_skypos.ra, gs_skypos.dec)

    assert np.allclose(
        skypos.getRa().asRadians(), gs_skypos.ra / galsim.radians,
    )
    assert np.allclose(
        skypos.getDec().asRadians(), gs_skypos.dec / galsim.radians,
    )

    impos = dm_coadd_wcs_simple.skyToPixel(skypos)
    assert np.allclose(impos.x, x - 1)
    assert np.allclose(impos.y, y - 1)
