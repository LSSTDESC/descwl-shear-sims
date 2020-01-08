import numpy as np
import galsim

from ..gen_tanwcs import gen_tanwcs


def test_gen_tanwcs():
    seed = 42
    wcs = gen_tanwcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=galsim.CelestialCoord(
            ra=20 * galsim.degrees,
            dec=-10 * galsim.degrees),
        uv_origin=galsim.PositionD(x=20, y=21),
        origin=galsim.PositionD(x=10, y=11))

    rng = np.random.RandomState(seed=seed)
    g1 = rng.normal() * 0.1
    g2 = rng.normal() * 0.1
    scale = (1.0 + rng.normal() * 0.1) * 0.25
    theta = rng.uniform(low=0, high=360)
    dither_u = rng.uniform(
        low=-0.5,
        high=0.5) * scale
    dither_v = rng.uniform(
        low=-0.5,
        high=0.5) * scale

    # check that crpix has the right dither

    # CD matrix is in degrees per pixel so convert uv or dither_uv to degrees
    inv_jac_mat = np.linalg.inv(wcs.cd)
    fac = galsim.arcsec / galsim.degrees

    # first get offset put in by the uv_origin
    # this produces the offset in pixels in the image plane of the fiducial
    # uv origin
    # this offset is subtracted from the input origin to recenter the
    # uv plane at (0,0)
    dxdy_uv = np.dot(inv_jac_mat, np.array([20 * fac, 21 * fac]))

    # here we compute the offset in pixels of (u, v) dither that was translated
    # to image coords and applied
    # this translation was done before TanWCS set u to -u so undo that too
    # the minus sign is because FITS has u -> -u so we undo it
    dxdy = np.dot(inv_jac_mat, np.array([-dither_u * fac, dither_v * fac]))

    # now recompute crpix and make sure it is right
    # we have to subtract the dither we applied and put back the uv offset
    assert np.allclose(wcs.crpix[0], 10 - dxdy_uv[0] + dxdy[0])
    assert np.allclose(wcs.crpix[1], 11 - dxdy_uv[1] + dxdy[1])
    assert np.allclose(wcs.x0, 0)
    assert np.allclose(wcs.y0, 0)

    # at the center of the tangent plane, the local WCS should match the
    # input affine WCS
    jac_wcs = wcs.jacobian(world_pos=wcs.center)
    _scale, _shear, _theta, _ = jac_wcs.getDecomposition()

    assert np.allclose(_shear.g1, g1)
    assert np.allclose(_shear.g2, g2)
    assert np.allclose(_scale, scale)
    assert np.allclose(_theta / galsim.degrees, theta)


def test_gen_tanwcs_seed():
    seed = 42
    wcs1 = gen_tanwcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=galsim.CelestialCoord(
            ra=20 * galsim.degrees,
            dec=-10 * galsim.degrees),
        uv_origin=galsim.PositionD(x=20, y=21),
        origin=galsim.PositionD(x=10, y=11))

    wcs2 = gen_tanwcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=galsim.CelestialCoord(
            ra=20 * galsim.degrees,
            dec=-10 * galsim.degrees),
        uv_origin=galsim.PositionD(x=20, y=21),
        origin=galsim.PositionD(x=10, y=11))

    assert wcs1 == wcs2
    assert str(wcs1) == str(wcs2)
