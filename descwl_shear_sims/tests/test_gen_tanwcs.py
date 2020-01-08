import numpy as np
import galsim

from ..gen_tanwcs import gen_tanwcs


def test_gen_tanwcs_smoke():
    seed = 42
    world_origin = galsim.CelestialCoord(
        ra=20 * galsim.degrees,
        dec=-10 * galsim.degrees)
    origin = galsim.PositionD(x=10, y=11)
    gen_tanwcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=world_origin,
        origin=origin)


def test_gen_tanwcs_jacobian():
    seed = 42
    world_origin = galsim.CelestialCoord(
        ra=20 * galsim.degrees,
        dec=-10 * galsim.degrees)
    origin = galsim.PositionD(x=10, y=11)
    wcs = gen_tanwcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=world_origin,
        origin=origin)

    rng = np.random.RandomState(seed=seed)
    g1 = rng.normal() * 0.1
    g2 = rng.normal() * 0.1
    scale = (1.0 + rng.normal() * 0.1) * 0.25
    theta = rng.uniform(low=0, high=360)

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
        origin=galsim.PositionD(x=10, y=11))

    assert wcs1 == wcs2
    assert str(wcs1) == str(wcs2)
