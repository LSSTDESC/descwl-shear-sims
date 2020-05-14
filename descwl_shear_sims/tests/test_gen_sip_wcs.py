import os
import pytest
import numpy as np
import galsim

from ..gen_sip_wcs import gen_sip_wcs


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_gen_sip_wcs_smoke():
    seed = 42
    world_origin = galsim.CelestialCoord(
        ra=20 * galsim.degrees,
        dec=-10 * galsim.degrees)
    origin = galsim.PositionD(x=10, y=11)
    gen_sip_wcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=world_origin,
        origin=origin)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_gen_sip_wcs_dither():
    seed = 42
    world_origin = galsim.CelestialCoord(
        ra=20 * galsim.degrees,
        dec=-10 * galsim.degrees)
    origin = galsim.PositionD(x=10, y=11)
    wcs = gen_sip_wcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=world_origin,
        origin=origin)

    rng = np.random.RandomState(seed=seed)
    # we have to call the RNG in the right order
    rng.normal() * 0.1  # g1
    rng.normal() * 0.1  # g2
    scale = (1.0 + rng.normal() * 0.1) * 0.25
    rng.uniform(low=0, high=360)  # theta
    du = rng.uniform(-0.5, 0.5) * scale
    dv = rng.uniform(-0.5, 0.5) * scale

    jac = wcs.jacobian(world_pos=wcs.center).getMatrix()
    dxdy = np.dot(np.linalg.inv(jac), np.array([du, dv]))

    assert np.allclose(wcs.crpix[0], origin.x + dxdy[0])
    assert np.allclose(wcs.crpix[1], origin.y + dxdy[1])


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_gen_sip_wcs_jacobian():
    seed0 = 81
    rng0 = np.random.RandomState(seed=seed0)
    for i in range(100):

        seed = rng0.randint(0, 2**30)

        world_origin = galsim.CelestialCoord(
            ra=20 * galsim.degrees,
            dec=-10 * galsim.degrees)
        origin = galsim.PositionD(x=10, y=11)
        wcs = gen_sip_wcs(
            rng=np.random.RandomState(seed=seed),
            position_angle_range=(0, 360),
            dither_range=(-0.5, 0.5),
            scale=0.25,
            scale_frac_std=0.1,
            shear_std=0.1,
            world_origin=world_origin,
            origin=origin,
        )

        rng = np.random.RandomState(seed=seed)
        g1 = rng.normal() * 0.1
        g2 = rng.normal() * 0.1
        scale = (1.0 + rng.normal() * 0.1) * 0.25
        theta = rng.uniform(low=0, high=360)

        # at the center of the tangent plane, the local WCS should match the
        # input affine WCS
        jac_wcs = wcs.jacobian(world_pos=wcs.center)
        _scale, _shear, _theta, _ = jac_wcs.getDecomposition()

        this_theta = _theta.deg
        if this_theta < 0.0:
            this_theta += 360
        assert np.allclose(_shear.g1, g1)
        assert np.allclose(_shear.g2, g2)
        assert np.allclose(_scale, scale)
        assert np.allclose(this_theta, theta)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
def test_gen_sip_wcs_seed():
    seed = 42
    wcs1 = gen_sip_wcs(
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

    wcs2 = gen_sip_wcs(
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
