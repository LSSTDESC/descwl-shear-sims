"""Testing code for function for rendering simulated sets of galaxies.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
"""

import numpy as np
import galsim

import pytest

from ..render_sim import render_objs_with_psf_shear


def test_render_sim_smoke():
    img_dim = 103
    img_cen = (img_dim - 1)/2
    scale = 0.25
    method = 'auto'
    g1 = 0.0
    g2 = 0.0
    shear_scene = False
    world_origin = galsim.CelestialCoord(
        ra=0 * galsim.degrees,
        dec=0 * galsim.degrees,
    )
    objs = [
        {
            'obj': galsim.Exponential(half_light_radius=5.5),
            'type': 'galaxy',
        },
    ]

    def _psf_function(*, x, y):
        assert np.allclose(x, img_cen)
        assert np.allclose(y, img_cen)
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [galsim.PositionD(x=0.0, y=0.0)]
    wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            scale, 0.0, 0.0, scale,
            origin=galsim.PositionD(x=img_cen, y=img_cen),
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )

    se_img, overlap_info = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    assert len(overlap_info) == len(objs)
    for info in overlap_info:
        assert 'pos' in info and 'overlaps' in info
        assert info['overlaps'] is True

    expected_img = galsim.Convolve(
        objs[0]['obj'], galsim.Gaussian(fwhm=0.9)
    ).drawImage(
        nx=img_dim, ny=img_dim, scale=scale)

    assert np.allclose(expected_img.array, se_img.array, rtol=0, atol=1e-9)


def test_render_sim_star_smoke():
    img_dim = 103
    img_cen = (img_dim - 1)/2
    scale = 0.25
    method = 'auto'
    g1 = 0.0
    g2 = 0.0
    shear_scene = False
    world_origin = galsim.CelestialCoord(
        ra=0 * galsim.degrees,
        dec=0 * galsim.degrees,
    )
    objs = [
        {
            'obj': galsim.Exponential(half_light_radius=5.5),
            'type': 'galaxy',
        },
        {
            'obj': galsim.Gaussian(half_light_radius=1.0e-4),
            'mag': 19,
            'type': 'star',
        },
    ]

    def _psf_function(*, x, y):
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [
        galsim.PositionD(x=0.0, y=0.0),
        galsim.PositionD(x=2.0, y=1.0),
    ]
    wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            scale, 0.0, 0.0, scale,
            origin=galsim.PositionD(x=img_cen, y=img_cen),
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )

    se_img, overlap_info = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    assert len(overlap_info) == len(objs)
    for info in overlap_info:
        assert 'pos' in info and 'overlaps' in info
        assert info['overlaps'] is True


@pytest.mark.parametrize('shear_scene', [True, False])
def test_render_sim_centered_shear_scene(shear_scene):
    img_dim = 103
    img_cen = (img_dim - 1)/2
    scale = 0.25
    method = 'auto'
    g1 = 0.5
    g2 = -0.2
    world_origin = galsim.CelestialCoord(
        ra=10 * galsim.degrees,
        dec=30 * galsim.degrees,
    )

    objs = [
        {
            'obj': galsim.Exponential(half_light_radius=5.5),
            'type': 'galaxy',
        },
    ]

    def _psf_function(*, x, y):
        assert np.allclose(x, img_cen)
        assert np.allclose(y, img_cen)
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [galsim.PositionD(x=0.0, y=0.0)]
    wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            scale, 0.0, 0.0, scale,
            origin=galsim.PositionD(x=img_cen, y=img_cen),
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )

    se_img, overlap_info = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    expected_img = galsim.Convolve(
        objs[0]['obj'].shear(g1=g1, g2=g2), galsim.Gaussian(fwhm=0.9)
    ).drawImage(
        nx=img_dim, ny=img_dim, wcs=wcs.local(world_pos=world_origin))

    assert np.allclose(expected_img.array, se_img.array, rtol=0, atol=1e-9)


@pytest.mark.parametrize('shear_scene', [True, False])
def test_render_sim_shear_scene(shear_scene):
    img_dim = 103
    img_cen = (img_dim - 1)/2
    origin = galsim.PositionD(x=img_cen, y=img_cen)
    scale = 0.25
    method = 'auto'
    g1 = 0.5
    g2 = -0.2
    world_origin = galsim.CelestialCoord(
        ra=10 * galsim.degrees, dec=30 * galsim.degrees)
    objs = [
        {
            'obj': galsim.Exponential(half_light_radius=5.5),
            'type': 'galaxy',
        }
    ]

    def _psf_function(*, x, y):
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [galsim.PositionD(x=-1.3, y=0.578)]
    wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            scale, 0.0, 0.0, scale,
            origin=origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )
    jac_wcs = wcs.jacobian(world_pos=wcs.center)

    se_img, _ = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=not shear_scene)

    se_img_shear_scene, _ = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    if shear_scene:
        smat = galsim.Shear(g1=g1, g2=g2).getMatrix()
        dxdy = np.dot(smat, np.array([uv_offsets[0].x, uv_offsets[0].y]))
        offset = jac_wcs.toImage(galsim.PositionD(x=dxdy[0], y=dxdy[1]))
    else:
        offset = jac_wcs.toImage(uv_offsets[0])

    obj_world_pos = wcs.toWorld(offset + origin)
    expected_img = galsim.Convolve(
        objs[0]['obj'].shear(g1=g1, g2=g2),
        galsim.Gaussian(fwhm=0.9)
    ).drawImage(
        nx=img_dim, ny=img_dim, offset=offset,
        wcs=wcs.local(world_pos=obj_world_pos))

    assert not np.allclose(expected_img.array, se_img.array, rtol=0, atol=1e-9)
    assert np.allclose(expected_img.array, se_img_shear_scene.array, rtol=0, atol=1e-9)
