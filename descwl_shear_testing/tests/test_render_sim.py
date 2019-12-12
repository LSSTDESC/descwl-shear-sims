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

    objs = [galsim.Exponential(half_light_radius=0.5)]

    def _psf_function(*, x, y):
        assert np.allclose(x, img_cen)
        assert np.allclose(y, img_cen)
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [galsim.PositionD(x=0.0, y=0.0)]
    uv_cen = galsim.PositionD(x=img_cen * scale, y=img_cen * scale)
    wcs = galsim.PixelScale(scale)

    se_img = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        uv_cen=uv_cen, wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    expected_img = galsim.Convolve(
        objs[0], galsim.Gaussian(fwhm=0.9)
    ).drawImage(
        nx=img_dim, ny=img_dim, scale=scale)

    assert np.allclose(expected_img.array, se_img.array, rtol=0, atol=1e-6)


@pytest.mark.parametrize('shear_scene', [True, False])
def test_render_sim_centered_shear_scene(shear_scene):
    img_dim = 103
    img_cen = (img_dim - 1)/2
    scale = 0.25
    method = 'auto'
    g1 = 0.5
    g2 = -0.2

    objs = [galsim.Exponential(half_light_radius=5.5)]

    def _psf_function(*, x, y):
        assert np.allclose(x, img_cen)
        assert np.allclose(y, img_cen)
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [galsim.PositionD(x=0.0, y=0.0)]
    uv_cen = galsim.PositionD(x=img_cen * scale, y=img_cen * scale)
    wcs = galsim.PixelScale(scale)

    se_img = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        uv_cen=uv_cen, wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    expected_img = galsim.Convolve(
        objs[0].shear(g1=g1, g2=g2), galsim.Gaussian(fwhm=0.9)
    ).drawImage(
        nx=img_dim, ny=img_dim, wcs=wcs.local(world_pos=uv_cen))

    assert np.allclose(expected_img.array, se_img.array, rtol=0, atol=1e-9)


@pytest.mark.parametrize('shear_scene', [True, False])
def test_render_sim_shear_scene(shear_scene):
    img_dim = 103
    img_cen = (img_dim - 1)/2
    scale = 0.25
    method = 'auto'
    g1 = 0.5
    g2 = -0.2

    objs = [galsim.Exponential(half_light_radius=5.5)]

    def _psf_function(*, x, y):
        return galsim.Gaussian(fwhm=0.9)

    uv_offsets = [galsim.PositionD(x=-1.3, y=0.578)]
    uv_cen = galsim.PositionD(x=img_cen * scale, y=img_cen * scale)
    wcs = galsim.PixelScale(scale)

    se_img = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        uv_cen=uv_cen, wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=not shear_scene)

    se_img_shear_scene = render_objs_with_psf_shear(
        objs=objs, psf_function=_psf_function, uv_offsets=uv_offsets,
        uv_cen=uv_cen, wcs=wcs, img_dim=img_dim, method=method,
        g1=g1, g2=g2, shear_scene=shear_scene)

    if shear_scene:
        expected_img = galsim.Convolve(
            objs[0].shift(uv_offsets[0]).shear(g1=g1, g2=g2),
            galsim.Gaussian(fwhm=0.9)
        ).drawImage(
            nx=img_dim, ny=img_dim, wcs=wcs.local(world_pos=uv_cen))
    else:
        expected_img = galsim.Convolve(
            objs[0].shear(g1=g1, g2=g2).shift(uv_offsets[0]),
            galsim.Gaussian(fwhm=0.9)
        ).drawImage(
            nx=img_dim, ny=img_dim, wcs=wcs.local(world_pos=uv_cen))

    assert not np.allclose(expected_img.array, se_img.array, rtol=0, atol=1e-9)
    assert np.allclose(
        expected_img.array, se_img_shear_scene.array, rtol=0, atol=1e-9)
