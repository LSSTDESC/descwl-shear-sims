import pytest
import numpy as np
from ..galaxies import (
    WLDeblendGalaxyCatalog,
    make_galaxy_catalog,
    FixedGalaxyCatalog,
    GalaxyCatalog,
    FixedPairGalaxyCatalog,
    PairGalaxyCatalog,
)
from ..psfs import make_fixed_psf
from ..sim import make_sim, get_coadd_center_gs_pos
from ..shear import ShearConstant

import galsim

shear_obj = ShearConstant(g1=0.02, g2=0.)


@pytest.mark.parametrize('layout', ('pair', 'random', 'hex', 'custom'))
@pytest.mark.parametrize('gal_type', ('fixed', 'varying', 'custom'))
@pytest.mark.parametrize('morph', ('exp', 'dev', 'bd', 'bdk'))
def test_galaxies_smoke(layout, gal_type, morph):
    """
    test sim can run and is repeatable.  This is relevant as we now support
    varying galaxies
    """

    if gal_type == 'custom' and layout != 'custom':
        pytest.skip("gal_type='custom' requires layout='custom'")
    if layout == 'custom' and gal_type != 'custom':
        pytest.skip("layout='custom' requires gal_type='custom'")

    seed = 74321

    for trial in (1, 2):
        rng = np.random.RandomState(seed)

        sep = 4.0  # arcseconds

        coadd_dim = 100
        buff = 10
        bands = ['i']

        gal_config = {
            'hlr': 1.0,
            'mag': 22,
            'morph': morph,
        }

        kwargs = dict(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            gal_type=gal_type,
            layout=layout,
            sep=sep,
        )

        # For the custom path, supply explicit gal_list + uv_shift
        if gal_type == 'custom':
            gal_list = [
                galsim.Exponential(half_light_radius=0.8, flux=1200.0),
                galsim.DeVaucouleurs(half_light_radius=0.5, flux=800.0),
                galsim.Exponential(half_light_radius=0.6, flux=600.0),
            ]
            uv_shift = [(0.0, 0.0), (8.0, -5.0), (-6.0, 3.0)]  # arcsec
            kwargs.update(gal_list=gal_list, uv_shift=uv_shift)
        else:
            kwargs.update(gal_config=gal_config)

        galaxy_catalog = make_galaxy_catalog(
            **kwargs
        )

        if layout == 'pair':
            if gal_type == 'fixed':
                assert isinstance(galaxy_catalog, FixedPairGalaxyCatalog)
            elif gal_type == 'varying':
                assert isinstance(galaxy_catalog, PairGalaxyCatalog)
        else:
            if gal_type == 'fixed':
                assert isinstance(galaxy_catalog, FixedGalaxyCatalog)
            elif gal_type == 'varying':
                assert isinstance(galaxy_catalog, GalaxyCatalog)

        psf = make_fixed_psf(psf_type='gauss', psf_fwhm=0.1)
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            bands=bands,
            shear_obj=shear_obj,
            psf=psf,
        )

        if trial == 1:
            image = sim_data['band_data']['i'][0].image.array
        else:
            new_image = sim_data['band_data']['i'][0].image.array

            assert np.all(image == new_image)


@pytest.mark.parametrize('layout', ('random', None))
def test_wldeblend_galaxies_smoke(layout):
    """
    test sim can run and is repeatable.  This is relevant as we now support
    varying galaxies
    """
    seed = 912

    for trial in (1, 2):
        rng = np.random.RandomState(seed)

        sep = 4.0  # arcseconds

        coadd_dim = 100
        buff = 10
        bands = ['i']

        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            gal_type='wldeblend',
            layout=layout,
            sep=sep,
        )
        assert isinstance(galaxy_catalog, WLDeblendGalaxyCatalog)

        psf = make_fixed_psf(psf_type='gauss')
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            bands=bands,
            shear_obj=shear_obj,
            psf=psf,
        )

        if trial == 1:
            image = sim_data['band_data']['i'][0].image.array
        else:
            new_image = sim_data['band_data']['i'][0].image.array

            assert np.all(image == new_image)


def test_wlgalaxies_selection():
    seed = 74321
    rng = np.random.RandomState(seed)
    coadd_dim = 100
    buff = 10

    for _ in ["g_ab", "r_ab", "i_ab"]:
        galaxy_catalog = WLDeblendGalaxyCatalog(
            pixel_scale=0.2,
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
            select_observable=_,
            select_upper_limit=27,
            select_lower_limit=25,
        )
        assert np.min(galaxy_catalog._wldeblend_cat[_]) >= 25.0
        assert np.max(galaxy_catalog._wldeblend_cat[_]) <= 27.0
    galaxy_catalog = WLDeblendGalaxyCatalog(
        pixel_scale=0.2,
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout="random",
        select_observable=["r_ab", "z_ab"],
        select_upper_limit=[27, 26],
        select_lower_limit=[25, 22],
    )
    assert np.min(galaxy_catalog._wldeblend_cat["r_ab"]) >= 25.0
    assert np.max(galaxy_catalog._wldeblend_cat["r_ab"]) <= 27.0
    assert np.min(galaxy_catalog._wldeblend_cat["z_ab"]) >= 22.0
    assert np.max(galaxy_catalog._wldeblend_cat["z_ab"]) <= 26.0
    return


def _top_n_peaks(arr, n):
    arr = arr.copy()
    idxs = []
    for _ in range(n):
        flat = np.argmax(arr)
        iy, ix = np.unravel_index(flat, arr.shape)
        idxs.append((iy, ix))
        # suppress neighborhood to avoid duplicate
        y0, y1 = max(0, iy-2), min(arr.shape[0], iy+3)
        x0, x1 = max(0, ix-2), min(arr.shape[1], ix+3)
        arr[y0:y1, x0:x1] = -np.inf
    return np.array(idxs, dtype=float)  # (N,2) rows, cols


def test_custom_layout_position_recovery():
    """
    Project the truth back to UV:
      - Provide custom galaxies + uv_shift (arcsec)
      - Make PSF small and galaxies bright
      - Find image peaks and compare to input UV
      - Tolerance: 0.5 pixel
    """
    coadd_dim = 101
    pixel_scale = 0.2
    tol_pix = 0.5

    gal_list = [
        galsim.Exponential(half_light_radius=0.6, flux=5e5),
        galsim.DeVaucouleurs(half_light_radius=0.6, flux=4e5),
        galsim.Exponential(half_light_radius=0.6, flux=3e5),
    ]

    # Shifts in the uv plane (arcsec)
    uv_shift = [
        (0.0, 0.0),
        (8.0, -5.0),
        (-6.0, 3.0),
    ]

    rng = np.random.RandomState(2026)

    # Small PSF
    psf = make_fixed_psf(psf_type='gauss', psf_fwhm=0.05)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type='custom',
        layout='custom',
        coadd_dim=coadd_dim,
        buff=10,
        pixel_scale=pixel_scale,
        gal_list=gal_list,
        uv_shift=uv_shift,
        simple_coadd_bbox=True,
    )

    sim = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        bands=['i'],
        psf=psf,
        g1=0.,
        g2=0.,
    )

    coadd_bbox = sim['coadd_bbox']
    coadd_wcs = sim['coadd_wcs']
    se_wcs = sim['se_wcs']['i'][0]

    coadd_cen_gs_skypos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    )

    world_pos_list = [
        coadd_cen_gs_skypos.deproject(u * galsim.arcsec, v * galsim.arcsec)
        for (u, v) in uv_shift]

    image_pos_list = np.array([
        se_wcs.toImage(pos) for pos in world_pos_list
    ])

    img = sim['band_data']['i'][0].image.array

    peak_uv_pix = _top_n_peaks(img, n=len(uv_shift))

    # Convert to numpy indexing (0-based)
    image_pos_list = np.array(
        [np.array([pos.y - 1, pos.x - 1]) for pos in image_pos_list]
    )

    # Match each truth to its closest peak
    used = set()
    for i in range(len(image_pos_list)):
        d2 = np.sum((peak_uv_pix - image_pos_list[i])**2, axis=1)
        j = int(np.argmin(d2))
        assert j not in used, "Duplicate peak assignment"
        used.add(j)
        dy = image_pos_list[i, 0] - peak_uv_pix[j, 0]
        dx = image_pos_list[i, 1] - peak_uv_pix[j, 1]
        dr = np.hypot(dx, dy)
        # if False:
        #     import matplotlib.pyplot as plt
        #     plt.figure(figsize=(12,12))
        #     plt.imshow(img, origin='lower', cmap='Greys', interpolation='nearest')
        #     plt.scatter(peak_uv_pix[:,1], peak_uv_pix[:,0], marker='x', color='red')
        #     plt.scatter(image_pos_list[:,1], image_pos_list[:,0],
        #                 marker='o', facecolors='none', edgecolors='blue')
        #     plt.title('Red X: detected peaks; Blue O: truth positions')
        #     plt.savefig('test_custom_layout_position_recovery.png')
        assert dr <= tol_pix, (
            f"Peak offset {dr:.3f} px (tol {tol_pix});" + f"Δx={dx:.3f}, Δy={dy:.3f}"
        )
