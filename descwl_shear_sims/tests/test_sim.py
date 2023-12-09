import os
import pytest
import numpy as np
from copy import deepcopy
import lsst.afw.image as afw_image
import lsst.afw.geom as afw_geom
from descwl_shear_sims.surveys import get_survey, DEFAULT_SURVEY_BANDS

from descwl_shear_sims.galaxies import make_galaxy_catalog, DEFAULT_FIXED_GAL_CONFIG
from descwl_shear_sims.stars import StarCatalog, make_star_catalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf

from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.constants import ZERO_POINT

from descwl_shear_sims.shear import ShearConstant

shear_obj = ShearConstant(g1=0.02, g2=0.)


@pytest.mark.parametrize('dither,rotate', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_sim_smoke(dither, rotate):
    """
    test sim can run
    """
    seed = 74321
    rng = np.random.RandomState(seed)

    coadd_dim = 351
    psf_dim = 51
    bands = ["i"]
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=30,
        layout="grid",
    )

    psf = make_fixed_psf(psf_type="gauss")
    data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        bands=bands,
        shear_obj=shear_obj,
        psf=psf,
        dither=dither,
        rotate=rotate,
    )

    for key in ['band_data', 'coadd_wcs', 'psf_dims', 'coadd_bbox']:
        assert key in data

    assert isinstance(data['coadd_wcs'], afw_geom.SkyWcs)
    assert data['psf_dims'] == (psf_dim, )*2
    extent = data['coadd_bbox'].getDimensions()
    edims = (extent.getX(), extent.getY())
    assert edims == (coadd_dim, )*2

    for band in bands:
        assert band in data['band_data']

    for band, bdata in data['band_data'].items():
        assert len(bdata) == 1
        assert isinstance(bdata[0], afw_image.ExposureF)


def test_sim_se_dim():
    """
    test sim can run
    """
    seed = 74321
    rng = np.random.RandomState(seed)

    coadd_dim = 351
    se_dim = 351
    psf_dim = 51
    bands = ["i"]
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=30,
        layout="grid",
    )

    psf = make_fixed_psf(psf_type="gauss")
    data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        psf_dim=psf_dim,
        bands=bands,
        shear_obj=shear_obj,
        psf=psf,
    )

    dims = (se_dim, )*2
    assert data['band_data']['i'][0].image.array.shape == dims


@pytest.mark.parametrize("rotate", [False, True])
def test_sim_exp_mag(rotate, show=False):
    """
    test we get the right mag.  Also test we get small flux when we rotate and
    there is nothing at the sub image location we choose

    This requires getting lucky with the rotation, so try a few
    """

    ntrial = 10

    bands = ["i"]
    seed = 55
    coadd_dim = 301
    rng = np.random.RandomState(seed)

    # use fixed single epoch dim so we can look in the same spot for the object
    se_dim = get_se_dim(coadd_dim=coadd_dim, dither=False, rotate=True)

    ok = False
    for i in range(ntrial):
        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type="fixed",
            coadd_dim=coadd_dim,
            buff=30,
            layout="grid",
        )

        psf = make_fixed_psf(psf_type="gauss")
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            se_dim=se_dim,
            shear_obj=shear_obj,
            psf=psf,
            bands=bands,
            rotate=rotate,
        )

        image = sim_data["band_data"]["i"][0].image.array
        sub_image = image[93:93+25, 88:88+25]
        subim_sum = sub_image.sum()

        if show:
            import matplotlib.pyplot as mplt
            fig, ax = mplt.subplots(nrows=1, ncols=2)
            ax[0].imshow(image)
            ax[1].imshow(sub_image)
            mplt.show()

        if rotate:
            # we expect nothing there
            if abs(subim_sum) < 30:
                ok = True
                break

        else:
            # we expect something there with about the right magnitude
            mag = ZERO_POINT - 2.5*np.log10(subim_sum)
            assert abs(mag - DEFAULT_FIXED_GAL_CONFIG['mag']) < 0.005

            break

    if rotate:
        assert ok, 'expected at least one to be empty upon rotation'


@pytest.mark.parametrize("psf_type", ["gauss", "moffat", "ps"])
def test_sim_psf_type(psf_type):

    seed = 431
    rng = np.random.RandomState(seed)

    dither = True
    rotate = True
    coadd_dim = 101
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=5,
        layout="grid",
    )

    if psf_type == "ps":
        se_dim = get_se_dim(coadd_dim=coadd_dim, dither=dither, rotate=rotate)
        psf = make_ps_psf(rng=rng, dim=se_dim)
    else:
        psf = make_fixed_psf(psf_type=psf_type)

    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
        dither=dither,
        rotate=rotate,
    )


@pytest.mark.parametrize('epochs_per_band', [1, 2, 3])
def test_sim_epochs(epochs_per_band):

    seed = 7421
    bands = ["r", "i", "z"]
    coadd_dim = 301
    psf_dim = 47

    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=10,
        layout="grid",
    )

    psf = make_fixed_psf(psf_type="gauss")
    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        shear_obj=shear_obj,
        psf=psf,
        bands=bands,
        epochs_per_band=epochs_per_band,
    )

    band_data = sim_data['band_data']
    for band in bands:
        assert band in band_data
        assert len(band_data[band]) == epochs_per_band


@pytest.mark.parametrize(
    "layout, gal_type",
    [
        ("grid", "fixed"),
        ("random", "fixed"),
        ("random_disk", "fixed"),
        ("hex", "fixed"),
        ("grid", "wldeblend"),
        ("hex", "wldeblend"),
    ],
)
def test_sim_layout(layout, gal_type):
    seed = 7421
    coadd_dim = 201
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type=gal_type,
        coadd_dim=coadd_dim,
        buff=30,
        layout=layout,
    )

    psf = make_fixed_psf(psf_type="gauss")
    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
    )


@pytest.mark.parametrize(
    "cosmic_rays, bad_columns",
    [(True, True),
     (True, False),
     (False, True),
     (True, True)],
)
def test_sim_defects(cosmic_rays, bad_columns):
    ntrial = 10
    seed = 7421
    rng = np.random.RandomState(seed)

    coadd_dim = 201

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        layout="grid",
        buff=30,
    )

    psf = make_fixed_psf(psf_type="gauss")

    for itrial in range(ntrial):
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            shear_obj=shear_obj,
            psf=psf,
            cosmic_rays=cosmic_rays,
            bad_columns=bad_columns,
        )

        for band, band_exps in sim_data['band_data'].items():
            for exp in band_exps:
                image = exp.image.array
                mask = exp.mask.array
                flags = exp.mask.getPlaneBitMask(('CR', 'BAD'))

                if bad_columns or cosmic_rays:

                    wnan = np.where(np.isnan(image))
                    wflagged = np.where((mask & flags) != 0)
                    assert wnan[0].size == wflagged[0].size


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_sim_wldeblend():
    seed = 7421
    coadd_dim = 201
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="wldeblend",
        coadd_dim=coadd_dim,
        buff=30,
        layout="random",
    )

    psf = make_fixed_psf(psf_type="moffat")
    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
    )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('density,min_density,max_density', [
    (None, 40, 100),
    (20, None, None),
])
def test_sim_stars(density, min_density, max_density):
    seed = 7421
    coadd_dim = 201
    buff = 30

    config = {
        'density': density,
        'min_density': min_density,
        'max_density': max_density,
    }
    for use_maker in (False, True):
        rng = np.random.RandomState(seed)

        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type="wldeblend",
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
        )
        assert len(galaxy_catalog) == galaxy_catalog.shifts_array.size

        if use_maker:
            star_catalog = make_star_catalog(
                rng=rng,
                coadd_dim=coadd_dim,
                buff=buff,
                star_config=config,
            )

        else:
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=coadd_dim,
                buff=buff,
                density=config['density'],
                min_density=config['min_density'],
                max_density=config['max_density'],
            )

        assert len(star_catalog) == star_catalog.shifts_array.size

        psf = make_fixed_psf(psf_type="moffat")

        # tests that we actually get bright objects set are in
        # test_star_masks_and_bleeds

        data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=coadd_dim,
            shear_obj=shear_obj,
            psf=psf,
        )

        if not use_maker:
            data_nomaker = data
        else:
            assert np.all(
                data['band_data']['i'][0].image.array ==
                data_nomaker['band_data']['i'][0].image.array
            )


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_sim_star_bleeds():
    seed = 7421
    coadd_dim = 201
    buff = 30
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="wldeblend",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="random",
    )

    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=100,
    )

    psf = make_fixed_psf(psf_type="moffat")

    # tests that we actually get saturation are in test_star_masks_and_bleeds

    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
        star_bleeds=True,
    )


@pytest.mark.parametrize("draw_method", (None, "auto", "phot"))
def test_sim_draw_method_smoke(draw_method):
    seed = 881
    coadd_dim = 201
    rng = np.random.RandomState(seed)

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=30,
        layout='grid',
    )

    kw = {}
    if draw_method is not None:
        kw['draw_method'] = draw_method

    psf = make_fixed_psf(psf_type="gauss")
    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
        **kw
    )


@pytest.mark.parametrize(
    "psf_fwhm",
    [0.6, 0.7],
)
def test_sim_hsc(psf_fwhm=0.6):
    seed = 7421
    coadd_dim = 201
    buff = 20
    layout = "random"
    rng = np.random.RandomState(seed)
    calib_mag_zero = 27
    survey_name = "HSC"
    gal_type = "wldeblend"

    pixel_scale = get_survey(
        gal_type=gal_type,
        band=deepcopy(DEFAULT_SURVEY_BANDS)[survey_name],
        survey_name=survey_name,
    ).pixel_scale

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type=gal_type,
        coadd_dim=coadd_dim,
        buff=buff,
        pixel_scale=pixel_scale,
        layout=layout,
    )

    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=10,
        pixel_scale=pixel_scale,
        layout=layout,
    )

    psf = make_fixed_psf(
        psf_type="moffat",
        psf_fwhm=psf_fwhm,
    )
    _ = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
        star_catalog=star_catalog,
        calib_mag_zero=calib_mag_zero,
        survey_name=survey_name,
    )

    exposure = _["band_data"]["i"][0]
    # check the pixel scale
    wcs = exposure.getWcs()
    pix_scale = wcs.getPixelScale()
    assert np.abs(pix_scale.asArcseconds() - pixel_scale) < 1e-5
    assert np.abs(0.168 - pixel_scale) < 1e-5

    # check the magnitude zero point
    zero_flux = 10. ** (0.4 * calib_mag_zero)
    flux = exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
    assert np.abs(zero_flux - flux) < 1e-3


@pytest.mark.parametrize(
    "psf_fwhm",
    [1.0, 1.2],
)
def test_sim_des(psf_fwhm=1.0):
    seed = 7421
    coadd_dim = 201
    buff = 20
    layout = "random"
    rng = np.random.RandomState(seed)
    calib_mag_zero = 30
    survey_name = "DES"
    gal_type = "wldeblend"

    pixel_scale = get_survey(
        gal_type=gal_type,
        band=deepcopy(DEFAULT_SURVEY_BANDS)[survey_name],
        survey_name=survey_name,
    ).pixel_scale

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type=gal_type,
        coadd_dim=coadd_dim,
        buff=buff,
        pixel_scale=pixel_scale,
        layout=layout,
    )

    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=10,
        pixel_scale=pixel_scale,
        layout=layout,
    )

    psf = make_fixed_psf(
        psf_type="moffat",
        psf_fwhm=psf_fwhm,
    )
    _ = make_sim(
        rng=rng,
        bands=["i"],
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        shear_obj=shear_obj,
        psf=psf,
        star_catalog=star_catalog,
        calib_mag_zero=calib_mag_zero,
        survey_name=survey_name,
    )

    exposure = _["band_data"]["i"][0]
    # check the pixel scale
    wcs = exposure.getWcs()
    pix_scale = wcs.getPixelScale()
    assert np.abs(pix_scale.asArcseconds() - pixel_scale) < 1e-5
    assert np.abs(0.263 - pixel_scale) < 1e-5

    # check the magnitude zero point
    zero_flux = 10. ** (0.4 * calib_mag_zero)
    flux = exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
    assert np.abs(zero_flux - flux) < 1e-3


if __name__ == '__main__':
    # test_sim_layout("hex", "wldeblend")
    # for rotate in (False, True):
    #     test_sim_exp_mag(rotate, show=True)
    test_sim_des()
    test_sim_hsc()
