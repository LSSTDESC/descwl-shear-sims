from copy import deepcopy
import esutil as eu
import galsim
import numpy as np

import lsst.afw.image as afw_image
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

from .lsst_bits import get_flagval
from .saturation import saturate_image_and_mask, BAND_SAT_VALS
from .surveys import get_survey, rescale_wldeblend_exp, DEFAULT_SURVEY_BANDS
from .constants import SCALE, WORLD_ORIGIN, ZERO_POINT, SIM_INCLUSION_PADDING
from .artifacts import add_bleed, get_max_mag_with_bleed
from .masking import (
    get_bmask_and_set_image,
    calculate_bright_star_mask_radius,
)
from .objlists import get_objlist
from .psfs import make_dm_psf
from .wcs import (
    make_dm_wcs, make_se_wcs, make_coadd_dm_wcs, make_coadd_dm_wcs_simple
)
from .shear import ShearConstant


DEFAULT_SIM_CONFIG = {
    "gal_type": "fixed",  # note "exp" also means "fixed" for back compat
    "psf_type": "gauss",
    "psf_dim": 51,
    "psf_variation_factor": 1,  # for power spectrum psf
    # randomize the psf fwhm and shape for each trial.  PSF is still same
    # for all epochs/bands
    "randomize_psf": False,
    "coadd_dim": 250,
    "se_dim": None,
    "buff": 0,
    "layout": "grid",
    "sep": None,
    "dither": False,
    "rotate": False,
    "bands": ["i"],
    "epochs_per_band": 1,
    "noise_factor": 1.0,
    "stars": False,
    "star_bleeds": False,
    "draw_stars": True,
    "cosmic_rays": False,
    "bad_columns": False,
    "sky_n_sigma": None,
    "survey_name": "LSST",
    "draw_noise": True,
}


def make_sim(
    *,
    rng,
    galaxy_catalog,
    psf,
    shear_obj=None,
    se_dim=None,
    draw_gals=True,
    star_catalog=None,
    draw_stars=True,
    draw_bright=True,
    psf_dim=51,
    dither=False,
    dither_size=0.5,
    rotate=False,
    bands=["i"],
    epochs_per_band=1,
    noise_factor=1.0,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
    sky_n_sigma=None,
    draw_method="auto",
    calib_mag_zero=ZERO_POINT,
    survey_name="LSST",
    theta0=0.0,
    g1=None,
    g2=None,
    coadd_dim=None,
    simple_coadd_bbox=False,
    draw_noise=True,
    im_precision="float"
):
    """
    Make simulation data

    Parameters
    ----------
    rng: numpy.random.RandomState
        Numpy random state
    galaxy_catalog: catalog
        E.g. WLDeblendGalaxyCatalog or FixedGalaxyCatalog
    shear_obj:
        shear distortion object
    psf: GSObject or PowerSpectrumPSF
        The psf object or power spectrum psf
    se_dim: int, optional
        Force the single epoch images to have this dimension.  If not
        sent it is calculated to be large enough to encompass the coadd
        with rotations and small dithers.
    star_catalog: catalog, optional
        A catalog to generate star locations and fluxes.  See the psfs module
    psf_dim: int, optional
        Dimensions of psf image.  Default 51
    dither: bool, optional
        Whether to dither the images at the pixel level, default False
    dither_size: float, optional
        The amplitude of dithering in unit of a fraction of a pixel
        for testing pixel interpolation.
        All SE WCS will be given random dithers with this amplitude in both image
        x and y direction.
        Value must be between 0 and 1.  default 0.5.
    rotate: bool, optional
        Whether to randomly rotate the image exposures randomly [not the
        rotation of intrinsic galaxies], default False
    bands: list, optional
        Default ['i']
    epochs_per_band: int, optional
        Number of epochs per band, default 1
    noise_factor: float, optional
        Factor by which to multiply the noise, default 1
    cosmic_rays: bool, optional
        If set to True, add cosmic rays.  Default False.
    bad_columns: bool, optional
        If set to True, add bad columns.  Default False.
    star_bleeds: bool, optional
        If set to True, draw simulated bleed trails for saturated stars.
        Default False
    sky_n_sigma: float, optional
        Number of sigma to set the sky value.  Can be negative to
        mock up a sky oversubtraction.  Default None.
    draw_method: string, optional
        Draw method for galaxy galsim objects, default 'auto'.  Set to
        'phot' to get poisson noise.  Note this is much slower.
    theta0: float, optional
        rotation angle of intrinsic galaxy shapes and positions on the sky,
        default 0, in units of radians
    g1,g2: float optional
        reduced shear distortions
    coadd_dim: optional, int
        Dimensions for planned final coadd.  This is used for generating
        the final coadd WCS and deteremines some properties of
        the single epoch images.
    simple_coadd_bbox: optional, bool. Default: False
        Whether to force the center of coadd boundary box (which is the default
        center single exposure) at the world_origin
    draw_noise: optional, bool
        Whether draw image noise
    im_precision: optional, str
        Image precision, 'float' or 'double'.  Default 'float'

    Returns
    -------
    sim_data: dict
        band_data: a dict keyed by band name, holding a list of exp
        coadd_wcs: lsst.afw.geom.makeSkyWcs
        psf_dims: (int, int)
        coadd_dims: (int, int)
        coadd_bbox: lsst.geom.Box2I
        bright_info: structured array
            fields are
            ra, dec: sky position of bright stars
            radius_pixels: radius of mask in pixels
            has_bleed: bool, True if there is a bleed trail
        truth_info: structured array
            fields are
            ra, dec: sky position of input galaxies
            z: redshift of input galaxies
            image_x, image_y: image position of input galaxies
        se_wcs: a dict keyed by band name, holding a list of se_wcs
    """

    if im_precision == "float":
        im_dtype = np.float32
    elif im_precision == "double":
        im_dtype = np.float64
    else:
        raise ValueError("im_precision must be 'float' or 'double'")

    # Get the pixel scale using a default band from the survey
    _bd = deepcopy(DEFAULT_SURVEY_BANDS)[survey_name]
    pixel_scale = get_survey(
        gal_type=galaxy_catalog.gal_type,
        band=_bd,
        survey_name=survey_name,
    ).pixel_scale

    if simple_coadd_bbox:
        # Force to use a simple coadd boundary box
        # where the center of the boundary box is the image origin and it
        # matches to the world origin of the catalog's layout. Note that he
        # center of the boundary box is the image_origin of the single exposure
        # (SE).
        if hasattr(galaxy_catalog.layout, "wcs"):
            origin = galaxy_catalog.layout.wcs.getSkyOrigin()
            world_origin = galsim.CelestialCoord(
                ra=origin.getRa().asDegrees() * galsim.degrees,
                dec=origin.getDec().asDegrees() * galsim.degrees,
            )
        else:
            world_origin = WORLD_ORIGIN
        coadd_wcs, coadd_bbox = make_coadd_dm_wcs_simple(
            coadd_dim,
            pixel_scale=pixel_scale,
            world_origin=world_origin,
        )
    else:
        if (
            hasattr(galaxy_catalog.layout, "wcs")
            and hasattr(galaxy_catalog.layout, "bbox")
        ):
            coadd_wcs = galaxy_catalog.layout.wcs
            coadd_bbox = galaxy_catalog.layout.bbox
        else:
            coadd_wcs, coadd_bbox = make_coadd_dm_wcs(
                coadd_dim,
                pixel_scale=pixel_scale,
            )

    # get the sky position of the coadd image center. For simple_coadd_bbox ==
    # True, coadd_bbox_cen_gs_skypos is WORLD_ORIGIN (see unit test_make_exp in
    # test_sim.py)
    coadd_bbox_cen_gs_skypos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    )
    if se_dim is None:
        coadd_scale = coadd_wcs.getPixelScale().asArcseconds()
        coadd_dim = coadd_bbox.getHeight()
        se_dim = get_se_dim(
            coadd_scale=coadd_scale,
            coadd_dim=coadd_dim,
            se_scale=pixel_scale,
            rotate=rotate,
        )

    if shear_obj is None:
        assert g1 is not None and g2 is not None
        shear_obj = ShearConstant(g1=float(g1), g2=float(g2))
    band_data = {}
    bright_info = []
    truth_info = []
    se_wcs = {}
    for band in bands:
        survey = get_survey(
            gal_type=galaxy_catalog.gal_type,
            band=band,
            survey_name=survey_name,
        )
        noise_for_gsparams = survey.noise * noise_factor
        noise_per_epoch = survey.noise * np.sqrt(epochs_per_band) * noise_factor

        # go down to coadd depth in this band, not dividing by sqrt(nbands)
        # but we could if we want to be more conservative
        mask_threshold = survey.noise * noise_factor

        lists = get_objlist(
            galaxy_catalog=galaxy_catalog,
            survey=survey,
            star_catalog=star_catalog,
            noise=noise_for_gsparams,
        )

        # note the bright list is the same for all exps, so we only add
        # bright_info once

        bdata_list = []
        se_wcs_list = []
        for epoch in range(epochs_per_band):
            exp, this_bright_info, this_truth_info, this_se_wcs = make_exp(
                rng=rng,
                band=band,
                noise=noise_per_epoch,
                objlist=lists["objlist"],
                shifts=lists["shifts"],
                redshifts=lists["redshifts"],
                dim=se_dim,
                psf=psf,
                psf_dim=psf_dim,
                shear_obj=shear_obj,
                draw_gals=draw_gals,
                star_objlist=lists["star_objlist"],
                star_shifts=lists["star_shifts"],
                draw_stars=draw_stars,
                draw_bright=draw_bright,
                bright_objlist=lists["bright_objlist"],
                bright_shifts=lists["bright_shifts"],
                bright_mags=lists["bright_mags"],
                coadd_bbox_cen_gs_skypos=coadd_bbox_cen_gs_skypos,
                dither=dither,
                dither_size=dither_size,
                rotate=rotate,
                mask_threshold=mask_threshold,
                cosmic_rays=cosmic_rays,
                bad_columns=bad_columns,
                star_bleeds=star_bleeds,
                sky_n_sigma=sky_n_sigma,
                draw_method=draw_method,
                theta0=theta0,
                pixel_scale=pixel_scale,
                calib_mag_zero=calib_mag_zero,
                draw_noise=draw_noise,
                indexes=lists["indexes"],
                im_dtype=im_dtype,
            )
            if epoch == 0:
                bright_info += this_bright_info
            if epoch == 0 and band == bands[0]:
                # only record the input catalog info for one band
                truth_info += this_truth_info
            if galaxy_catalog.gal_type == "wldeblend":
                # rescale the image to calibrate it to magnitude zero point
                # = calib_mag_zero
                rescale_wldeblend_exp(
                    survey=survey.descwl_survey,
                    exp=exp,
                    calib_mag_zero=calib_mag_zero,
                )

            if survey_name == "LSST":
                # mark high pixels SAT and also set sat value in image for
                # any pixels already marked SAT
                saturate_image_and_mask(
                    image=exp.image.array,
                    bmask=exp.mask.array,
                    sat_val=BAND_SAT_VALS[band],
                    flagval=get_flagval("SAT"),
                )

            bdata_list.append(exp)
            se_wcs_list.append(this_se_wcs)

        band_data[band] = bdata_list
        se_wcs[band] = se_wcs_list

    bright_info = eu.numpy_util.combine_arrlist(bright_info)
    truth_info = eu.numpy_util.combine_arrlist(truth_info)

    return {
        "band_data": band_data,
        "coadd_wcs": coadd_wcs,
        "psf_dims": (psf_dim,) * 2,
        "coadd_dims": (coadd_dim,) * 2,
        "coadd_bbox": coadd_bbox,
        "bright_info": bright_info,
        "truth_info": truth_info,
        "se_wcs": se_wcs,
    }


def make_exp(
    *,
    rng,
    band,
    noise,
    objlist,
    shifts,
    redshifts,
    dim,
    psf,
    psf_dim,
    shear_obj,
    draw_gals=True,
    star_objlist=None,
    star_shifts=None,
    draw_stars=True,
    draw_bright=True,
    bright_objlist=None,
    bright_shifts=None,
    bright_mags=None,
    coadd_bbox_cen_gs_skypos=None,
    dither=False,
    dither_size=None,
    rotate=False,
    mask_threshold=None,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
    sky_n_sigma=None,
    draw_method="auto",
    theta0=0.0,
    pixel_scale=SCALE,
    calib_mag_zero=ZERO_POINT,
    draw_noise=True,
    indexes=None,
    se_wcs=None,
    im_dtype=None,
):
    """
    Make an Signle Exposure (SE) observation

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    band: str
        Band as a string, e.g. 'i'
    noise: float
        Gaussian noise level
    objlist: list
        List of GSObj
    shifts: array
        List of PositionD representing offsets
    dim: int
        Dimension of image
    psf: GSObject or PowerSpectrumPSF
        the psf
    psf_dim: int
        Dimensions of psf image that will be drawn when psf func is called
    dither: bool
        If set to True, dither randomly by a pixel width
    dither_size: float, optional
        The amplitude of dithering in unit of a fraction of a pixel
        for testing pixel interpolation.
        All se WCS will be dithered by this amount in both x and y directions.
        Value must be between 0 and 1.  default None.
    rotate: bool
        If set to True, rotate the image exposure randomly, note, this is not
        the rotation of intrinsic galaxies in ring test
    star_objlist: list
        List of GSObj for stars
    star_shifts: array
        List of PositionD for stars representing offsets
    draw_stars: bool, optional
        Draw the stars, don't just mask bright ones.  Default True.
    bright_objlist: list, optional
        List of GSObj for bright objects
    bright_shifts: array, optional
        List of PositionD for stars representing offsets
    bright_mags: array, optional
        Mags for the bright objects
    coadd_bbox_cen_gs_skypos: galsim.CelestialCoord, optional
        The sky position of the center (origin) of the coadd we
        will make, as a galsim object not stack object
    mask_threshold: float
        Bright masks are created such that the profile goes out
        to this threshold
    cosmic_rays: bool
        If True, put in cosmic rays
    bad_columns: bool
        If True, put in bad columns
    star_bleeds: bool
        If True, add bleed trails to stars
    sky_n_sigma: float
        Number of sigma to set the sky value.  Can be negative to
        mock up a sky oversubtraction.  Default None.
    draw_method: string
        Draw method for galsim objects, default 'auto'.  Set to
        'phot' to get poisson noise.  Note this is much slower.
    theta0: float
        rotation angle of intrinsic galaxies and positions [for ring test],
        default 0, in units of radians
    pixel_scale: float
        pixel scale of single exposure in arcsec
    calib_mag_zero: float
        magnitude zero point after calibration
    indexes: list
        list of indexes in the input galaxy catalog, default: None
    se_wcs: galsim WCS
        wcs for single exposure, default: None
    im_dtype: numpy dtype
        numpy dtype for images

    Returns
    -------
    exp: lsst.afw.image.ExposureF
        Exposure data
    bright_info: list of structured arrays
        fields are
        ra, dec: sky position of bright stars
        radius_pixels: radius of mask in pixels
        has_bleed: bool, True if there is a bleed trail
    truth_info: structured array
        fields are
        index: index in the input catalog
        ra, dec: sky position of input galaxies
        z: redshift of input galaxies
        image_x, image_y: image position of input galaxies
    se_wcs: galsim wcs
        the wcs of the single exposure

    """
    dims = [int(dim)] * 2

    if se_wcs is None:
        # If no se_wcs input, we make a wcs with world origin set to the center
        # of the coadds (the center of the galaxy_catalog.layout)

        # The se origin is set to the center of the image
        # Galsim image uses 1 offset. An array with length =dim=5
        # The center is at 3=(5+1)/2
        cen = (np.array(dims) + 1) / 2
        se_origin = galsim.PositionD(x=cen[1], y=cen[0])
        se_wcs = make_se_wcs(
            pixel_scale=pixel_scale,
            image_origin=se_origin,
            world_origin=coadd_bbox_cen_gs_skypos,
            dither=dither,
            dither_size=dither_size,
            rotate=rotate,
            rng=rng,
        )
    else:
        # if with se_wcs input, we make sure the wcs is consistent with the
        # other inputs
        cen = se_wcs.crpix
        se_origin = galsim.PositionD(x=cen[1], y=cen[0])
        pixel_area = se_wcs.pixelArea(se_origin)
        if not (pixel_area - pixel_scale ** 2.0) < pixel_scale ** 2.0 / 100.0:
            raise ValueError("The input se_wcs has wrong pixel scale")

    image = galsim.Image(dim, dim, wcs=se_wcs, dtype=im_dtype)

    if objlist is not None and draw_gals:
        assert shifts is not None
        truth_info = _draw_objects(
            image,
            objlist,
            shifts,
            redshifts,
            psf,
            draw_method,
            coadd_bbox_cen_gs_skypos,
            rng,
            shear_obj=shear_obj,
            theta0=theta0,
            indexes=indexes,
            im_dtype=im_dtype,
        )
    else:
        truth_info = []

    if star_objlist is not None and draw_stars:
        assert star_shifts is not None, "send star_shifts with star_objlist"
        _draw_objects(
            image,
            star_objlist,
            star_shifts,
            None,
            psf,
            draw_method,
            coadd_bbox_cen_gs_skypos,
            rng,
            im_dtype=im_dtype,
        )

    if draw_noise:
        image.array[:, :] += rng.normal(scale=noise, size=dims)
    if sky_n_sigma is not None:
        image.array[:, :] += sky_n_sigma * noise

    # pixels flagged as bad cols and cosmics will get
    # set to np.nan
    bmask = get_bmask_and_set_image(
        image=image,
        rng=rng,
        cosmic_rays=cosmic_rays,
        bad_columns=bad_columns,
    )

    if bright_objlist is not None and draw_bright:
        bright_info = _draw_bright_objects(
            image=image,
            noise=noise,
            origin=se_origin,
            bmask=bmask,
            band=band,
            objlist=bright_objlist,
            shifts=bright_shifts,
            mags=bright_mags,
            psf=psf,
            coadd_bbox_cen_gs_skypos=coadd_bbox_cen_gs_skypos,
            mask_threshold=mask_threshold,
            rng=rng,
            star_bleeds=star_bleeds,
            draw_stars=draw_stars,
            im_dtype=im_dtype,
        )
    else:
        bright_info = []

    dm_wcs = make_dm_wcs(se_wcs)
    dm_psf = make_dm_psf(psf=psf, psf_dim=psf_dim, wcs=se_wcs)

    variance = image.copy()
    variance.array[:, :] = noise**2

    masked_image = afw_image.MaskedImage(dim, dim, dtype=im_dtype)
    masked_image.image.array[:, :] = image.array
    masked_image.variance.array[:, :] = variance.array
    masked_image.mask.array[:, :] = bmask.array

    exp = afw_image.Exposure(masked_image, dtype=im_dtype)

    # Prepare the frc, and save it to the DM exposure
    # It can be retrieved as follow
    # zero_flux =  exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
    # magz    =   np.log10(zero_flux)*2.5 # magnitude zero point
    zero_flux = 10.0 ** (0.4 * calib_mag_zero)
    photoCalib = afw_image.makePhotoCalibFromCalibZeroPoint(zero_flux)
    exp.setPhotoCalib(photoCalib)

    filter_label = afw_image.FilterLabel(band=band, physical=band)
    exp.setFilter(filter_label)

    exp.setPsf(dm_psf)

    exp.setWcs(dm_wcs)

    detector = DetectorWrapper().detector
    exp.setDetector(detector)

    return exp, bright_info, truth_info, se_wcs


def _draw_objects(
    image,
    objlist,
    shifts,
    redshifts,
    psf,
    draw_method,
    coadd_bbox_cen_gs_skypos,
    rng,
    shear_obj=None,
    theta0=None,
    indexes=None,
    im_dtype=None,
):
    """
    draw objects and return the input galaxy catalog.

    Returns
    -------
        truth_info: structured array
        fields are
        index: index in the input galaxy catalog
        ra, dec: sky position of input galaxies
        z: redshift of input galaxies
        image_x, image_y: image position of input galaxies
    """

    wcs = image.wcs
    kw = {}
    if draw_method == "phot":
        kw["maxN"] = 1_000_000
        kw["rng"] = galsim.BaseDeviate(seed=rng.randint(low=0, high=2**30))

    if redshifts is None:
        # set redshifts to -1 if not sepcified
        redshifts = np.ones(len(objlist)) * -1.0

    if indexes is None:
        # set input galaxy indexes to -1 if not sepcified
        indexes = np.ones(len(objlist)) * -1.0

    truth_info = []

    for obj, shift, z, ind in zip(objlist, shifts, redshifts, indexes):

        if theta0 is not None:
            ang = theta0 * galsim.radians
            # rotation on intrinsic galaxies comes before shear distortion
            obj = obj.rotate(ang)
            shift = _rotate_pos(shift, theta0)

        if shear_obj is not None:
            distor_res = shear_obj.distort_galaxy(obj, shift, z)
            obj = distor_res["gso"]
            lensed_shift = distor_res["lensed_shift"]
            gamma1 = distor_res["gamma1"]
            gamma2 = distor_res["gamma2"]
            kappa = distor_res["kappa"]
        else:
            lensed_shift = shift
            gamma1, gamma2, kappa = 0.0, 0.0, 0.0

        # Deproject from u,v onto sphere. Then use wcs to get to image pos.
        world_pos = coadd_bbox_cen_gs_skypos.deproject(
            lensed_shift.x * galsim.arcsec,
            lensed_shift.y * galsim.arcsec,
        )

        image_pos = wcs.toImage(world_pos)

        prelensed_world_pos = coadd_bbox_cen_gs_skypos.deproject(
            shift.x * galsim.arcsec,
            shift.y * galsim.arcsec,
        )
        prelensed_image_pos = wcs.toImage(prelensed_world_pos)

        if (
            (image.bounds.xmin - SIM_INCLUSION_PADDING) <
            image_pos.x < (image.bounds.xmax + SIM_INCLUSION_PADDING)
        ) and (
            (image.bounds.ymin - SIM_INCLUSION_PADDING)
            < image_pos.y < (image.bounds.ymax + SIM_INCLUSION_PADDING)
        ):
            local_wcs = wcs.local(image_pos=image_pos)
            convolved_object = get_convolved_object(obj, psf, image_pos)
            stamp = convolved_object.drawImage(
                center=image_pos, wcs=local_wcs, method=draw_method,
                dtype=im_dtype, **kw
            )

            b = stamp.bounds & image.bounds
            if b.isDefined():
                image[b] += stamp[b]

        info = get_truth_info_struct()

        info["index"] = (ind,)
        info["ra"] = world_pos.ra / galsim.degrees
        info["dec"] = world_pos.dec / galsim.degrees
        info["shift_x"] = (shift.x,)
        info["shift_y"] = (shift.y,)
        info["lensed_shift_x"] = (lensed_shift.x,)
        info["lensed_shift_y"] = (lensed_shift.y,)
        info["z"] = (z,)
        info["image_x"] = (image_pos.x - 1,)
        info["image_y"] = (image_pos.y - 1,)
        info["gamma1"] = (gamma1,)
        info["gamma2"] = (gamma2,)
        info["kappa"] = (kappa,)
        info["prelensed_image_x"] = (prelensed_image_pos.x - 1,)
        info["prelensed_image_y"] = (prelensed_image_pos.y - 1,)
        info["prelensed_ra"] = (prelensed_world_pos.ra / galsim.degrees,)
        info["prelensed_dec"] = (prelensed_world_pos.dec / galsim.degrees,)

        truth_info.append(info)

    return truth_info


def _draw_bright_objects(
    image,
    noise,
    origin,
    bmask,
    band,
    objlist,
    shifts,
    mags,
    psf,
    coadd_bbox_cen_gs_skypos,
    mask_threshold,
    rng,
    star_bleeds,
    draw_stars,
    im_dtype,
):
    """
    draw bright objects and return bright object information.

    Returns
    -------
    bright_info: list of structured arrays
        fields are
        ra, dec: sky position of bright stars
        radius_pixels: radius of mask in pixels
        has_bleed: bool, True if there is a bleed trail
    """
    # extra array needed to determine star mask accurately
    timage = image.copy()
    timage.setZero()

    grng = galsim.BaseDeviate(rng.randint(0, 2**30))

    assert shifts is not None
    assert mags is not None and len(mags) == len(shifts)
    assert mask_threshold is not None

    max_bleed_mag = get_max_mag_with_bleed(band=band)

    wcs = image.wcs

    bright_info = []

    indices = np.arange(len(objlist))
    for index, obj, shift, mag in zip(indices, objlist, shifts, mags):

        # profiles can have detectably sharp edges if the
        # profile is very high s/n and we have not set the
        # thresholds right in the gs params.
        #
        # photon shooting reduces these sharp edges, reducing
        # sensitivity to such an error

        world_pos = coadd_bbox_cen_gs_skypos.deproject(
            shift.x * galsim.arcsec,
            shift.y * galsim.arcsec,
        )

        image_pos = wcs.toImage(world_pos)
        local_wcs = wcs.local(image_pos=image_pos)

        convolved_object = get_convolved_object(obj, psf, image_pos)

        max_n_photons = 10_000_000
        # 0 means use the flux for n_photons
        n_photons = 0 if obj.flux < max_n_photons else max_n_photons

        stamp = convolved_object.drawImage(
            center=image_pos,
            wcs=local_wcs,
            method="phot",
            n_photons=n_photons,
            poisson_flux=True,
            maxN=1_000_000,  # shoot in batches this size
            rng=grng,
            dtype=im_dtype,
        )
        b = stamp.bounds & image.bounds
        if b.isDefined():
            if draw_stars:
                image[b] += stamp[b]

            # use smooth version for radius calculation
            stamp_fft = convolved_object.drawImage(
                center=image_pos,
                wcs=local_wcs,
                dtype=im_dtype,
            )

            timage[b] += stamp_fft[b]

            radius_pixels = calculate_bright_star_mask_radius(
                image=timage.array,
                objrow=image_pos.y,
                objcol=image_pos.x,
                threshold=mask_threshold,
            )

            info = get_bright_info_struct()
            info["ra"] = world_pos.ra / galsim.degrees
            info["dec"] = world_pos.dec / galsim.degrees
            info["radius_pixels"] = radius_pixels

            if star_bleeds and mag < max_bleed_mag:
                info["has_bleed"] = True
                add_bleed(
                    image=image.array,
                    bmask=bmask.array,
                    pos=image_pos,
                    mag=mag,
                    band=band,
                )
            else:
                info["has_bleed"] = False

            bright_info.append(info)

            # reset for next object
            timage.setZero()

    return bright_info


def get_sim_config(config=None):
    """
    Get a simulation configuration, with defaults that can
    be over-ridden by the input.  The defaults are in
    DEFAULT_SIM_CONFIG

    Parameters
    ----------
    config: dict, optional
        Dict of options to over ride the defaults

    Returns
    -------
    config dict
    """
    out_config = deepcopy(DEFAULT_SIM_CONFIG)
    sub_configs = ["gal_config", "star_config"]

    if config is not None:
        for key in config:
            if key not in out_config and key not in sub_configs:
                raise ValueError("bad key for sim: '%s'" % key)

        out_config.update(deepcopy(config))

    return out_config


def get_se_dim(
    *, coadd_dim, coadd_scale=None, se_scale=None, rotate=False
):
    """
    get single epoch (se) dimensions given coadd dim.

    Parameters
    ----------
    coadd_dim: int
        dimensions of coadd
    coadd_scale: float, optional
        pixel scale of coadd
    se_scale: float, optional
        pixel scale of single exposure
    rotate: bool, optional
        Whether there are random rotations of image exposure or not

    Returns
    -------
    integer dimensions of SE image
    """
    if (coadd_scale is None) or (se_scale is None):
        print("no coadd_scale or se_scale. Assume they are the same")
        dim = coadd_dim
    else:
        coadd_length = coadd_scale * coadd_dim
        dim = int((coadd_length) / se_scale + 0.5)
    if rotate:
        # make sure to completely cover the coadd
        se_dim = int(np.ceil(dim * np.sqrt(2))) + 20
    else:
        # make big enough to avoid boundary checks for downstream
        # which are 3 pixels
        se_dim = dim + 10

    return se_dim


def get_coadd_center_gs_pos(coadd_wcs, coadd_bbox):
    """
    get the sky position of the center of the coadd within the
    bbox as a galsim CelestialCoord

    Parameters
    ----------
    coadd_wcs: DM wcs
        The wcs for the coadd
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system

    Returns
    -------
    galsim CelestialCoord
    """

    # world origin is at center of the coadd, which itself
    # is in a bbox shifted from the overall WORLD_ORIGIN

    bbox_cen_skypos = coadd_wcs.pixelToSky(coadd_bbox.getCenter())

    return galsim.CelestialCoord(
        ra=float(bbox_cen_skypos.getRa()) * galsim.radians,
        dec=float(bbox_cen_skypos.getDec()) * galsim.radians,
    )


def get_convolved_object(obj, psf, image_pos):
    """
    Get a convolved object for either constant or varying psf

    Parameters
    ----------
    obj: galsim object
        The object to be convolved
    psf: galsim object or PowerSpectrumPSF
        Constant or varying psf
    image_pos: galsim.PositionD
        For varying psfs, will be used to get the psf

    Returns
    -------
    galsim object
    """
    if isinstance(psf, galsim.GSObject):
        convolved_object = galsim.Convolve(obj, psf)
    else:
        psf_gsobj = psf.getPSF(image_pos)
        convolved_object = galsim.Convolve(obj, psf_gsobj)

    return convolved_object


def get_bright_info_struct():
    dt = [
        ("ra", "f8"),
        ("dec", "f8"),
        ("radius_pixels", "f4"),
        ("has_bleed", bool),
    ]
    return np.zeros(1, dtype=dt)


def get_truth_info_struct():
    dt = [
        ("index", "i4"),
        ("ra", "f8"),
        ("dec", "f8"),
        ("shift_x", "f8"),
        ("shift_y", "f8"),
        ("lensed_shift_x", "f8"),
        ("lensed_shift_y", "f8"),
        ("z", "f8"),
        ("image_x", "f8"),
        ("image_y", "f8"),
        ("prelensed_image_x", "f8"),
        ("prelensed_image_y", "f8"),
        ("prelensed_ra", "f8"),
        ("prelensed_dec", "f8"),
        ("kappa", "f8"),
        ("gamma1", "f8"),
        ("gamma2", "f8"),]
    return np.zeros(1, dtype=dt)


def _rotate_pos(pos, theta):
    """Rotates coordinates by an angle theta

    Args:
        pos (PositionD):a galsim position
        theta (float):  rotation angle [rads]
    Returns:
        x2 (ndarray):   rotated coordiantes [x]
        y2 (ndarray):   rotated coordiantes [y]
    """
    x = pos.x
    y = pos.y
    cost = np.cos(theta)
    sint = np.sin(theta)
    x2 = cost * x - sint * y
    y2 = sint * x + cost * y
    return galsim.PositionD(x=x2, y=y2)
