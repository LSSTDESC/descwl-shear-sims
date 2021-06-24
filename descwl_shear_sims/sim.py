import copy
import galsim
import numpy as np

import lsst.afw.image as afw_image
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

from .lsst_bits import get_flagval
from .saturation import saturate_image_and_mask, BAND_SAT_VALS
from .surveys import get_survey, rescale_wldeblend_exp
from .constants import SCALE
from .artifacts import add_bleeds
from .masking import get_bmask, calculate_and_add_bright_star_mask
from .objlists import get_objlist, get_convolved_objects
from .psfs import make_dm_psf
from .wcs import make_wcs, make_dm_wcs, make_coadd_dm_wcs


DEFAULT_SIM_CONFIG = {
    "gal_type": "exp",
    "psf_type": "gauss",
    "psf_dim": 51,
    "coadd_dim": 351,
    "buff": 50,
    "layout": "grid",
    "dither": False,
    "rotate": False,
    "bands": ["i"],
    "epochs_per_band": 1,
    "noise_factor": 1.0,
    "stars": False,
    "star_bleeds": False,
    "cosmic_rays": False,
    "bad_columns": False,
}


def make_sim(
    *,
    rng,
    galaxy_catalog,
    coadd_dim,
    g1,
    g2,
    psf,
    se_dim=None,
    star_catalog=None,
    psf_dim=51,
    dither=False,
    rotate=False,
    bands=['i'],
    epochs_per_band=1,
    noise_factor=1.0,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
):
    """
    Make simulation data

    Parameters
    ----------
    rng: numpy.random.RandomState
        Numpy random state
    galaxy_catalog: catalog
        E.g. WLDeblendGalaxyCatalog or FixedGalaxyCatalog
    coadd_dim: int
        Default 351
    g1: float
        Shear g1 for galaxies
    g2: float
        Shear g2 for galaxies
    psf: GSObject or PowerSpectrumPSF
        The psf object or power spectrum psf
    se_dim: int, optional
        Force the single epoch images to have this dimension.  If not
        sent it is calculated to be large enough to encompass the coadd
        with rotations and small dithers.
    star_catalog: catalog
        e.g. StarCatalog
    psf_dim: int, optional
        Dimensions of psf image.  Default 51
    dither: bool, optional
        Whether to dither the images at the pixel level, default False
    rotate: bool, optional
        Whether to rotate the images randomly, default False
    bands: list, optional
        Default ['i']
    epochs_per_band: int, optional
        Number of epochs per band
    noise_factor: float, optional
        Factor by which to multiply the noise, default 1
    cosmic_rays: bool
        If True, add cosmic rays
    bad_columns: bool
        If True, add bad columns
    """

    coadd_wcs, coadd_bbox = make_coadd_dm_wcs(coadd_dim)
    coadd_bbox_cen_gs_skypos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
    )

    if se_dim is None:
        se_dim = get_se_dim(coadd_dim=coadd_dim)

    band_data = {}
    for band in bands:

        survey = get_survey(gal_type=galaxy_catalog.gal_type, band=band)
        noise_per_epoch = survey.noise*np.sqrt(epochs_per_band)*noise_factor

        # go down to coadd depth in this band, not dividing by sqrt(nbands)
        # but we could if we want to be more conservative
        mask_threshold = survey.noise*noise_factor

        objlist, shifts = galaxy_catalog.get_objlist(
            survey=survey,
            g1=g1,
            g2=g2,
        )
        objlist, shifts, bright_objlist, bright_shifts, bright_mags = get_objlist(
            galaxy_catalog=galaxy_catalog,
            survey=survey,
            g1=g1,
            g2=g2,
            star_catalog=star_catalog,
            noise=noise_per_epoch,
        )

        bdata_list = []
        for epoch in range(epochs_per_band):
            exp = make_exp(
                rng=rng,
                band=band,
                noise=noise_per_epoch,
                objlist=objlist,
                shifts=shifts,
                dim=se_dim,
                psf=psf,
                psf_dim=psf_dim,
                coadd_bbox_cen_gs_skypos=coadd_bbox_cen_gs_skypos,
                dither=dither,
                rotate=rotate,
                bright_objlist=bright_objlist,
                bright_shifts=bright_shifts,
                bright_mags=bright_mags,
                mask_threshold=mask_threshold,
                cosmic_rays=cosmic_rays,
                bad_columns=bad_columns,
                star_bleeds=star_bleeds,
            )
            if galaxy_catalog.gal_type == 'wldeblend':
                rescale_wldeblend_exp(
                    survey=survey.descwl_survey,
                    exp=exp,
                )

            # mark high pixels SAT and also set sat value in image for
            # any pixels already marked SAT
            saturate_image_and_mask(
                image=exp.image.array,
                bmask=exp.mask.array,
                sat_val=BAND_SAT_VALS[band],
                flagval=get_flagval('SAT'),
            )

            bdata_list.append(exp)

        band_data[band] = bdata_list

    return {
        'band_data': band_data,
        'coadd_wcs': coadd_wcs,
        'psf_dims': (psf_dim, )*2,
        'coadd_dims': (coadd_dim, )*2,
        'coadd_bbox': coadd_bbox,
    }


def make_exp(
    *,
    rng,
    band,
    noise,
    objlist,
    shifts,
    dim,
    psf,
    psf_dim,
    coadd_bbox_cen_gs_skypos,
    dither=False,
    rotate=False,
    bright_objlist=None,
    bright_shifts=None,
    bright_mags=None,
    mask_threshold=None,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
):
    """
    Make an SEObs

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
        Array with fields dx and dy, which are du, dv offsets
        in sky coords.
    dim: int
        Dimension of image
    psf: GSObject or PowerSpectrumPSF
        the psf
    psf_dim: int
        Dimensions of psf image that will be drawn when psf func is called
    coadd_bbox_cen_gs_skypos: galsim.CelestialCoord
        The sky position of the center (origin) of the coadd we
        will make, as a galsim object not stack object
    dither: bool
        If set to True, dither randomly by a pixel width
    rotate: bool
        If set to True, rotate the image randomly
    bright_objlist: list, optional
        List of bright objects to mask
    bright_shifts: array, optional
        Shifts for the bright objects
    bright_mags: array, optional
        Mags for the bright objects
    mask_threshold: float
        Bright masks are created such that the profile goes out
        to this threshold
    cosmic_rays: bool
        If True, put in cosmic rays
    bad_columns: bool
        If True, put in bad columns
    star_bleeds: bool
        If True, add bleed trails to stars
    """

    dims = [dim]*2
    cen = (np.array(dims)-1)/2

    se_origin = galsim.PositionD(x=cen[1], y=cen[0])
    if dither:
        dither_range = 0.5
        off = rng.uniform(low=-dither_range, high=dither_range, size=2)
        offset = galsim.PositionD(x=off[0], y=off[1])
        se_origin = se_origin + offset
    else:
        offset = None

    if rotate:
        theta = rng.uniform(low=0, high=2*np.pi)
    else:
        theta = None

    # galsim wcs
    se_wcs = make_wcs(
        scale=SCALE,
        theta=theta,
        image_origin=se_origin,
        world_origin=coadd_bbox_cen_gs_skypos,
    )

    convolved_objects, _ = get_convolved_objects(
        objlist=objlist,
        psf=psf,
        shifts=shifts,
        offset=offset,
        se_wcs=se_wcs,
        se_origin=se_origin,
    )

    objects = galsim.Add(convolved_objects)

    # everything gets shifted by the dither offset
    image = objects.drawImage(
        nx=dim,
        ny=dim,
        wcs=se_wcs,
        offset=offset,
    )

    image.array[:, :] += rng.normal(scale=noise, size=dims)

    bmask = get_bmask(
        image=image,
        rng=rng,
        cosmic_rays=cosmic_rays,
        bad_columns=bad_columns,
    )

    if bright_objlist is not None:
        timage = image.copy()

        assert bright_shifts is not None
        assert bright_mags is not None
        assert mask_threshold is not None

        bright_convolved_objects, _ = get_convolved_objects(
            objlist=bright_objlist,
            psf=psf,
            shifts=bright_shifts,
            offset=offset,
            se_wcs=se_wcs,
            se_origin=se_origin,
        )
        for i, obj in enumerate(bright_convolved_objects):

            # profiles can have detectably sharp edges if the
            # profile is very high s/n and we have not set the
            # thresholds right in the gs params.
            #
            # photon shooting reduces these sharp edges, reducing
            # sensitivity to such an error

            obj.drawImage(
                image=timage,
                offset=offset,
                method='phot',
                n_photons=10_000_000,
                maxN=1_000_000,
                rng=galsim.BaseDeviate(rng.randint(0, 2**30)),
            )

            image += timage

            calculate_and_add_bright_star_mask(
                image=timage.array,
                bmask=bmask.array,
                shift=bright_shifts[i],
                wcs=se_wcs,
                origin=se_origin,
                threshold=mask_threshold,
            )

    if star_bleeds and bright_objlist is not None:
        # add bleeds at bright star locations if they are saturated
        add_bleeds(
            image=image,
            origin=se_origin,
            bmask=bmask,
            shifts=bright_shifts,
            mags=bright_mags,
            band=band,
        )

    dm_wcs = make_dm_wcs(se_wcs)
    dm_psf = make_dm_psf(psf=psf, psf_dim=psf_dim, wcs=se_wcs)

    variance = image.copy()
    variance.array[:, :] = noise**2

    masked_image = afw_image.MaskedImageF(dim, dim)
    masked_image.image.array[:, :] = image.array
    masked_image.variance.array[:, :] = variance.array
    masked_image.mask.array[:, :] = bmask.array

    exp = afw_image.ExposureF(masked_image)

    filter_label = afw_image.FilterLabel(band=band, physical=band)
    exp.setFilterLabel(filter_label)

    exp.setPsf(dm_psf)

    exp.setWcs(dm_wcs)

    detector = DetectorWrapper().detector
    exp.setDetector(detector)

    return exp


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
    out_config = copy.deepcopy(DEFAULT_SIM_CONFIG)
    sub_configs = ['gal_config', 'star_config']

    if config is not None:
        for key in config:
            if key not in out_config and key not in sub_configs:
                raise ValueError("bad key for sim: '%s'" % key)
        out_config.update(config)
    return out_config


def get_se_dim(*, coadd_dim):
    """
    get se dim given coadd dim.  The se dim is padded out as
        int(np.ceil(coadd_dim * np.sqrt(2))) + 20

    Parameters
    ----------
    coadd_dim: int
        dimensions of coadd

    Returns
    -------
    integer dimensions of SE image
    """
    return int(np.ceil(coadd_dim * np.sqrt(2))) + 20


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

    bbox_cen_skypos = coadd_wcs.pixelToSky(
        coadd_bbox.getCenter()
    )

    return galsim.CelestialCoord(
        ra=float(bbox_cen_skypos.getRa()) * galsim.radians,
        dec=float(bbox_cen_skypos.getDec()) * galsim.radians,
    )
