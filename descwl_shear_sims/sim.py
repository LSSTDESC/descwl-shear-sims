import copy
import galsim
import numpy as np

import lsst.afw.image as afw_image
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

from .lsst_bits import get_flagval
from .saturation import saturate_image_and_mask, BAND_SAT_VALS
from .surveys import get_survey, rescale_wldeblend_exp
from .constants import SCALE
from .artifacts import add_bleed, get_max_mag_with_bleed
from .masking import get_bmask, calculate_and_add_bright_star_mask
from .objlists import get_objlist
from .psfs import make_dm_psf
from .wcs import make_wcs, make_dm_wcs, make_coadd_dm_wcs


DEFAULT_SIM_CONFIG = {
    "gal_type": "exp",
    "psf_type": "gauss",
    "psf_dim": 51,
    "coadd_dim": 351,
    "se_dim": None,
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
    draw_method='auto',
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
    draw_method: string
        Draw method for galsim objects, default 'auto'.  Set to
        'phot' to get poisson noise.  Note this is much slower.
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

        lists = get_objlist(
            galaxy_catalog=galaxy_catalog,
            survey=survey,
            star_catalog=star_catalog,
            noise=noise_per_epoch,
        )

        bdata_list = []
        for epoch in range(epochs_per_band):
            exp = make_exp(
                rng=rng,
                band=band,
                noise=noise_per_epoch,
                objlist=lists['objlist'],
                shifts=lists['shifts'],
                dim=se_dim,
                psf=psf,
                psf_dim=psf_dim,
                g1=g1, g2=g2,
                star_objlist=lists['star_objlist'],
                star_shifts=lists['star_shifts'],
                bright_objlist=lists['bright_objlist'],
                bright_shifts=lists['bright_shifts'],
                bright_mags=lists['bright_mags'],
                coadd_bbox_cen_gs_skypos=coadd_bbox_cen_gs_skypos,
                dither=dither,
                rotate=rotate,
                mask_threshold=mask_threshold,
                cosmic_rays=cosmic_rays,
                bad_columns=bad_columns,
                star_bleeds=star_bleeds,
                draw_method=draw_method,
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
            # if show:
            #     # TODO expose this as a keyword
            #     import lsst.afw.display as afw_display
            #     display = afw_display.getDisplay(backend='ds9')
            #     display.mtv(exp)
            #     display.scale('log', 'minmax')
            #     input('hit a key')

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
    g1,
    g2,
    star_objlist=None,
    star_shifts=None,
    bright_objlist=None,
    bright_shifts=None,
    bright_mags=None,
    coadd_bbox_cen_gs_skypos,
    dither=False,
    rotate=False,
    mask_threshold=None,
    cosmic_rays=False,
    bad_columns=False,
    star_bleeds=False,
    draw_method='auto',
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
        List of PositionD representing offsets
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
    star_objlist: list
        List of GSObj for stars
    star_shifts: array
        List of PositionD for stars representing offsets
    bright_objlist: list, optional
        List of GSObj for bright objects
    bright_shifts: array, optional
        List of PositionD for stars representing offsets
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
    draw_method: string
        Draw method for galsim objects, default 'auto'.  Set to
        'phot' to get poisson noise.  Note this is much slower.
    """

    shear = galsim.Shear(g1=g1, g2=g2)
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

    image = galsim.Image(dim, dim, wcs=se_wcs)

    _draw_objects(
        image,
        objlist, shifts, psf, draw_method,
        coadd_bbox_cen_gs_skypos,
        rng,
        shear=shear,
    )
    if star_objlist is not None:
        assert star_shifts is not None, 'send star_shifts with star_objlist'
        _draw_objects(
            image,
            star_objlist, star_shifts, psf, draw_method,
            coadd_bbox_cen_gs_skypos,
            rng,
        )

    image.array[:, :] += rng.normal(scale=noise, size=dims)

    bmask = get_bmask(
        image=image,
        rng=rng,
        cosmic_rays=cosmic_rays,
        bad_columns=bad_columns,
    )

    if bright_objlist is not None:
        _draw_bright_objects(
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


def _draw_objects(
    image, objlist, shifts, psf, draw_method,
    coadd_bbox_cen_gs_skypos,
    rng,
    shear=None,
):

    wcs = image.wcs
    kw = {}
    if draw_method == 'phot':
        kw['maxN'] = 1_000_000
        kw['rng'] = galsim.BaseDeviate(seed=rng.randint(low=0, high=2**30))

    for obj, shift in zip(objlist, shifts):

        if shear is not None:
            obj = obj.shear(shear)
            shift = shift.shear(shear)

        # Deproject from u,v onto sphere.  Then use wcs to get to image pos.
        world_pos = coadd_bbox_cen_gs_skypos.deproject(
            shift.x * galsim.arcsec,
            shift.y * galsim.arcsec,
        )

        image_pos = wcs.toImage(world_pos)
        local_wcs = wcs.local(image_pos=image_pos)

        convolved_object = get_convolved_object(obj, psf, image_pos)

        stamp = convolved_object.drawImage(
            center=image_pos, wcs=local_wcs, method=draw_method, **kw
        )

        b = stamp.bounds & image.bounds
        if b.isDefined():
            image[b] += stamp[b]


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
):
    # extra array needed to determine star mask accurately
    timage = image.copy()
    timage.setZero()

    grng = galsim.BaseDeviate(rng.randint(0, 2**30))

    assert shifts is not None
    assert mags is not None and len(mags) == len(shifts)
    assert mask_threshold is not None

    max_bleed_mag = get_max_mag_with_bleed(band=band)

    wcs = image.wcs
    for obj, shift, mag in zip(objlist, shifts, mags):

        obj = _set_star_gsparams(obj, mag, noise)

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
        n_photons = None if obj.flux < max_n_photons else max_n_photons

        stamp = convolved_object.drawImage(
            center=image_pos, wcs=local_wcs,
            method='phot',
            n_photons=n_photons,
            poisson_flux=True,
            maxN=1_000_000,  # shoot in batches this size
            rng=grng,
        )
        b = stamp.bounds & image.bounds
        if b.isDefined():
            image[b] += stamp[b]

            # use smooth version for radius calculation
            stamp_fft = convolved_object.drawImage(
                center=image_pos, wcs=local_wcs,
            )

            timage[b] += stamp_fft[b]

            calculate_and_add_bright_star_mask(
                image=timage.array,
                bmask=bmask.array,
                image_pos=image_pos,
                threshold=mask_threshold,
            )
            if star_bleeds and mag < max_bleed_mag:
                add_bleed(
                    image=image.array,
                    bmask=bmask.array,
                    pos=image_pos,
                    mag=mag,
                    band=band,
                )

            # reset for next object
            timage.setZero()


def _set_star_gsparams(obj, mag, noise):
    # do_thresh = do_acc = False
    do_thresh = do_acc = True
    if mag < 18:
        do_thresh = True
    if mag < 15:
        do_acc = True

    if do_thresh or do_acc:
        kw = {}
        if do_thresh:

            # this is designed to quantize the folding_threshold values,
            # so that there are fewer objects in the GalSim C++ cache.
            # With continuous values of folding_threshold, there would be
            # a moderately largish overhead for each object.

            folding_threshold = noise / obj.flux
            folding_threshold = np.exp(
                np.floor(np.log(folding_threshold))
            )

            kw['folding_threshold'] = min(folding_threshold, 0.005)

        if do_acc:
            kw['kvalue_accuracy'] = 1.0e-8
            kw['maxk_threshold'] = 1.0e-5

        gsp = galsim.GSParams(**kw)
        obj = obj.withGSParams(gsp)

    return obj


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
