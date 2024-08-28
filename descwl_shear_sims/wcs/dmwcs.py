import galsim
import coord
import lsst.geom as geom
from lsst.geom import Point2I, Extent2I, Point2D, Box2I
from lsst.afw.geom import makeSkyWcs
from lsst.daf.base import PropertyList
from .wcstools import make_wcs
from ..constants import SCALE, WORLD_ORIGIN


def make_dm_wcs(galsim_wcs):
    """
    convert galsim wcs to stack wcs

    Parameters
    ----------
    galsim_wcs: galsim WCS
        Should be TAN or TAN-SIP

    Returns
    -------
    DM Stack sky wcs
    """

    if galsim_wcs.wcs_type == 'TAN':
        crpix = galsim_wcs.crpix
        # DM uses 0 offset, galsim uses FITS 1 offset
        # we are converting from a galsim position to a dm position
        # so we need to subtract 1
        stack_crpix = Point2D(crpix[0]-1, crpix[1]-1)
        cd_matrix = galsim_wcs.cd

        crval = geom.SpherePoint(
            galsim_wcs.center.ra/coord.radians,
            galsim_wcs.center.dec/coord.radians,
            geom.radians,
        )
        stack_wcs = makeSkyWcs(
            crpix=stack_crpix,
            crval=crval,
            cdMatrix=cd_matrix,
        )
    elif galsim_wcs.wcs_type == 'TAN-SIP':

        # No currently supported
        # this works with the 1-offset assumption from galsim
        #
        # this is not used if the lower bounds are 1, but the extra keywords
        # GS_{X,Y}MIN are set which we will remove below

        fake_bounds = galsim.BoundsI(1, 10, 1, 10)
        hdr = {}
        galsim_wcs.writeToFitsHeader(hdr, fake_bounds)

        del hdr["GS_XMIN"]
        del hdr["GS_YMIN"]

        metadata = PropertyList()

        for key, value in hdr.items():
            metadata.set(key, value)

        stack_wcs = makeSkyWcs(metadata)
    else:
        raise RuntimeError(
            "Does not support galsim_wcs type: %s" % galsim_wcs.wcs_type
        )

    return stack_wcs


def make_coadd_dm_wcs(coadd_dim, pixel_scale=SCALE,
                      big_coadd_dim_padding=3000,
                      xoff=1000, yoff=450):
    """
    make a coadd wcs, using the default world origin.  Create
    a bbox within larger box

    Parameters
    ----------
    coadd_dim: int
        Pixel dimension of the coadd image
    pixel_scale: float
        pixel scale
    big_coadd_dim_padding: int
        padding for the larger coadd image. dim(big_coadd) = dim(coadd) + padding
    xoff: int
        x offset of the coadd image within the larger image
    yoff: int
        y offset of the coadd image within the larger image
    Returns
    --------
    (coadd dm wcs, coadd dm bbox), see make_dm_wcs for return wcs type
    """

    # make a larger coadd region
    big_coadd_dim = big_coadd_dim_padding + coadd_dim
    big_coadd_bbox = Box2I(Point2I(0), Extent2I(big_coadd_dim))

    # make this coadd a subset of larger coadd
    coadd_bbox = Box2I(Point2I(xoff, yoff), Extent2I(coadd_dim))

    # center the coadd wcs in the bigger coord system
    coadd_origin = big_coadd_bbox.getCenter()

    # DM uses 0 offset, galsim uses FITS 1 offset
    # we are converting from a dm position to a galsim position
    # so we need to add 1
    gs_coadd_origin = galsim.PositionD(
        x=coadd_origin.x + 1,
        y=coadd_origin.y + 1,
    )

    coadd_wcs = make_dm_wcs(
        make_wcs(
            scale=pixel_scale,
            image_origin=gs_coadd_origin,
            world_origin=WORLD_ORIGIN,
        )
    )
    return coadd_wcs, coadd_bbox


def make_coadd_dm_wcs_simple(coadd_dim, pixel_scale=SCALE):
    """
    make a coadd wcs without big coadd bbox padding, using the default world origin.

    Parameters
    ----------
    coadd_dim: int
        Pixel dimension of the coadd image
    pixel_scale: float
        pixel scale

    Returns
    --------
    (coadd dm wcs, coadd dm bbox), see make_dm_wcs for return wcs type
    """

    return make_coadd_dm_wcs(coadd_dim, pixel_scale=pixel_scale,
                             big_coadd_dim_padding=0,
                             xoff=0, yoff=0)
