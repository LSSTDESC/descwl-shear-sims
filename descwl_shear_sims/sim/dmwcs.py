import coord
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
from lsst.daf.base import PropertyList


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
        stack_crpix = geom.Point2D(crpix[0], crpix[1])
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
        import galsim

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

    return stack_wcs
