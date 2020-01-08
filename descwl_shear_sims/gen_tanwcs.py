import logging
import numpy as np
import galsim

LOGGER = logging.getLogger(__name__)


def gen_tanwcs(
        *, rng, position_angle_range, dither_range,
        scale, scale_frac_std, shear_std,
        world_origin, origin):
    """Generate a random TanWCS.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use to generate the random WCS.
    position_angle_range : 2-tuple of floats
        The range of position angles to select from for rotating the image
        WCS coordinares. In degrees.
    dither_range : 2-tuple of floats
        The lowest and highest dither in uv coordinate pixels.
    scale : float
        The mean pixel scale of the image in arcseconds.
    scale_frac_std : float
        The fractional variance in the generated image pixel scale.
    shear_std : float
        The standard deviation of the Gaussian shear put into the WCS.
    world_origin : galsim.CelestialCoord
        The location of the origin of the uv coordinate system in the
        world coordinate system.
    origin : galsim.PositionD
        The location of the origin of the uv coordinate system in the
        image coordinate system. Note that the image origin is dithered if
        requested to keep the uv (and thus world) origin fixed. Units are
        pixels.

    Returns
    -------
    wcs : galsim.TanWCS
        The randomly generated TanWCS object.
    """
    # an se wcs is generated from
    # 1) a pixel scale
    # 2) a shear
    # 3) a rotation angle
    # 4) a dither in the u,v plane of the location of the
    #    the image origin
    g1 = rng.normal() * shear_std
    g2 = rng.normal() * shear_std
    scale = (1.0 + rng.normal() * scale_frac_std) * scale
    theta = rng.uniform(
        low=position_angle_range[0],
        high=position_angle_range[1]) / 180.0 * np.pi
    dither_ra = rng.uniform(
        low=dither_range[0],
        high=dither_range[1]) * scale / world_origin.dec.cos()
    dither_dec = rng.uniform(
        low=dither_range[0],
        high=dither_range[1]) * scale
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    jac_matrix = scale * np.dot(
        galsim.Shear(g1=g1, g2=g2).getMatrix(),
        np.array([[costheta, -sintheta], [sintheta, costheta]])
        )
    dudx = jac_matrix[0, 0]
    dudy = jac_matrix[0, 1]
    dvdx = jac_matrix[1, 0]
    dvdy = jac_matrix[1, 1]

    fid_wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            dudx, dudy, dvdx, dvdy,
            origin=origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )

    dxdy = fid_wcs.toImage(galsim.CelestialCoord(
        ra=world_origin.ra + dither_ra * galsim.arcsec,
        dec=world_origin.dec + dither_dec * galsim.arcsec))
    dxdy -= fid_wcs.toImage(world_origin)

    LOGGER.debug(
        'making WCS with g1|g2|scale|theta|du|dv: % f|% f|% f|% f|% f|% f',
        g1, g2, scale, theta, dither_ra, dither_dec)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            dudx, dudy, dvdx, dvdy,
            origin=origin + dxdy,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )
