import os
import logging
import copy
import numpy as np
import galsim
from .cache_tools import cached_header_read

LOGGER = logging.getLogger(__name__)


def gen_sip_wcs(
    *,
    rng,
    position_angle_range,
    dither_range,
    scale,
    scale_frac_std,
    shear_std,
    world_origin, origin,
):
    """Generate a random sim WCS

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
        The location of the world_origin in the image coordinate system.
        Note that the image origin is dithered if requested to keep the
        world origin fixed. Units are pixels.

    Returns
    -------
    wcs : galsim.TanWCS
        The randomly generated TanWCS object.
    """

    hdr = get_sip_example()

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
        high=position_angle_range[1],
    )

    # conversions to FITS convention
    theta = np.radians(90 - theta)
    g2 = -g2

    # in arcsec
    dither_u = rng.uniform(
        low=dither_range[0],
        high=dither_range[1],
    ) * scale
    dither_v = rng.uniform(
        low=dither_range[0],
        high=dither_range[1],
    ) * scale

    # rotation matrix
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    jac_matrix = scale * np.dot(
        galsim.Shear(g1=g1, g2=g2).getMatrix(),
        np.array([[costheta, -sintheta], [sintheta, costheta]])
    )

    hdr['CD1_1'] = jac_matrix[0, 0]/3600
    hdr['CD1_2'] = jac_matrix[0, 1]/3600
    hdr['CD2_1'] = jac_matrix[1, 0]/3600
    hdr['CD2_2'] = jac_matrix[1, 1]/3600

    hdr['CRVAL1'] = world_origin.ra.deg
    hdr['CRVAL2'] = world_origin.dec.deg
    hdr['CRPIX1'] = origin.x
    hdr['CRPIX2'] = origin.y

    wcs0 = galsim.GSFitsWCS(header=hdr)

    jac = wcs0.jacobian(world_pos=world_origin).getMatrix()
    dxdy = np.dot(np.linalg.inv(jac), np.array([dither_u, dither_v]))

    new_origin = origin + galsim.PositionD(x=dxdy[0], y=dxdy[1])

    hdr['CRPIX1'] = new_origin.x
    hdr['CRPIX2'] = new_origin.y

    wcs = galsim.GSFitsWCS(header=hdr)
    return wcs


def get_sip_example():
    fname = os.path.join(
        os.environ['CATSIM_DIR'],
        'example-sip-small.fits.gz',
    )

    hdr = cached_header_read(fname=fname, ext=0)
    return copy.deepcopy(hdr)
