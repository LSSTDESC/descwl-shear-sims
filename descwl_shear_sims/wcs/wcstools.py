import galsim
import numpy as np
from ..constants import SCALE, WORLD_ORIGIN


def make_wcs(*, scale, image_origin, world_origin, theta=None):
    """
    make and return a wcs object

    Parameters
    ----------
    scale: float
        Pixel scale
    image_origin: galsim.PositionD
        Image origin position
    world_origin: galsim.CelestialCoord
        Origin on the sky
    theta: float, optional
        Rotation angle in radians

    Returns
    -------
    A galsim wcs object, currently a TanWCS
    """
    mat = np.array(
        [[scale, 0.0],
         [0.0, scale]],
    )
    if theta is not None:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        rot = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]],
        )
        mat = np.dot(mat, rot)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def make_coadd_wcs(coadd_dim):
    """
    make a coadd wcs, using the default world origin

    Parameters
    ----------
    coadd_dim: int
        dimensions of the coadd

    Returns
    --------
    A galsim wcs, see make_wcs for return type
    """
    coadd_dims = [coadd_dim]*2
    coadd_cen = (np.array(coadd_dims)-1)/2
    coadd_origin = galsim.PositionD(x=coadd_cen[1], y=coadd_cen[0])
    return make_wcs(
        scale=SCALE,
        image_origin=coadd_origin,
        world_origin=WORLD_ORIGIN,
    )
