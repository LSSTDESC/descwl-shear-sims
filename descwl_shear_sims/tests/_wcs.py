import numpy as np
import galsim

WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)
SCALE = 0.263


def make_sim_wcs(dim):

    dims = [dim]*2
    cen = (np.array(dims)-1)/2
    image_origin = galsim.PositionD(x=cen[1], y=cen[0])

    mat = np.array(
        [[SCALE, 0.0],
         [0.0, SCALE]],
    )

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=WORLD_ORIGIN,
        units=galsim.arcsec,
    )
