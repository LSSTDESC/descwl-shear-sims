import copy
import galsim
import numpy as np
from .se_obs import SEObs


class TrivialSim(object):
    def __init__(self, *, rng, noise, g1, g2, dither=False):
        """
        make a grid sim with trivial pixel scale, fixed sized
        exponentials and gaussian psf

        Parameters
        ----------
        rng: numpy.RandomState
            The random number generator
        noise: float
            Gaussian noise level
        g1: float
            Shear g1
        g2: float
            Shear g2
        dither: bool
            If set to True, dither randomly by a pixel width
        """

        self.object_data = None
        dim = 351
        self.psf_dim = 51
        scale = 0.2
        dims = [dim]*2
        cen = (np.array(dims)-1)/2
        n_on_side = 6

        world_origin = galsim.CelestialCoord(
            ra=200 * galsim.degrees,
            dec=0 * galsim.degrees,
        )

        se_origin = galsim.PositionD(x=cen[1], y=cen[0])
        if dither:
            off = rng.uniform(low=-0.5, high=0.5, size=2)
            self._offset = galsim.PositionD(x=off[0], y=off[1])
            se_origin = se_origin + self._offset
        else:
            self._offset = None

        # the coadd will be placed on the undithered grid
        self.coadd_dim = dim
        coadd_origin = galsim.PositionD(x=cen[1], y=cen[0])

        spacing = dim/(n_on_side+1)

        objlist = []

        psf = galsim.Gaussian(fwhm=0.8)

        for ix in range(n_on_side):
            for iy in range(n_on_side):
                x = spacing + ix*spacing + rng.uniform(low=-0.5, high=0.5)
                y = spacing + iy*spacing + rng.uniform(low=-0.5, high=0.5)

                dx = scale*(x - cen[0])
                dy = scale*(y - cen[1])

                obj = galsim.Exponential(
                    half_light_radius=0.5,
                ).shift(
                    dx=dx,
                    dy=dy,
                )
                objlist.append(obj)

        all_obj = galsim.Add(objlist)
        all_obj = all_obj.shear(g1=g1, g2=g2)
        all_obj = galsim.Convolve(all_obj, psf)

        # everything gets shifted by the dither offset
        image = all_obj.drawImage(
            nx=dim,
            ny=dim,
            scale=scale,
            offset=self._offset,
        )
        weight = image.copy()
        weight.array[:, :] = 1.0/noise**2

        image.array[:, :] += rng.normal(scale=noise, size=dims)
        noise_image = image.copy()
        noise_image.array[:, :] = rng.normal(scale=noise, size=dims)

        self._psf = psf

        self.se_wcs = make_wcs(
            scale=scale,
            image_origin=se_origin,
            world_origin=world_origin,
        )
        self.coadd_wcs = make_wcs(
            scale=scale,
            image_origin=coadd_origin,
            world_origin=world_origin,
        )

        bmask = galsim.Image(
            np.zeros(dims, dtype='i4'),
            bounds=image.bounds,
            wcs=image.wcs,
            dtype=np.int32,
        )

        self._seobs = SEObs(
            image=image,
            noise=noise_image,
            weight=weight,
            wcs=self.se_wcs,
            psf_function=self._psf_func,
            bmask=bmask,
        )

    def gen_sim(self):
        """
        Returns a dict, keyed by band, with values lists
        of SEObs.  Currently the band is always 'i' and the
        lists are length 1
        """
        return {
            'i': [self._seobs],
        }

    def _psf_func(self, *, x, y, center_psf, get_offset=False):
        """
        center_psf is ignored
        """
        image_pos = galsim.PositionD(x=x, y=y)

        offset = copy.deepcopy(self._offset)

        if center_psf:
            print("ignoring request to center psf, using internal offset")

        gsimage = self._psf.drawImage(
            nx=self.psf_dim,
            ny=self.psf_dim,
            offset=offset,
            wcs=self.se_wcs.local(image_pos=image_pos),
        )
        if get_offset:
            return gsimage, offset
        else:
            return gsimage


def make_wcs(*, scale, image_origin, world_origin):
    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            scale, 0, 0, scale,
            origin=image_origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )
