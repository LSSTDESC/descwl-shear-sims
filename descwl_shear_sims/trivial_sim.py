import galsim
import ngmix
import numpy as np
from .se_obs import SEObs
from .gen_tanwcs import gen_tanwcs


class TrivialSim(object):
    def __init__(self, *, rng, noise, g1, g2):
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

        Returns
        --------
        obs: ngmix.Observation
        """

        self.object_data = None
        dim = 351
        self.psf_dim = 51
        scale = 0.2
        dims = [dim]*2
        cen = (np.array(dims)-1)/2
        n_on_side = 6

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

        image = all_obj.drawImage(
            nx=dim,
            ny=dim,
            scale=scale,
        )
        weight = image.copy()
        weight.array[:, :] = 1.0/noise**2

        image.array[:, :] += rng.normal(scale=noise, size=dims)
        noise_image = image.copy()
        noise_image.array[:, :] = rng.normal(scale=noise, size=dims)

        self._psf = psf

        world_origin = galsim.CelestialCoord(
            ra=200 * galsim.degrees,
            dec=0 * galsim.degrees,
        )
        se_origin = galsim.PositionD(x=cen[1], y=cen[0])

        self._tan_wcs = gen_tanwcs(
            position_angle_range=(0, 0),
            dither_range=(0, 0),
            scale_frac_std=0,
            shear_std=0,
            scale=scale,
            world_origin=world_origin,
            origin=se_origin,
            rng=rng,
        )
        self.coadd_wcs = self._tan_wcs
        self.coadd_dim = dim

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
            wcs=self._tan_wcs,
            psf_function=self._psf_func,
            bmask=bmask,
        )

    def gen_sim(self):
        return {
            'i': [self._seobs],
        }

    def _psf_func(self, *, x, y, center_psf, get_offset=False):

        image_pos = galsim.PositionD(x=x, y=y)

        offset = galsim.PositionD(x=0.0, y=0.0)

        if not center_psf:
            print("ignoring request to not center psf")

        gsimage = self._psf.drawImage(
            nx=self.psf_dim,
            ny=self.psf_dim,
            offset=offset,
            wcs=self._tan_wcs.local(image_pos=image_pos),
        )
        if get_offset:
            return gsimage, offset
        else:
            return gsimage
