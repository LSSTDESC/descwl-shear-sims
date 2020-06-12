import copy
import galsim
import numpy as np
from .se_obs import SEObs

SCALE = 0.2
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)


class TrivialSim(object):
    """
    simple sim with round galaxies on a grid, with dithers, rotations, possibly
    multiple epochs

    Parameters
    ----------
    rng: np.RandomState
        The input random state
    noise: float
        Noise level.  If there are multiple epochs, the noise per epoch is set
        to noise*sqrt(nepochs) to keep s/n constant
    g1: float
        Shear g1
    g2: float
        Shear g2
    dither: bool
        If True, dither the images by +/- 0.5 pixels, default False
    rotate: bool
        If True, rotate randomly, default False
    bands: sequence
        List of band names, default ['i']
    epochs_per_band: int
        Number of epochs per band, default 1
    """
    def __init__(self,
                 *, rng, noise, g1, g2,
                 dither=False, rotate=False,
                 bands=['i'], epochs_per_band=1):

        self.psf_dim = 51
        self.coadd_dim = 351
        self.object_data = None
        noise_per_epoch = noise*np.sqrt(epochs_per_band)

        self._sim_data = {}
        for band in bands:
            se_obslist = []
            for epoch in range(epochs_per_band):
                tim = TrivialImage(
                    rng=rng,
                    noise=noise_per_epoch,
                    g1=g1,
                    g2=g2,
                    coadd_dim=self.coadd_dim,
                    psf_dim=self.psf_dim,
                    dither=dither,
                    rotate=rotate,
                )
                se_obslist.append(tim.seobs)

            self._sim_data[band] = se_obslist

        self._set_coadd_wcs()

    def gen_sim(self):
        """
        Returns a dict, keyed by band, with values lists
        of SEObs.  Currently the band is always 'i' and the
        lists are length 1
        """
        return self._sim_data

    def _set_coadd_wcs(self):
        # the coadd will be placed on the undithered grid
        coadd_dims = [self.coadd_dim]*2
        coadd_cen = (np.array(coadd_dims)-1)/2
        coadd_origin = galsim.PositionD(x=coadd_cen[1], y=coadd_cen[0])
        self.coadd_wcs = make_wcs(
            scale=SCALE,
            image_origin=coadd_origin,
            world_origin=WORLD_ORIGIN,
        )


class TrivialImage(object):
    def __init__(self, *, rng, noise, g1, g2,
                 coadd_dim,
                 psf_dim,
                 dither=False, rotate=False):
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
        rotate: bool
            If set to True, rotate the image randomly
        """

        self.psf_dim = psf_dim

        self._psf = galsim.Gaussian(fwhm=0.8)

        se_dim = (
            int(np.ceil(coadd_dim * np.sqrt(2))) + 20
        )

        se_dims = [se_dim]*2
        cen = (np.array(se_dims)-1)/2
        n_on_side = 6

        se_origin = galsim.PositionD(x=cen[1], y=cen[0])
        if dither:
            dither_range = 0.5
            off = rng.uniform(low=-dither_range, high=dither_range, size=2)
            self._offset = galsim.PositionD(x=off[0], y=off[1])
            se_origin = se_origin + self._offset
        else:
            self._offset = None

        if rotate:
            theta = rng.uniform(low=0, high=2*np.pi)
        else:
            theta = None

        self.se_wcs = make_wcs(
            scale=SCALE,
            theta=theta,
            image_origin=se_origin,
            world_origin=WORLD_ORIGIN,
        )

        # we want to fit them into the coadd region
        spacing = coadd_dim/(n_on_side+1)*SCALE

        objlist = []

        # ix/iy are really on the sky
        grid = spacing*(np.arange(n_on_side) - (n_on_side-1)/2)
        for ix in range(n_on_side):
            for iy in range(n_on_side):
                dx = grid[ix] + SCALE*rng.uniform(low=-0.5, high=0.5)
                dy = grid[iy] + SCALE*rng.uniform(low=-0.5, high=0.5)

                obj = galsim.Exponential(
                    half_light_radius=0.5,
                ).shift(
                    dx=dx,
                    dy=dy,
                )
                objlist.append(obj)

        all_obj = galsim.Add(objlist)
        all_obj = all_obj.shear(g1=g1, g2=g2)
        all_obj = galsim.Convolve(all_obj, self._psf)

        # everything gets shifted by the dither offset
        image = all_obj.drawImage(
            nx=se_dim,
            ny=se_dim,
            wcs=self.se_wcs,
            offset=self._offset,
        )
        weight = image.copy()
        weight.array[:, :] = 1.0/noise**2

        image.array[:, :] += rng.normal(scale=noise, size=se_dims)
        noise_image = image.copy()
        noise_image.array[:, :] = rng.normal(scale=noise, size=se_dims)

        bmask = galsim.Image(
            np.zeros(se_dims, dtype='i4'),
            bounds=image.bounds,
            wcs=image.wcs,
            dtype=np.int32,
        )

        self.seobs = SEObs(
            image=image,
            noise=noise_image,
            weight=weight,
            wcs=self.se_wcs,
            psf_function=self._psf_func,
            bmask=bmask,
        )

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


def make_wcs(*, scale, image_origin, world_origin, theta=None):
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
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )
