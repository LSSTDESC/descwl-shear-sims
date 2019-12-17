"""
A container for coadd observations.
"""
import galsim


class CoaddObs(object):
    """
    Simple container for coadding SEObs and retrieving data.  It is assumed
    that the psf is the same for all images, the wcs is desribed by a spatially
    constant jacobian

    Parameters
    ----------
    data: OrderedDict
        keyed by band, with each element a list of SEObs
    """
    def __init__(self, data):
        self._data = data
        self._make_coadd()

    @property
    def image(self):
        """
        Get a reference to the image array
        """
        return self._image

    @property
    def weight(self):
        """
        Get a reference to the weight image array
        """
        return self._weight

    @property
    def jacobian(self):
        """
        get a reference to the jacobian
        """
        return self._jacobian

    @property
    def psf(self):
        """
        getter for psf observation
        """
        return self._psf

    def _make_coadd(self):
        import ngmix

        ntot = 0
        wsum = 0.0
        for band_ind, band in enumerate(self._data):
            for epoch_ind, se_obs in enumerate(self._data[band]):

                x, y = 100, 100

                wt = se_obs.weight[0, 0]
                wsum += wt

                if ntot == 0:

                    image = se_obs.image.array.copy()
                    weight = se_obs.weight.array.copy()

                    psf_image = se_obs.get_psf(x, y).array
                    psf_weight = psf_image*0 + 1.0/1.0**2

                    pos = galsim.PositionD(x=x, y=y)
                    wjac = se_obs.wcs.jacobian(image_pos=pos)
                    wscale, wshear, wtheta, wflip = wjac.getDecomposition()
                    jac = ngmix.DiagonalJacobian(
                        x=x,
                        y=x,
                        scale=wscale,
                        dudx=wjac.dudx,
                        dudy=wjac.dudy,
                        dvdx=wjac.dvdx,
                        dvdy=wjac.dvdy,
                    )

                else:
                    image += se_obs.image.array[:, :]*wt
                    weight[:, :] += se_obs.weight.array[:, :]

                ntot += 1

        image *= 1.0/wsum

        self._psf = ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=jac,
        )
        self._image = image
        self._weight = weight
        self._jacobian = jac
