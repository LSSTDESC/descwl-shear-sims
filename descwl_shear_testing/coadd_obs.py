"""
A container for coadd observations.
"""
import numpy as np
import ngmix
import galsim


class CoaddObs(ngmix.Observation):
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

    def _make_coadd(self):
        import ngmix

        ntot = 0
        wsum = 0.0
        for epoch_ind, se_obs in enumerate(self._data):

            x, y = 100, 100

            wt = se_obs.weight[0, 0]
            wsum += wt

            if ntot == 0:

                image = se_obs.image.array.copy()*wt
                noise = se_obs.noise.array.copy()*wt

                weight = se_obs.weight.array.copy()

                psf_image = se_obs.get_psf(x, y).array
                psf_err = psf_image.max()*0.0001
                psf_weight = psf_image*0 + 1.0/psf_err**2
                psf_cen = (np.array(psf_image.shape)-1.0)/2.0

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

                psf_jac = ngmix.DiagonalJacobian(
                    x=psf_cen[1],
                    y=psf_cen[0],
                    scale=wscale,
                    dudx=wjac.dudx,
                    dudy=wjac.dudy,
                    dvdx=wjac.dvdx,
                    dvdy=wjac.dvdy,
                )

            else:
                image += se_obs.image.array[:, :]*wt
                noise += se_obs.noise.array[:, :]*wt
                weight[:, :] += se_obs.weight.array[:, :]

            ntot += 1

        image *= 1.0/wsum
        noise *= 1.0/wsum

        psf_obs = ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jac,
        )

        super().__init__(
            image=image,
            noise=noise,
            weight=weight,
            bmask=np.zeros(image.shape, dtype='i4'),
            jacobian=jac,
            psf=psf_obs,
        )
