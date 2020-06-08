import galsim
import ngmix
import numpy as np


def make_trivial_sim(*, rng, noise, g1, g2):
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
    psf_noise = 1.0e-6
    dim = 351
    scale = 0.2
    dims = [dim]*2
    cen = (np.array(dims)-1)/2
    n_on_side = 6

    spacing = dim/n_on_side

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
            obj = galsim.Convolve(obj, psf)
            objlist.append(obj)

    all_obj = galsim.Add(objlist)
    all_obj = all_obj.shear(g1=g1, g2=g2)

    psf_image = psf.drawImage(scale=scale).array
    psf_image += rng.normal(scale=psf_noise, size=psf_image.shape)
    psf_weight = psf_image.copy()
    psf_weight[:, :] = 1/psf_noise**2

    image = all_obj.drawImage(
        nx=dim,
        ny=dim,
        scale=scale,
    ).array
    weight = image.copy()
    weight[:, :] = 1.0/noise**2

    image += rng.gaussian(scale=noise, size=image.shape)
    noise += rng.gaussian(scale=noise, size=image.shape)

    psf_cen = (np.array(psf_image.shape)-1)/2
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0],
        col=psf_cen[1],
        scale=scale,
    )
    jacobian = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=scale,
    )

    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=psf_weight,
        jacobian=psf_jacobian,
    )
    obs = ngmix.Observation(
        image=image,
        weight=weight,
        noise=noise,
        psf=psf_obs,
        jacobian=jacobian,
        store_pixels=False,
    )
    return obs
