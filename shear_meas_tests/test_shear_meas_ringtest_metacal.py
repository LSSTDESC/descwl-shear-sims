import ngmix
import pytest
import numpy as np

from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf  # for making a power spectrum PSF
from descwl_shear_sims.constants import SCALE
from descwl_shear_sims.shear import ShearConstant

shear_obj = ShearConstant(g1=0.02, g2=0.)

rng0 = np.random.RandomState(1024)
# We will measure moments with a fixed gaussian weight function
weight_fwhm = 1.2
fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

# these "runners" run the measurement code on observations
psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
runner = ngmix.runners.Runner(fitter=fitter)

boot = ngmix.metacal.MetacalBootstrapper(
    runner=runner,
    psf_runner=psf_runner,
    rng=rng0,
    psf="gauss",
    types=["noshear", "1p", "1m"],
)


def make_desc_sim(ran_seed, psf):
    """Makes 4 desc images using descwl_shear_sims, each of the image is a
    rotated version of the same intrinsic galaxy (then sheared and smeared).

    Parameters:
        ran_seed (int):     random seed
        psf (galsim.PSF):   Galsim PSF
    """

    rng = np.random.RandomState(ran_seed)
    args = {
        "rotate": False,
        "dither": False,
        "cosmic_rays": False,
        "bad_columns": False,
        "star_bleeds": False,
        "star_catalog": None,
    }
    band_list = ["i"]
    # make simulations, the galaxies have 0~1.5 pixel offsets
    # (53-25*2) / 2. = 1.5
    coadd_dim = 53
    buff = 25

    # galaxy catalog; you can make your own
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="fixed",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="random_disk",
    )

    nrot = 4
    rot_list = [np.pi / nrot * i for i in range(nrot)]
    img_array_list = []
    for theta0 in rot_list:
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            shear_obj=shear_obj,
            psf=psf,
            bands=band_list,
            noise_factor=0.0,
            theta0=theta0,
            **args
        )
        img_array_list.append(
            sim_data["band_data"][band_list[0]][0].getImage().getArray()
        )
    return img_array_list


def make_ngmix_obs(gal_array, psf_array):
    """Transforms to Ngmix data

    Parameters:
        gal_array (ndarray):    galaxy array
        psf_array (ndarray):    psf array
    """

    cen = (np.array(gal_array.shape) - 1.0) / 2.0
    psf_cen = (np.array(psf_array.shape) - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=SCALE,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0],
        col=psf_cen[1],
        scale=SCALE,
    )

    gal_noise = 1e-10
    psf_noise = 1e-10
    wt = np.ones_like(gal_array) / gal_noise**2.0
    psf_wt = np.ones_like(psf_array) / psf_noise**2.0

    psf_obs = ngmix.Observation(
        psf_array,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )
    obs = ngmix.Observation(
        gal_array,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )
    return obs


def select(data, shear_type):
    """Selects the data by shear type and size
    Parameters
    ----------
    data: array
        The array with fields shear_type and T
    shear_type: str
        e.g. 'noshear', '1p', etc.
    Returns
    -------
    array of indices
    """

    (w,) = np.where((data["flags"] == 0) & (data["shear_type"] == shear_type))
    return w


def make_struct(res, obs, shear_type):
    """Makes the data structure

    Parameters:
    res: dict
        With keys 's2n', 'e', and 'T'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type
    Returns:
        1-element array with fields
    """
    dt = [
        ("flags", "i4"),
        ("shear_type", "U7"),
        ("s2n", "f8"),
        ("g", "f8", 2),
        ("T", "f8"),
        ("Tpsf", "f8"),
    ]
    data = np.zeros(1, dtype=dt)
    data["shear_type"] = shear_type
    data["flags"] = res["flags"]
    if res["flags"] == 0:
        # data['s2n'] = res['s2n']
        data["s2n"] = res["s2n"]
        # for moments we are actually measureing e, the elliptity
        data["g"] = res["e"]
        data["T"] = res["T"]
    else:
        data["s2n"] = np.nan
        data["g"] = np.nan
        data["T"] = np.nan
        data["Tpsf"] = np.nan

        # we only have one epoch and band, so we can get the psf T from the
        # observation rather than averaging over epochs/bands
        data["Tpsf"] = obs.psf.meta["result"]["T"]
    return data


@pytest.mark.parametrize("ran_seed", [0, 1, 2, 3])
def test_sim_center(ran_seed):
    """Tests to 4 x 45 degree rotations cancel the shape noise in shear
    estimation
    """

    psf = make_fixed_psf(psf_type="moffat")
    gal_array_list = make_desc_sim(ran_seed, psf)
    psf_array = psf.drawImage(scale=SCALE).array
    outputs = []
    for gal_array in gal_array_list:
        obs = make_ngmix_obs(gal_array, psf_array)
        resdict, obsdict = boot.go(obs)
        for stype, sres in resdict.items():
            st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
            outputs.append(st)
        del obs, resdict, obsdict

    outputs = np.hstack(outputs)

    w = select(data=outputs, shear_type="noshear")
    w_1p = select(data=outputs, shear_type="1p")
    w_1m = select(data=outputs, shear_type="1m")
    g_1p = outputs["g"][w_1p, 0].mean()
    g_1m = outputs["g"][w_1m, 0].mean()
    R11 = (g_1p - g_1m) / 0.02

    g = outputs["g"][w].mean(axis=0)
    shear = g / R11

    # assert the difference between the measured shear and the input shear is
    # less than 1e-5
    np.testing.assert_almost_equal(shear[0] - 0.02, 0.0, 5)
    np.testing.assert_almost_equal(shear[1], 0.0, 5)

    return


if __name__ == "__main__":
    test_sim_center(0)
