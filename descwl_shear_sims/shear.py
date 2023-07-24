import clmm
import galsim
import numpy as np
from astropy.coordinates import SkyCoord

# maximum kappa allowed
# values greater than it will be clipped
kappa_max = 0.6

# factor from arcsec to radians
arcsec2rad = 1./3600./180.*np.pi


def get_shear(
    *,
    shear_type="constant",
    z_gals=None,
    shifts=None,
    **kwargs,
    ):
    """
    A shear wrapper to return g1 and g2 with different shear type

    Parameters
    ----------
    cluster_obj:    cluster object
    """
    if shear_type == "constant":
        shear_obj = ShearConstant(**kwargs)
    elif shear_type == "NFW":
        shear_obj = ShearNFW(**kwargs)
    else:
        raise ValueError("Do not support the shear type: %s" % shear_type)
    return shear_obj.get_shear(z_gals, shifts)


class ShearNFW(object):
    """
    Shear object from NFW

    Parameters
    ----------
    cluster_obj (object):    cluster object from clmm
    z_cl (float):    redshift of the cluster
    x_cl (float):    ra of the cluster [arcsec]
    y_cl (float):    dec of the cluster [arcsec]

    """
    def __init__(self, cluster_obj, z_cl, ra_cl=0., dec_cl=0.):
        self.z_cl = z_cl
        self.ra_cl = ra_cl
        self.dec_cl = dec_cl
        self.cobj = cluster_obj
        self.cosmo = cluster_obj.cosmo
        return


    def get_shear(self, z_gals, shifts):

        z_cl = self.z_cl
        # Create the SkyCoord objects
        coord_cl = SkyCoord(self.ra_cl, self.dec_cl, unit="arcsec")
        coord_gals = SkyCoord(shifts["dx"], shifts["dy"], unit="arcsec")
        # Calculate the separation
        sep = coord_cl.separation(coord_gals).rads
        r3d = self.cosmo.rad2mpc(sep, self.z_cl)
        phi = coord_cl.position_angle(coord_gals).rads

        # TODO: confirm whether the units is Mpc/h or Mpc?

        DeltaSigma = self.cobj.eval_excess_surface_density(r3d, z_cl)
        gammat = self.cobj.eval_tangential_shear(r3d, z_cl, z_gals)
        kappa = self.cobj.eval_convergence(r3d, z_cl, z_gals)
        gamma1 = gammat * -np.cos(2. * phi)
        gamma2 = gammat * -np.sin(2. * phi)
        return gamma1/(1-kappa), gamma2/(1-kappa)


class ShearConstant(object):
    """
    Constant shear along every redshift slice
    """
    def __init__(self, mode="0000", g_dist="g1"):
        # note that there are three options in each redshift bin
        # 0: g=0.00; 1: g=-0.02; 2: g=0.02
        # "0000" means that we divide into 4 redshift bins, and every bin
        # is distorted by -0.02
        nz_bins = len(mode)
        self.nz_bins = nz_bins
        # number of possible modes
        self.n_modes = 3 ** nz_bins
        self.mode = mode
        self.z_bounds = np.linspace(0, 4, nz_bins+1)
        self.dz_bin = self.z_bounds[1]-self.z_bounds[0]
        self.g_dist = g_dist
        return

    def get_shear(self, z_gals=None, shifts=None):
        if z_gals is None:
            assert self.mode == "0" * self.nz_bins
        z_gal_bins = z_gals // self.dz_bin
        gamma1, gamma2 = (None, None)
        # TODO: Finish implementing the z-dependent shear
        return gamma1, gamma2
