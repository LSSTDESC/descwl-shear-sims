import galsim
import numpy as np
# maximum kappa allowed
# values greater than it will be clipped
kappa_max = 0.8

def get_shear(
    *,
    shear_type="constant",
    z_gals=None,
    shift=None,
    **kwargs,
):
    """
    A shear wrapper to return g1 and g2 with different shear type

    Parameters
    ----------
    shear_type:     the constant shear or shear from NFW halos
    z_gals:         redshifts of galaxies
    shift:          Galsim positionD shift
    """
    if shear_type == "constant":
        shear_obj = ShearConstant(**kwargs)
    elif shear_type =="redshift":
        shear_obj = ShearRedshift(**kwargs)
    elif shear_type == "NFW":
        shear_obj = ShearNFW(**kwargs)
    else:
        raise ValueError("Do not support the shear type: %s" % shear_type)
    return shear_obj.get_shear(z_gals, shift)


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

    def get_shear(self, z_gals, shift):
        """
        A shear wrapper to return g1 and g2 with different shear type

        Parameters
        ----------
        z_gals (float):             redshifts of galaxies
        shift (galsim.positionD):  Galsim positionD shift [arcsec]

        Returns
        ---------
        shear (galsim.Shear)        shear distortion on the galaxy
        """
        from astropy.coordinates import SkyCoord
        z_cl = self.z_cl
        # Create the SkyCoord objects
        coord_cl = SkyCoord(self.ra_cl, self.dec_cl, unit="arcsec")
        coord_gals = SkyCoord(shift.x, shift.y, unit="arcsec")
        # Calculate the separation
        sep = coord_cl.separation(coord_gals).radian
        # What is the unit of r3d?
        r3d = self.cosmo.rad2mpc(sep, self.z_cl)
        # position angle
        phi = coord_cl.position_angle(coord_gals).radian

        # TODO: confirm whether the units is Mpc/h or Mpc?
        gammat = self.cobj.eval_tangential_shear(r3d, z_cl, z_gals)
        kappa0 = self.cobj.eval_convergence(r3d, z_cl, z_gals)
        # we are forcing kappa to be less than kappa_max
        # and scale gamma by the same ratio
        kappa = min(kappa0, kappa_max)
        ratio = kappa / kappa0
        gamma1 = gammat * np.cos(2. * phi) * ratio
        gamma2 = -gammat * np.sin(2. * phi) * ratio
        shear = galsim.Shear(g1=gamma1/(1-kappa), g2=gamma2/(1-kappa))
        return shear


class ShearConstant(object):
    """
    Constant shear along every redshift slice
    """
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        return

    def get_shear(self):
        shear = galsim.Shear(g1= self.g1, g2=self.g2)
        return shear


class ShearRedshift(object):
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

    def get_shear(self, z_gals=None, shift=None):
        if z_gals is None:
            assert self.mode == "0" * self.nz_bins
        #z_gal_bins = z_gals // self.dz_bin
        gamma1, gamma2 = (None, None)
        # TODO: Finish implementing the z-dependent shear
        shear = galsim.Shear(g1= gamma1, g2=gamma2)
        return shear
