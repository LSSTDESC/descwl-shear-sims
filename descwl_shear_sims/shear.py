import galsim
import numpy as np
# maximum kappa allowed
# values greater than it will be clipped
g_max = 0.6
"""
shear_obj = ShearConstant(cluster_obj, z_cl, ra_cl, dec_cl)
shear_obj = ShearRedshift(g1=0.02, g2=0.00)
shear_obj = ShearNFW(mode="0000", g_dist="g1")
"""


class ShearNFW(object):
    """
    Shear object from NFW halos

    Parameters
    ----------
    cluster_obj (object):   cluster object from clmm
    z_cl (float):           redshift of the cluster
    x_cl (float):           ra of the cluster [arcsec]
    y_cl (float):           dec of the cluster [arcsec]

    """
    def __init__(self, cluster_obj, z_cl, ra_cl=0., dec_cl=0.):
        self.z_cl = z_cl
        self.ra_cl = ra_cl
        self.dec_cl = dec_cl
        self.cobj = cluster_obj
        self.cosmo = cluster_obj.cosmo
        return

    def get_shear(self, redshift, shift):
        """
        A shear wrapper to return g1 and g2 with different shear type

        Parameters
        ----------
        redshift (float):           redshifts of galaxies
        shift (galsim.positionD):   Galsim positionD shift [arcsec]

        Returns
        ---------
        shear (galsim.Shear)        shear distortion on the galaxy
        """
        z_cl = self.z_cl
        if redshift > z_cl:
            from astropy.coordinates import SkyCoord
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
            gammat = self.cobj.eval_tangential_shear(r3d, z_cl, redshift)
            kappa = self.cobj.eval_convergence(r3d, z_cl, redshift)
            gamma1 = gammat * np.cos(2. * phi)
            gamma2 = -gammat * np.sin(2. * phi)
            g1 = gamma1 #/ (1-kappa)
            g2 = gamma2 #/ (1-kappa)
            # we are forcing g to be less than g_max
            g = np.sqrt(g1 ** 2. + g2 ** 2.)
            ratio = min(g_max / g, 1.0)
            # and rescale g1 and g2 if g > g_max
            g1 = g1 * ratio
            g2 = g2 * ratio
        else:
            g1 = 0.
            g2 = 0.
        shear = galsim.Shear(g1=g1, g2=g2)
        return shear


class ShearConstant(object):
    """
    Constant shear along every redshift slice
    Parameters
    ----------
    g1, g2:    Constant shear distortion
    """
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        self.shear = galsim.Shear(g1=self.g1, g2=self.g2)
        return

    def get_shear(self, redshift=None, shift=None):
        """
        Returns
        ---------
        shear (galsim.Shear)        shear distortion on the galaxy
        """
        return self.shear


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
        self.shear_list = self.determine_shear_list(mode)
        return

    def determine_shear_list(self, mode):
        shear_list = []
        return shear_list

    def get_bin(self, refshift):
        bin_num = 0
        return bin_num

    def get_shear(self, redshift, shift=None):
        # z_gal_bins = redshift // self.dz_bin
        gamma1, gamma2 = (None, None)
        # TODO: Finish implementing the z-dependent shear
        bin_number = self.get_bin(redshift)
        shear = galsim.Shear(g1=gamma1, g2=gamma2)
        return shear
