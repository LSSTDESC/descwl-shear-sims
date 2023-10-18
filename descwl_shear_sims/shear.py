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


def _ternary(n, n_bins):
    """CREDIT: https://stackoverflow.com/questions/34559663/\
        convert-decimal-to-ternarybase3-in-python"""
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums)).zfill(n_bins)


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
    Constant shear in the full exposure
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
    Constant shear in each redshift slice
    """
    def __init__(self, z_bounds, mode, g_dist="g1", shear_value=0.02):
        assert isinstance(mode, int), "mode must be an integer"
        nz_bins = len(z_bounds) - 1
        # nz_bins is the number of redshift bins
        # note that there are three options in each redshift bin
        # 0: g=0.00; 1: g=-0.02; 2: g=0.02
        # for example, number of redshift bins is 4, if mode = 7 which in
        # ternary is "0021" --- meaning that the shear is (0.0, 0.0, 0.02,
        # -0.02) in each bin.
        self.nz_bins = int(nz_bins)
        self.code = _ternary(int(mode), self.nz_bins)
        assert 0 <= int(mode) < 3 ** self.nz_bins, "mode code is too large"
        # maybe we need it to be more flexible in the future
        # but now we keep the linear spacing
        self.z_bounds = z_bounds
        self.g_dist = g_dist
        self.shear_list = self.determine_shear_list(self.code)
        self.shear_value = shear_value
        return

    def determine_shear_list(self, code):
        values = [0.00, -self.shear_value, self.shear_value]
        shear_list = [values[int(i)] for i in code]
        return shear_list

    def get_bin(self, redshift):
        bin_num = np.searchsorted(self.z_bounds, redshift, side="left") - 1
        return bin_num

    def get_shear(self, redshift, shift=None):
        bin_number = self.get_bin(redshift)
        shear = self.shear_list[bin_number]

        if self.g_dist == 'g1':
            gamma1, gamma2 = (shear, 0.)
        elif self.g_dist == 'g2':
            gamma1, gamma2 = (0., shear)
        else:
            raise ValueError("g_dist must be either 'g1' or 'g2'")

        shear = galsim.Shear(g1=gamma1, g2=gamma2)
        return shear
