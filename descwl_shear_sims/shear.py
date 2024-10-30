import galsim
import numpy as np


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
        """This function returns the galsim shear object
        Parameters
        ---------
        redshift (float):           redshift of the galaxy
        shift (galsim.PositionD):   position of the galaxy

        Returns
        ---------
        shear (galsim.Shear):       shear distortion on the galaxy
        """
        return self.shear

    def distort_galaxy(self, gso, shift, redshift):
        """This function distorts the galaxy's shape and position
        Parameters
        ---------
        gso (galsim object):        galsim galaxy
        shift (galsim.PositionD):   position of the galaxy
        redshift (float):           redshift of galaxy

        Returns
        ---------
        gso, shift:
            distorted galaxy object and shift
        """
        shear = self.get_shear(redshift, shift)
        gso = gso.shear(shear)
        shift = shift.shear(shear)
        return gso, shift


class ShearRedshift(ShearConstant):
    """
    Constant shear in each redshift slice
    """
    def __init__(self, z_bounds, mode, g_dist="g1", shear_value=0.02):
        assert isinstance(mode, int), "mode must be an integer"
        self.nz_bins = int(len(z_bounds) - 1)
        # nz_bins is the number of redshift bins
        # note that there are three options in each redshift bin
        # 0: g=-0.02; 1: g=0.02; 2: g=0.00
        # for example, number of redshift bins is 4, (nz_bins = [0., 0.5, 1.0,
        # 1.5, 2.0]) if mode = 7 which in ternary is "0021" --- meaning that
        # the shear is (-0.02, -0.02, 0.00, 0.02) in each bin, respectively.
        self.code = _ternary(int(mode), self.nz_bins)
        assert 0 <= int(mode) < 3 ** self.nz_bins, "mode code is too large"
        # maybe we need it to be more flexible in the future
        # but now we keep the linear spacing
        self.z_bounds = z_bounds
        self.g_dist = g_dist
        self.shear_value = shear_value
        self.shear_list = self.determine_shear_list(self.code)
        return

    def determine_shear_list(self, code):
        values = [-self.shear_value, self.shear_value, 0.0]
        shear_list = [values[int(i)] for i in code]
        return shear_list

    def _get_zshear(self, redshift):
        bin_num = np.searchsorted(self.z_bounds, redshift, side="left") - 1
        nz = len(self.z_bounds) - 1
        if bin_num < nz and bin_num >= 0:
            # if the redshift is within the boundaries of lower and uper limits
            # we add shear
            shear = self.shear_list[bin_num]
        else:
            # if not, we set shear to 0 and leave the galaxy image undistorted
            shear = 0.0
        return shear

    def get_shear(self, redshift, shift=None):
        shear = self._get_zshear(redshift)
        if self.g_dist == 'g1':
            gamma1, gamma2 = (shear, 0.)
        elif self.g_dist == 'g2':
            gamma1, gamma2 = (0., shear)
        else:
            raise ValueError("g_dist must be either 'g1' or 'g2'")

        shear_obj = galsim.Shear(g1=gamma1, g2=gamma2)
        return shear_obj
