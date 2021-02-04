import numpy as np


class ZSliceShear(object):
    """
    Class for storing a "z-slice"
    shear. This has one shear value
    for zmin<z<=zmax, and another
    otherwise

    Parameters
    ----------
    g: float
        shear value for objects with
    redshift zmin<z<=zmax
    gother: float
        shear value outside the interval
    zmin: float
        minimum redshift of interval
    zmax: float
        maximum redshift of interval
    """
    def __init__(self, g,
                 gother, zmin,
                 zmax):
        self.g = g
        self.zmin = zmin
        self.zmax = zmax
        self.gother = gother

    def __call__(self, z):
        """
        Return the shear for
        redshift z
        """
        return np.where((z > self.zmin) * (z <= self.zmax),
                        self.g, self.gother)
