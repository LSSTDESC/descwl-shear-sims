import numpy as np
from ..sim import ZSliceShear


def test_zslice_shear():
    # input parameters
    # for ZSliceShear
    g = 0.2
    gother = -0.2
    zmin = 0.3
    zmax = 0.7
    s = ZSliceShear(g, gother,
                    zmin, zmax)
    # Generate some redshifts and
    # assert that they get assigned
    # the correct shears
    zs = np.arange(0., 3., 300)
    shears = s(zs)

    # check the objects inside and outside
    # the slice got the right shear values
    in_zslice = (zs > zmin) * (zs <= zmax)
    assert np.allclose(shears[in_zslice], g)
    assert np.allclose(shears[~in_zslice], gother)
