import lsst.afw.image as afw_image
from ..lsst_bits import get_flagval


def test_lsst_mask_bits():

    mim = afw_image.ExposureF(20, 20)
    names = ['CR', 'BAD', 'SAT']
    for name in names:
        val = afw_image.Mask.getPlaneBitMask(name)
        imval = mim.mask.getPlaneBitMask(name)

        assert val == imval
        assert val == get_flagval(name)
