from ..lsst_bits import get_flagval
import lsst.afw.image as afw_image


def test_lsst_mask_bits():
    cr_val = afw_image.Mask.getPlaneBitMask('CR')
    assert cr_val == get_flagval('CR')

    bad_val = afw_image.Mask.getPlaneBitMask('BAD')
    assert bad_val == get_flagval('BAD')

    edge_val = afw_image.Mask.getPlaneBitMask('EDGE')
    assert edge_val == get_flagval('EDGE')

    sat_val = afw_image.Mask.getPlaneBitMask('SAT')
    assert sat_val == get_flagval('SAT')
