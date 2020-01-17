import pytest

try:
    import lsst.afw.image as afw_image
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

from ..simple_sim import COSMIC_RAY, BAD_COLUMN


@pytest.mark.skipif(not HAVE_STACK, reason='the DM stack is not installed')
def test_lsst_mask_bits():
    mask = afw_image.Mask()
    cr_val = 2**mask.getMaskPlane('CR')
    assert cr_val == COSMIC_RAY

    mask = afw_image.Mask()
    bad_val = 2**mask.getMaskPlane('BAD')

    assert bad_val == BAD_COLUMN
