import pytest

try:
    import lsst.afw.image as afw_image
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

from ..simple_sim import (
    COSMIC_RAY,
    BAD_COLUMN,
    EDGE,
)


@pytest.mark.skipif(not HAVE_STACK, reason='the DM stack is not installed')
def test_lsst_mask_bits():
    cr_val = 2**afw_image.Mask.getMaskPlane('CR')
    assert cr_val == COSMIC_RAY

    bad_val = 2**afw_image.Mask.getMaskPlane('BAD')
    assert bad_val == BAD_COLUMN

    edge_val = 2**afw_image.Mask.getMaskPlane('EDGE')
    assert edge_val == EDGE
