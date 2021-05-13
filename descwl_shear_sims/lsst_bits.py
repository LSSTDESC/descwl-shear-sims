import lsst.afw.image as afw_image
from lsst.pex.exceptions import InvalidParameterError

try:
    afw_image.Mask.getPlaneBitMask('BRIGHT')
except InvalidParameterError:
    afw_image.Mask.addMaskPlane('BRIGHT')


def get_flagval(name):
    return afw_image.Mask.getPlaneBitMask(name)
