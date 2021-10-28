import lsst.afw.image as afw_image


def get_flagval(name):
    return afw_image.Mask.getPlaneBitMask(name)
