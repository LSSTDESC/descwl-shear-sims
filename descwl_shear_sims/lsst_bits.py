import numpy as np

# these are not official values, as far as I know
# the stack doesn't mark such things (ES 2020-02-18)
BRIGHT = np.int32(2**30)


def get_flagval(name):
    import lsst.afw.image as afw_image

    name = name.upper()
    if name == 'BRIGHT':
        return BRIGHT

    return afw_image.Mask.getPlaneBitMask(name)
