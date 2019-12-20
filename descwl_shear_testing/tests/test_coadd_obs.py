import numpy as np
import galsim
import pytest

from ..se_obs import SEObs
from ..coadd_obs import CoaddObs

DIMS = (11, 13)


@pytest.fixture
def se_data():

    def psf_function(*, x, y):
        return galsim.ImageD(np.ones(DIMS) * 6)

    data = {
        'image': galsim.ImageD(np.ones(DIMS)),
        'weight': galsim.ImageD(np.ones(DIMS) * 2),
        'noise': galsim.ImageD(np.ones(DIMS) * 3),
        'bmask': galsim.ImageI(np.ones(DIMS) * 4),
        'ormask': galsim.ImageI(np.ones(DIMS) * 5),
        'wcs': galsim.PixelScale(0.2),
        'psf_function': psf_function,
    }
    return data


def test_coadd_obs_smoke(se_data):
    data = [SEObs(**se_data)]*3

    coadd_obs = CoaddObs(data)  # noqa
