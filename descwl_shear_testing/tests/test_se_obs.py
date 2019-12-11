import numpy as np
import galsim
import pytest

from ..se_obs import SEObs

DIMS = (11, 13)


@pytest.fixture
def se_data():

    def psf_function(x, y):
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


def test_se_obs_smoke(se_data):
    obs = SEObs(**se_data)
    assert np.array_equal(obs.image.array, np.ones(DIMS))
    assert np.array_equal(obs.weight.array, np.ones(DIMS) * 2)
    assert np.array_equal(obs.noise.array, np.ones(DIMS) * 3)
    assert np.array_equal(obs.bmask.array, np.ones(DIMS) * 4)
    assert np.array_equal(obs.ormask.array, np.ones(DIMS) * 5)
    assert np.array_equal(obs.psf(1, 0).array, np.ones(DIMS) * 6)
    assert obs.wcs == galsim.PixelScale(0.2)


@pytest.mark.parametrize('attr,val', [
    ('image', galsim.ImageD(np.ones(DIMS) * 56)),
    ('weight', galsim.ImageD(np.ones(DIMS) * 56)),
    ('noise', galsim.ImageD(np.ones(DIMS) * 56)),
    ('bmask', galsim.ImageI(np.ones(DIMS) * 56)),
    ('ormask', galsim.ImageI(np.ones(DIMS) * 56)),
    ('wcs', galsim.PixelScale(0.5)),
])
def test_se_obs_set(attr, val, se_data):
    obs = SEObs(**se_data)
    setattr(obs, attr, val)
    if attr != 'wcs':
        assert np.array_equal(getattr(obs, attr).array, val.array)
    else:
        assert getattr(obs, attr) == val
    # the "or" statement catches the case when the attribute has been changed
    assert attr == 'image' or np.array_equal(obs.image.array, np.ones(DIMS))
    assert attr == 'weight' or np.array_equal(obs.weight.array, np.ones(DIMS) * 2)
    assert attr == 'noise' or np.array_equal(obs.noise.array, np.ones(DIMS) * 3)
    assert attr == 'bmask' or np.array_equal(obs.bmask.array, np.ones(DIMS) * 4)
    assert attr == 'ormask' or np.array_equal(obs.ormask.array, np.ones(DIMS) * 5)
    assert np.array_equal(obs.psf(1, 0).array, np.ones(DIMS) * 6)
    assert attr == 'wcs' or obs.wcs == galsim.PixelScale(0.2)


def test_se_obs_psf_call():

    def psf_function(x, y):
        assert x == 10
        assert y == 5
        return 11

    obs = SEObs(
        image=galsim.ImageD(np.ones(DIMS)),
        weight=galsim.ImageD(np.ones(DIMS)),
        wcs=galsim.PixelScale(0.2),
        psf_function=psf_function,
    )

    assert obs.psf(10, 5) == 11


@pytest.mark.parametrize('attr', ['image', 'weight', 'noise', 'bmask', 'ormask'])
def test_se_obs_raises(attr, se_data):
    with pytest.raises(ValueError) as e:
        obs = SEObs(**se_data)
        setattr(obs, attr, 10)
    if attr == 'bmask':
        assert "bit mask" in str(e)
    elif attr == 'ormask':
        assert "\"or\" mask" in str(e)
    else:
        assert attr in str(e)
    assert isinstance(getattr(obs, attr), galsim.Image)


@pytest.mark.parametrize('attr', ['wcs', 'image', 'weight', 'noise', 'bmask', 'ormask'])
def test_se_obs_init_raises(attr, se_data):
    idata = {}
    idata.update(se_data)
    idata[attr] = 10
    with pytest.raises(ValueError) as e:
        SEObs(**idata)
    if attr == 'bmask':
        assert "bit mask" in str(e)
    elif attr == 'ormask':
        assert "\"or\" mask" in str(e)
    elif attr == 'wcs':
        assert "WCS" in str(e)
    else:
        assert attr in str(e)
