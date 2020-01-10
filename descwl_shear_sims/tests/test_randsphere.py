import numpy as np
import pytest

from ..randsphere import randsphere


def test_randsphere_smoke():
    ra, dec = randsphere(np.random.RandomState(seed=10), 100)
    assert len(ra) == 100
    assert len(dec) == 100
    assert np.all((ra >= 0) & (ra <= 360))
    assert np.all((dec >= -90) & (dec <= 90))


@pytest.mark.parametrize(
    'ra_range,dec_range',
    [[None, None],
     [(10, 20), None],
     [None, (-10, 20)],
     [(10, 20), (-10, 20)]])
def test_randsphere_ra_range_dec_range(ra_range, dec_range):
    _ra_range = ra_range or [0, 360]
    _dec_range = dec_range or [-90, 90]
    ra, dec = randsphere(
        np.random.RandomState(seed=10), 100, ra_range=_ra_range, dec_range=_dec_range)
    assert len(ra) == 100
    assert len(dec) == 100
    assert np.all((ra >= _ra_range[0]) & (ra <= _ra_range[1]))
    assert np.all((dec >= _dec_range[0]) & (dec <= _dec_range[1]))


@pytest.mark.parametrize('ra_range,dec_range', [
    (10, None),
    (None, 10),
    ((-10, 40), None),
    ((10, 400), None),
    (None, (-100, 0)),
    (None, (-80, 1000)),
])
def test_randsphere_raises(ra_range, dec_range):
    with pytest.raises(ValueError):
        ra, dec = randsphere(
            np.random.RandomState(seed=10), 100, ra_range=ra_range, dec_range=dec_range)
