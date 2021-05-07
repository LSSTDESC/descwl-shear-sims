"""
copy-paste from my (beckermr) personal code here
https://github.com/beckermr/metadetect-coadding-sims
"""
import numpy as np

from ..gen_masks import (
    generate_basic_mask,
    generate_bad_columns,
    generate_cosmic_rays,
)
from ..lsst_bits import get_flagval


def test_generate_basic_mask():
    n = 100
    edge_width = 5
    bmask = generate_basic_mask(shape=(n, n), edge_width=edge_width)

    expected_count = edge_width*n*4 - edge_width**2*4
    w = np.where(bmask == get_flagval('EDGE'))
    assert w[0].size == expected_count


def test_generate_cosmic_rays_smoke():
    rng = np.random.RandomState(seed=10)
    msk = generate_cosmic_rays(shape=(64, 64), rng=rng)
    assert np.any(msk)


def test_generate_cosmic_rays_seed():
    rng = np.random.RandomState(seed=10)
    msk1 = generate_cosmic_rays(shape=(64, 64), rng=rng)

    rng = np.random.RandomState(seed=10)
    msk2 = generate_cosmic_rays(shape=(64, 64), rng=rng)

    assert np.array_equal(msk1, msk2)


def test_generate_bad_columns_smoke():
    rng = np.random.RandomState(seed=10)
    msk = generate_bad_columns(shape=(64, 64), rng=rng)

    assert np.any(msk)


def test_generate_bad_columns_seed():
    rng = np.random.RandomState(seed=10)
    msk1 = generate_bad_columns(shape=(64, 64), rng=rng)

    rng = np.random.RandomState(seed=10)
    msk2 = generate_bad_columns(shape=(64, 64), rng=rng)

    assert np.array_equal(msk1, msk2)
