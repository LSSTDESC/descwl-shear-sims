"""
copy-paste from my (beckermr) personal code here
https://github.com/beckermr/metadetect-coadding-sims
"""
import numpy as np
import galsim

from ..masking import get_bmask_and_set_image
from ..artifacts import (
    generate_bad_columns,
    generate_cosmic_rays,
)


def test_basic_mask():
    image = galsim.ImageD(np.zeros((100, 100)))
    bmask = get_bmask_and_set_image(
        image=image, rng=None, cosmic_rays=False, bad_columns=False,
    )

    assert np.all(bmask.array == 0)


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
