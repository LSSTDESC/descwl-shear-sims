import numpy as np


def generate_bad_columns(
        *, shape, mean_bad_cols=1,
        gap_prob=0.30,
        min_gap_frac=0.1,
        max_gap_frac=0.3,
        rng=None,
):
    """
    Generate a binary mask w/ bad columns.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the mask to generate.
    mean_bad_cols : float, optional
        The mean of the Poisson distribution for the total number of
        bad columns to generate.
    gap_prob : float
        The probability that the bad column has a gap in it.
    min_gap_frac : float
        The minimum fraction of the image that the gap spans.
    max_gap_frac : floatn
        The maximum fraction of the image that the gap spans.
    rng : np.random.RandomState or None, optional
        An RNG to use. If none is provided, a new `np.random.RandomState`
        state instance will be created.

    Returns
    -------
    msk : np.ndarray, shape `shape`
        A boolean mask marking the locations of the bad columns.
    """

    msk = np.zeros(shape)
    rng = rng or np.random.RandomState()
    n_bad_cols = rng.poisson(mean_bad_cols)

    for _ in range(n_bad_cols):

        x = rng.choice(msk.shape[1])

        # set the mask first
        msk[:, x] = 1

        # add gaps
        if rng.uniform() < gap_prob:
            gfrac = rng.uniform(min_gap_frac, max_gap_frac)
            glength = int(msk.shape[0] * gfrac)
            gloc = rng.randint(0, msk.shape[0] - glength)
            msk[gloc:gloc+glength, x] = 0

    return msk.astype(bool)
