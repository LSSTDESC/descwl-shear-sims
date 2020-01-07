import numpy as np


def randsphere(rng, num, ra_range=None, dec_range=None):
    """
    Generate random points on the sphere

    You can limit the range in ra and dec.  To generate on a spherical cap, see
    randcap()

    parameters
    ----------
    num: integer
        The number of randoms to generate
    ra_range: list, optional
        Should be within range [0,360].  Default [0,360]
    dec_range: list, optional
        Should be within range [-90,90].  Default [-90,90]

    output
    ------
    ra, dec

    examples
    --------
        ra, dec = randsphere(2000, ra_range=[10,35], dec_range=[-25,15])
    """

    ra_range = _check_range(ra_range, [0.0, 360.0])
    dec_range = _check_range(dec_range, [-90.0, 90.0])

    ra = rng.uniform(
        size=num,
        low=ra_range[0],
        high=ra_range[1],
    )

    cosdec_min = np.cos(np.radians(90.0+dec_range[0]))
    cosdec_max = np.cos(np.radians(90.0+dec_range[1]))
    v = rng.uniform(
        size=num,
        low=cosdec_min,
        high=cosdec_max,
    )

    np.clip(v, -1.0, 1.0, v)

    # Now this generates on [0,pi)
    dec = np.arccos(v)

    # convert to degrees
    np.degrees(dec, dec)

    # now in range [-90,90.0)
    dec -= 90.0

    return ra, dec


def _check_range(rng, allowed):
    if rng is None:
        rng = allowed
    else:
        if not hasattr(rng, '__len__'):
            raise ValueError("range object does not have len() method")

        if rng[0] < allowed[0] or rng[1] > allowed[1]:
            raise ValueError("lon_range should be within [%s,%s]" % allowed)
    return rng
