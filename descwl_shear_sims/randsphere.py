import numpy as np
import esutil as eu


def randcap(*,
            rng,
            nrand,
            ra,
            dec,
            radius,
            get_radius=False,
            dorot=False):
    """
    Generate random points in a sherical cap

    parameters
    ----------

    nrand:
        The number of random points
    ra,dec:
        The center of the cap in degrees.  The ra should be within [0,360) and
        dec from [-90,90]
    radius: float
        radius of the cap, same units as ra,dec
    get_radius: bool, optional
        if true, return radius of each point in radians
    dorot: bool
        If dorot is True, generate the points on the equator and rotate them to
        be centered at the desired location.  This is the default when the dec
        is within 0.1 degrees of the pole, to avoid calculation issues
    """

    # generate uniformly in r**2
    if dec >= 89.9 or dec <= -89.9:
        dorot = True

    if dorot:
        tra, tdec = 90.0, 0.0
        rand_ra, rand_dec, rand_r = randcap(
            rng=rng,
            nrand=nrand,
            ra=90.0,
            dec=0.0,
            radius=radius,
            get_radius=True,
        )
        rand_ra, rand_dec = eu.coords.rotate(
            0.0,
            dec-tdec,
            0.0,
            rand_ra,
            rand_dec,
        )
        rand_ra, rand_dec = eu.coords.rotate(
            ra-tra,
            0.0,
            0.0,
            rand_ra,
            rand_dec,
        )
    else:

        rand_r = rng.uniform(size=nrand)
        rand_r = np.sqrt(rand_r)*radius

        # put in degrees
        np.deg2rad(rand_r, rand_r)

        # generate position angle uniformly 0, 2*PI
        rand_posangle = rng.uniform(low=0, high=2*np.pi, size=nrand)

        theta = np.array(dec, dtype='f8', ndmin=1, copy=True)
        phi = np.array(ra, dtype='f8', ndmin=1, copy=True)
        theta += 90

        np.deg2rad(theta, theta)
        np.deg2rad(phi, phi)

        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        sinr = np.sin(rand_r)
        cosr = np.cos(rand_r)

        cospsi = np.cos(rand_posangle)
        costheta2 = costheta*cosr + sintheta*sinr*cospsi

        np.clip(costheta2, -1, 1, costheta2)

        # gives [0,pi)
        theta2 = np.arccos(costheta2)
        sintheta2 = np.sin(theta2)

        cos_dphi = (cosr - costheta*costheta2)/(sintheta*sintheta2)

        np.clip(cos_dphi, -1, 1, cos_dphi)
        dphi = np.arccos(cos_dphi)

        # note fancy usage of where
        phi2 = np.where(rand_posangle > np.pi, phi+dphi, phi-dphi)

        np.rad2deg(phi2, phi2)
        np.rad2deg(theta2, theta2)
        rand_ra = phi2
        rand_dec = theta2-90.0

        eu.coords.atbound(rand_ra, 0.0, 360.0)

    if get_radius:
        np.rad2deg(rand_r, rand_r)
        return rand_ra, rand_dec, rand_r
    else:
        return rand_ra, rand_dec


def randsphere(rng, num, ra_range=None, dec_range=None):
    """Generate random points on the sphere, possibly on a subset of it.

    Routine due to Erin Sheldon.

    Parameters
    ----------
    num: integer
        The number of randoms to generate
    ra_range: list, optional
        Should be within range [0,360].  Default [0,360]
    dec_range: list, optional
        Should be within range [-90,90].  Default [-90,90]

    Returns
    -------
    ra : array-like
        ra values for the random points
    dec : array-like
        dec values for the random points
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
            raise ValueError("range input does not have len() method")

        if rng[0] < allowed[0] or rng[1] > allowed[1]:
            raise ValueError("%s should be within %s" % (rng, allowed))
    return rng
