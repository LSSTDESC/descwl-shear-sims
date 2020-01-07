"""Class for making a mulit-band, multi-epoch sim w/ galsim.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
and cleaned up a fair bit.
"""
import logging

import galsim
import numpy as np

from .simple_sim import SimpleSim
from .util import randsphere

LOGGER = logging.getLogger(__name__)


class TanWCSSim(SimpleSim):
    """
    Sim to use a TanWCS with offsets and rotations
    """
    def __init__(self, **args):
        super().__init__(**args)

        self._setup_se_ranges()

    def _setup_se_ranges(self):
        """
        set up position range for input single epoch images

        Currently the offsets are uniform in both directions, with excursions
        equal to half the coadd size
        """

        self.coadd_ra, self.coadd_dec = 10, 0
        cosdec = np.cos(np.radians(self.coadd_dec))

        offset = self.coadd_dim/2*self.scale/3600
        self.ra_range = [
            self.coadd_ra - offset*cosdec,
            self.coadd_ra + offset*cosdec,
        ]
        self.dec_range = [
            self.coadd_dec - offset,
            self.coadd_dec + offset,
        ]

    def _get_se_pos(self):
        """
        random position within ra, dec ranges
        """
        rav, decv = randsphere(
            self._rng,
            1,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
        )
        return galsim.CelestialCoord(
            rav[0]*galsim.degrees,
            decv[0]*galsim.degrees,
        )

    def _get_theta(self):
        """
        random rotation
        """
        return self._rng.uniform(low=0, high=np.pi*2)

    def _get_random_wcs(self):
        """
        get a TanWCS for random offsets and random rotation
        """
        world_pos = self._get_se_pos()
        theta = self._get_theta()

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s), (s, c)))
        vec = np.array([[self.scale, 0], [0, self.scale]])

        cd = np.dot(rot, vec)

        se_cen = (self.se_dim - 1) / 2
        se_origin = galsim.PositionD(se_cen, se_cen)

        affine = galsim.AffineTransform(
            cd[0, 0],
            cd[0, 1],
            cd[1, 0],
            cd[1, 1],
            origin=se_origin,
        )

        return galsim.TanWCS(affine, world_origin=world_pos)

    def _get_wcs_for_band(self, band):
        """
        get wcs for all epochs
        """
        wcs_list = []

        for i in range(self.epochs_per_band):
            wcs = self._get_random_wcs()
            wcs_list.append(wcs)

        return wcs_list
