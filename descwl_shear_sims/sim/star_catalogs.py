import numpy as np
import galsim
from ..stars import load_sample_stars, sample_star_density, get_star_mag
from .constants import SCALE
from .shifts import get_shifts


class StarCatalog(object):
    """
    Star catalog with variable density

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int
        Buffer region with no objects, on all sides of image
    density: float, optional
        Optional density for catalog, if not sent the density is variable and
        drawn from the expected galactic density
    """
    def __init__(self, *, rng, coadd_dim, buff, density=None):
        self.rng = rng

        self._star_cat = load_sample_stars()

        if density is None:
            density_mean = sample_star_density(
                rng=self.rng,
                min_density=2,
                max_density=100,
            )
        else:
            density_mean = density

        # one square degree catalog, convert to arcmin
        area = ((coadd_dim - 2*buff)*SCALE/60)**2
        nobj_mean = area * density_mean
        nobj = rng.poisson(nobj_mean)

        self.density = nobj/area

        self.shifts = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout="random",
            nobj=nobj,
        )

        num = self.shifts.size
        self.indices = self.rng.randint(
            0,
            self._star_cat.size,
            size=num,
        )

    def get_objlist(self, *, survey, noise):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey or BasicSurvey
            The survey object
        noise: float
            The noise level, needed for setting gsparams

        Returns
        -------
        [galsim objects]
        """

        num = self.shifts.size

        band = survey.filter_band
        objlist = []
        shift_ind = []
        bright_objlist = []
        bright_shift_ind = []
        bright_mags = []
        for i in range(num):
            star, mag, isbright = self._get_star(survey, band, i, noise)
            if isbright:
                bright_objlist.append(star)
                bright_shift_ind.append(i)
                bright_mags.append(mag)
            else:
                objlist.append(star)
                shift_ind.append(i)

        # objlist = [
        #     self._get_star(survey, band, i, noise)
        #     for i in range(num)
        # ]

        shifts = self.shifts[shift_ind].copy()
        bright_shifts = self.shifts[bright_shift_ind].copy()
        return objlist, shifts, bright_objlist, bright_shifts, bright_mags

    def _get_star(self, survey, band, i, noise):
        """
        Parameters
        ----------
        survey: WLDeblendSurvey or BasicSurvey
            The survey object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object
        noise: float
            The noise level, needed for setting gsparams

        Returns
        -------
        galsim.GSObject
        """

        index = self.indices[i]
        dx = self.shifts['dx'][i]
        dy = self.shifts['dy'][i]

        mag = get_star_mag(stars=self._star_cat, index=index, band=band)
        flux = survey.get_flux(mag)

        gsparams, isbright = get_star_gsparams(mag, flux, noise)
        star = galsim.Gaussian(
            fwhm=1.0e-4,
            flux=flux,
            gsparams=gsparams,
        ).shift(
            dx=dx,
            dy=dy,
        )

        return star, mag, isbright


def get_star_gsparams(mag, flux, noise):
    """
    Get appropriate gsparams given flux and noise

    Parameters
    ----------
    mag: float
        mag of star
    flux: float
        flux of star
    noise: float
        noise of image

    Returns
    --------
    GSParams, isbright where isbright is true for stars with mag less than 18
    """
    do_thresh = do_acc = False
    if mag < 18:
        do_thresh = True
    if mag < 15:
        do_acc = True

    if do_thresh or do_acc:
        isbright = True

        kw = {}
        if do_thresh:
            folding_threshold = noise/flux
            folding_threshold = np.exp(
                np.floor(np.log(folding_threshold))
            )
            kw['folding_threshold'] = min(folding_threshold, 0.005)

        if do_acc:
            kw['kvalue_accuracy'] = 1.0e-8
            kw['maxk_threshold'] = 1.0e-5

        gsparams = galsim.GSParams(**kw)
    else:
        gsparams = None
        isbright = False

    return gsparams, isbright
