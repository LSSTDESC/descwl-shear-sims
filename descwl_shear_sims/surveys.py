import numpy as np
import descwl
from .constants import (
    ZERO_POINT,
    SCALE,
)


def get_survey(*, gal_type, band):
    """
    Get a survey object

    Parameters
    ----------
    gal_type: string
        'wldeblend' or 'exp'
    band: string
        e.g. 'r'

    Returns
    -------
    For gal_type 'wldeblend' returns a WLDeblendSurvey object,
    for gal_type 'exp' returns BasicSurvey.
    """
    if gal_type == 'wldeblend':
        survey = WLDeblendSurvey(band=band)
    elif gal_type in ['exp']:
        survey = BasicSurvey(band=band)
    else:
        raise ValueError("bad gal_type: '%s'" % gal_type)

    return survey


def rescale_wldeblend_exp(*, survey, exp):
    """
    Rescale wldeblend images noise and weight to our zero point

    Parameters
    ----------
    survey: WLDeblendSurvey
        The survey object
    exp: ExposureF
        The exposure to rescale

    Returns
    -------
    None
    """
    fac = get_wldeblend_rescale_fac(survey)
    vfac = fac**2

    exp.image *= fac

    exp.variance *= vfac


def get_wldeblend_rescale_fac(survey):
    """
    Get the factor to rescale wldeblend images to our zero point

    Parameters
    ----------
    survey: WLDeblendSurvey
        The survey object

    Returns
    -------
    number by which to rescale images
    """
    s_zp = survey.zero_point
    s_et = survey.exposure_time
    return 10.0**(0.4*(ZERO_POINT - 24.0))/s_zp/s_et


class WLDeblendSurvey(object):
    """
    wrapper for wldeblend surveys

    Parameters
    ----------
    band: string
        The band, e.g. 'r'
    """
    def __init__(self, *, band):

        pars = descwl.survey.Survey.get_defaults(
            survey_name='LSST',
            filter_band=band,
        )

        pars['survey_name'] = 'LSST'
        pars['filter_band'] = band
        pars['pixel_scale'] = SCALE

        # note in the way we call the descwl package, the image width
        # and height is not actually used
        pars['image_width'] = 10
        pars['image_height'] = 10

        # some versions take in the PSF and will complain if it is not
        # given
        try:
            svy = descwl.survey.Survey(**pars)
        except Exception:
            pars['psf_model'] = None
            svy = descwl.survey.Survey(**pars)

        self.noise = np.sqrt(svy.mean_sky_level)

        self.descwl_survey = svy

    @property
    def filter_band(self):
        """
        get the filter band for this survey

        Returns
        -------
        string filter band, e.g. 'r'
        """
        return self.descwl_survey.filter_band

    def get_flux(self, mag):
        """
        convert mag to flux
        """
        return self.descwl_survey.get_flux(mag)


class BasicSurvey(object):
    """
    represent a simple survey with common interface

    Parameters
    ----------
    band: string
        e.g. 'r'
    """
    def __init__(self, *, band):
        self.band = band
        self.noise = 1.0
        self.filter_band = band

    def get_flux(self, mag):
        """
        get the flux for the input mag using the standard zero point
        """
        return 10**(0.4 * (ZERO_POINT - mag))
