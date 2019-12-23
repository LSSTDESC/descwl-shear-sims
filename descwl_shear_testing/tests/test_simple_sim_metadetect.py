import numpy as np
# import pytest
import ngmix

from ..simple_sim import Sim
from ..coadd_obs import CoaddObs
from ..metadetect import SimMetadetect


def test_simple_sim_metadetect_smoke():
    rng = np.random.RandomState(9123)
    sim = Sim(rng=rng)
    data = sim.gen_sim()

    # faking ngmix MultiBandObsList
    # note data is an OrderedDict
    coadd_mbobs = ngmix.MultiBandObsList(
        meta={'psf_fwhm': sim.psf_kws['fwhm']},
    )
    for band in data:
        coadd_obs = CoaddObs(data[band])
        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

    config = {
        'bmask_flags': 0,
        'metacal': {
            'use_noise_image': True,
            'psf': 'fitgauss',
        },
        'psf': {
            'model': 'gauss',
            'lm_pars': {},
            'ntry': 2,
        },
        'weight': {
            'fwhm': 1.2,
        },
        'meds': {},
    }

    md = SimMetadetect(config, coadd_mbobs, rng)
    md.go()

    res = md.result
    keys = list(res.keys())
    assert len(keys) == 5
    for k in ['noshear', '1p', '1m', '2p', '2m']:
        assert k in keys
