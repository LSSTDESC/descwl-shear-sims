import time
import copy
import numpy as np
import ngmix
import tqdm
import joblib

import pytest

import metadetect.lsst_metadetect as lsst_metadetect
import descwl_shear_sims as sim
import descwl_coadd.coadd as coadd

CONFIG = {
    "model": "wmom",
    "bmask_flags": 0,
    "metacal": {
        "use_noise_image": True,
        "psf": "fitgauss",
    },
    "psf": {
        "model": "gauss",
        "lm_pars": {},
        "ntry": 2,
    },
    "weight": {
        "fwhm": 1.2,
    },
    "detect": {
        "thresh": 10.0,
    },
    'meds': {},
}


def _make_lsst_sim(*, seed, g1, g2, layout):
    rng = np.random.RandomState(seed=seed)

    galaxy_catalog = sim.galaxies.make_galaxy_catalog(
        rng=rng,
        coadd_dim=sim.sim.DEFAULT_SIM_CONFIG["coadd_dim"],
        buff=sim.sim.DEFAULT_SIM_CONFIG["buff"],
        layout=layout,
        gal_type='exp',
    )

    psf = sim.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = sim.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=sim.sim.DEFAULT_SIM_CONFIG["coadd_dim"],
        g1=g1,
        g2=g2,
        psf=psf,
    )
    return sim_data


def _shear_cuts(arr):
    msk = (
        (arr['flags'] == 0)
        & (arr['wmom_s2n'] > 10)
        & (arr['wmom_T_ratio'] > 1.2)
    )
    return msk


def _meas_shear_data(res):
    msk = _shear_cuts(res['noshear'])
    g1 = np.mean(res['noshear']['wmom_g'][msk, 0])
    g2 = np.mean(res['noshear']['wmom_g'][msk, 1])

    msk = _shear_cuts(res['1p'])
    g1_1p = np.mean(res['1p']['wmom_g'][msk, 0])
    msk = _shear_cuts(res['1m'])
    g1_1m = np.mean(res['1m']['wmom_g'][msk, 0])
    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res['2p'])
    g2_2p = np.mean(res['2p']['wmom_g'][msk, 1])
    msk = _shear_cuts(res['2m'])
    g2_2m = np.mean(res['2m']['wmom_g'][msk, 1])
    R22 = (g2_2p - g2_2m) / 0.02

    dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
    return np.array([(g1, g2, R11, R22)], dtype=dt)


def _bootstrap_stat(d1, d2, func, seed, nboot=500):
    dim = d1.shape[0]
    rng = np.random.RandomState(seed=seed)
    stats = []
    for _ in tqdm.trange(nboot, leave=False):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))
    return stats


def _meas_m_c_cancel(pres, mres):
    x = np.mean(pres['g1'] - mres['g1'])/2
    y = np.mean(pres['R11'] + mres['R11'])/2
    m = x/y/0.02 - 1

    x = np.mean(pres['g2'] + mres['g2'])/2
    y = np.mean(pres['R22'] + mres['R22'])/2
    c = x/y

    return m, c


def _boostrap_m_c(pres, mres):
    m, c = _meas_m_c_cancel(pres, mres)
    bdata = _bootstrap_stat(pres, mres, _meas_m_c_cancel, 14324, nboot=500)
    merr, cerr = np.std(bdata, axis=0)
    return m, merr, c, cerr


def _run_sim_one(*, seed, mdet_seed, g1, g2, **kwargs):
    sim_data = _make_lsst_sim(seed=seed, g1=g1, g2=g2, **kwargs)
    mbc = coadd.MultiBandCoaddsDM(
        data=sim_data['band_data'],
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        psf_dims=sim_data['psf_dims'],
        byband=False,
    )
    coadd_obs = mbc.coadds['all']
    coadd_mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs.append(obslist)

    md = lsst_metadetect.LSSTMetadetect(
        copy.deepcopy(CONFIG),
        coadd_mbobs,
        np.random.RandomState(seed=mdet_seed),
    )
    md.go()
    return md.result


def run_sim(seed, mdet_seed, **kwargs):
    # positive shear
    _pres = _run_sim_one(seed=seed, mdet_seed=mdet_seed, g1=0.02, g2=0, **kwargs)
    if _pres is None:
        return None

    # negative shear
    _mres = _run_sim_one(seed=seed, mdet_seed=mdet_seed, g1=-0.02, g2=0, **kwargs)
    if _mres is None:
        return None

    return _meas_shear_data(_pres), _meas_shear_data(_mres)


@pytest.mark.parametrize(
    'layout,ntrial', [('grid', 50), ('random', 2500)]
)
def test_shear_meas(layout, ntrial):
    nsub = max(ntrial // 100, 10)
    nitr = ntrial // nsub
    rng = np.random.RandomState(seed=116)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mdet_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("")

    pres = []
    mres = []
    loc = 0
    for itr in tqdm.trange(nitr):
        jobs = [
            joblib.delayed(run_sim)(
                seeds[loc+i], mdet_seeds[loc+i], layout=layout,
            )
            for i in range(nsub)
        ]
        outputs = joblib.Parallel(n_jobs=2, verbose=0, backend='loky')(jobs)

        for out in outputs:
            if out is None:
                continue
            pres.append(out[0])
            mres.append(out[1])
        loc += nsub

        m, merr, c, cerr = _boostrap_m_c(
            np.concatenate(pres),
            np.concatenate(mres),
        )
        print(
            (
                "\n"
                "nsims: %d\n"
                "m [1e-3, 3sigma]: %s +/- %s\n"
                "c [1e-5, 3sigma]: %s +/- %s\n"
                "\n"
            ) % (
                len(pres),
                m/1e-3,
                3*merr/1e-3,
                c/1e-5,
                3*cerr/1e-5,
            ),
            flush=True,
        )

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial, flush=True)

    pres = np.concatenate(pres)
    mres = np.concatenate(mres)
    m, merr, c, cerr = _boostrap_m_c(pres, mres)

    print(
        (
            "\n\nm [1e-3, 3sigma]: %s +/- %s"
            "\nc [1e-5, 3sigma]: %s +/- %s"
        ) % (
            m/1e-3,
            3*merr/1e-3,
            c/1e-5,
            3*cerr/1e-5,
        ),
        flush=True,
    )

    assert np.abs(m) < max(1e-3, 3*merr)
    assert np.abs(c) < 3*cerr
