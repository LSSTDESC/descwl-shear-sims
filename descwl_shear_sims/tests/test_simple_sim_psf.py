import pytest
import numpy as np
import galsim

from ..simple_sim import Sim


def test_simple_sim_psf_smoke():
    sim = Sim(rng=10, gals_kws={'density': 10})
    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            # make sure this call works
            data[band][epoch].get_psf(10, 3)


@pytest.mark.parametrize('psf_type', ['gauss', 'moffat'])
def test_simple_sim_psf_type(psf_type):
    sim = Sim(
        rng=10,
        psf_type=psf_type,
        gals_kws={'density': 10},
    )
    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            psf_func, _ = sim._get_psf_funcs_for_band_epoch(band, epoch)
            psf = psf_func(x=10, y=3)

            psf_str = str(psf)

            if psf_type == 'gauss':
                assert 'Gaussian' in psf_str
            else:
                assert 'Moffat' in psf_str


def test_simple_sim_psf_center():
    sim = Sim(rng=10, gals_kws={'density': 10})
    data = sim.gen_sim()
    se_obs = data[sim.bands[0]][0]

    psf1 = se_obs.get_psf(10, 3, center_psf=False)
    psf2 = se_obs.get_psf(10, 3, center_psf=True)
    assert np.array_equal(psf1.array, psf2.array)
    assert np.allclose(psf1.array, psf1.array.T)
    assert np.allclose(psf2.array, psf2.array.T)

    psf1nc = se_obs.get_psf(10.3, 3.25, center_psf=False)
    psf2nc = se_obs.get_psf(10.3, 3.25, center_psf=True)
    assert not np.array_equal(psf1nc.array, psf2nc.array)
    assert not np.allclose(psf1nc.array, psf1nc.array.T)
    assert np.allclose(psf2nc.array, psf2nc.array.T)
    assert np.allclose(psf2nc.array, psf1.array)

    x = 10.2
    y = 3.75
    _, offset = se_obs.get_psf(x, y, center_psf=False, get_offset=True)
    assert offset.x == x - int(x+0.5)
    assert offset.y == y - int(y+0.5)

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        axs[0].imshow(psf1nc.array)
        axs[0].set_title('not centered')

        axs[1].imshow(psf2nc.array)
        axs[1].set_title('centered')

        assert False


def test_simple_sim_psf_shape():
    """
    test we get roughly the right psf shape out. cannot expect detailed
    agreement due to pixelization
    """
    import galsim

    shear = galsim.Shear(g1=0.2, g2=-0.2)
    sim = Sim(
        rng=10,
        psf_kws={'g1': shear.g1, 'g2': shear.g2},
        gals_kws={'density': 10},
    )
    data = sim.gen_sim()
    se_obs = data[sim.bands[0]][0]

    psf = se_obs.get_psf(10, 3, center_psf=True).array

    cen = (np.array(psf.shape)-1)/2
    ny, nx = psf.shape
    rows, cols = np.mgrid[
        0:ny,
        0:nx,
    ]

    rows = rows - cen[0]
    cols = cols - cen[1]

    mrr = (rows**2 * psf).sum()
    mcc = (cols**2 * psf).sum()
    mrc = (rows * cols * psf).sum()

    T = mrr + mcc  # noqa
    e1 = (mcc - mrr)/T
    e2 = 2*mrc/T

    assert abs(e1 - shear.e1) < 0.01
    assert abs(e2 - shear.e2) < 0.01


def test_simple_sim_se_ps_psf():
    sim = Sim(
        rng=10,
        psf_type='ps',
        psf_kws={'noise_level': 0},
        gals_kws={'density': 10},
    )
    data = sim.gen_sim()

    for band, bdata in data.items():
        for se_obs in bdata:
            psf1 = se_obs.get_psf(10, 3)
            psf2 = se_obs.get_psf(500, 100)
            assert not np.allclose(psf1.array, psf2.array)

            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
                axs[0].imshow(psf1.array)
                axs[1].imshow(psf2.array)
                axs[2].imshow(psf1.array - psf2.array)
                assert False
