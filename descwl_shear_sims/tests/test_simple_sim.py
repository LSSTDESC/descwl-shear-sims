import pytest
import numpy as np

from ..se_obs import SEObs
from ..simple_sim import Sim


def test_simple_sim_smoke():
    sim = Sim(rng=10)
    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            epoch_obs = data[band][epoch]
            assert isinstance(epoch_obs, SEObs)
            assert epoch_obs.noise is not None


def test_simple_sim_noise():
    sim = Sim(rng=10)
    data = sim.gen_sim()
    for band in sim.bands:
        for epoch in range(sim.epochs_per_band):
            epoch_obs = data[band][epoch]

            assert epoch_obs.noise is not None

            nvar = epoch_obs.noise.array.var()
            expected_var = 1/epoch_obs.weight.array[0, 0]
            assert abs(nvar/expected_var-1) < 0.015


def test_simple_sim_double_call_raises():
    sim = Sim(rng=10)
    sim.gen_sim()
    with pytest.raises(RuntimeError):
        sim.gen_sim()


def test_simple_sim_psf_smoke():
    sim = Sim(rng=10)
    data = sim.gen_sim()
    assert len(data) == sim.n_bands
    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            # make sure this call works
            data[band][epoch].get_psf(10, 3)


def test_simple_sim_psf_center():
    sim = Sim(rng=10)
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
