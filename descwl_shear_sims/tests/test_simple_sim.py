import pytest

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
            assert abs(nvar/expected_var-1) < 0.01


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
