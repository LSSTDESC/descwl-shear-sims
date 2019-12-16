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
            assert isinstance(data[band][epoch], SEObs)


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
