from ..simple_sim import SimpleSim


def test_simple_sim_psf_dims():
    psf_dim = 101
    sim = SimpleSim(
        rng=10,
        psf_dim=psf_dim,
        gals_kws={'density': 10},
    )

    data = sim.gen_sim()

    for band in sim.bands:
        assert len(data[band]) == sim.epochs_per_band
        for epoch in range(sim.epochs_per_band):
            epoch_obs = data[band][epoch]
            psf_im = epoch_obs.get_psf(10, 3)
            assert psf_im.array.shape == (psf_dim, psf_dim)
