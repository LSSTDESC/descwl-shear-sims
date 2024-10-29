from ..shear import ShearHalo
import galsim


def test_shear_halo():
    # Test shear halo

    mass = 8e14
    conc = 4.
    z_lens = 0.25
    z_source = 1.0
    ra_lens, dec_lens = 0.0, 0.0

    gso = galsim.Gaussian(sigma=1)
    input_shift = galsim.PositionD(1, 0)

    shear_halo = ShearHalo(mass=mass, conc=conc, z_lens=z_lens,
                           ra_lens=ra_lens, dec_lens=dec_lens)
    distor_res = \
        shear_halo.distort_galaxy(gso, input_shift, z_source)

    lensed_shift = distor_res["lensed_shift"]
    gamma1 = distor_res["gamma1"]
    gamma2 = distor_res["gamma2"]
    kappa = distor_res["kappa"]

    assert lensed_shift != input_shift, \
        "Lensed shift should not be the same as input shift"
    assert gamma1 < 0, "Gamma1 should be negative"
    assert gamma2 == 0, "Cross shear should be zero"

    shear_halo_no_kappa = ShearHalo(mass=mass, conc=conc,
                                    z_lens=z_lens, no_kappa=True)
    distor_res_no_kappa = \
        shear_halo_no_kappa.distort_galaxy(gso, input_shift, z_source)

    kappa = distor_res_no_kappa["kappa"]

    assert kappa == 0, "Kappa should be zero for ShearHalo with no_kappa=True"

    return
