import pytest

from ..simple_sim import SimpleSim
from ..gen_sip_wcs import gen_sip_wcs
from ..gen_tanwcs import gen_tanwcs

EXPECTED_FUNCS = {
    'tan': gen_tanwcs,
    'tan-sip': gen_sip_wcs,
}


@pytest.mark.parametrize('wcs_type', ['tan', 'tan-sip'])
def test_simple_sim_smoke(wcs_type):
    sim = SimpleSim(
        rng=10,
        gals_kws={'density': 10},
        wcs_kws={'type': wcs_type},
    )

    assert sim.wcs_func == EXPECTED_FUNCS[wcs_type]
