import pytest

from ..simple_sim import Sim
from ..gen_sip_wcs import gen_sip_wcs
from ..gen_tanwcs import gen_tanwcs

EXPECTED_FUNCS = {
    'tan': gen_tanwcs,
    'tan-sip': gen_sip_wcs,
}


@pytest.mark.parametrize('wcs_type', ['tan', 'tan-sip'])
def test_simple_sim_smoke(wcs_type):
    sim = Sim(
        rng=10,
        gals_kws={'density': 10},
        wcs_kws={'type': wcs_type},
    )
    if wcs_type == 'tan-sip':
        assert sim.wcs_func == gen_sip_wcs
    else:
        assert sim.wcs_func == gen_tanwcs
