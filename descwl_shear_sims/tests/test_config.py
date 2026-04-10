import pytest

from ..sim import get_sim_config


def test_config_smoke():
    get_sim_config()


def test_config_input():
    cin = {'psf_type': 'moffat', 'psf_fwhm': 0.9}
    config = get_sim_config(config=cin)
    assert config['psf_type'] == cin['psf_type']
    assert config['psf_fwhm'] == cin['psf_fwhm']


def test_config_pure():
    cin = {'bands': ['r', 'i']}
    config = get_sim_config(config=cin)

    assert config['bands'] == cin['bands']

    config['bands'].append('z')

    assert config['bands'] != cin['bands']


def test_config_badkey():
    cin = {'badkey': 5}
    with pytest.raises(ValueError):
        get_sim_config(config=cin)
