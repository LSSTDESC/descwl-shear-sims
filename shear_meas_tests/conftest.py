import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--save",
        action="store_true",
        default=False,
        help="--save: Wether to save the results in a separate file"
    )
    parser.addoption(
        "--save_dir",
        action="store",
        default='.',
        type=str,
        help="--save_dir: Directory where to store the output"
    )
    parser.addoption(
        "--g1_noise",
        action="store",
        default=0.,
        type=float,
        help="--g1_noise: g1 shear to apply on the noise"
    )
    parser.addoption(
        "--g2_noise",
        action="store",
        default=0.,
        type=float,
        help="--g2_noise: g2 shear to apply on the noise"
    )
    parser.addoption(
        "--n_jobs",
        action="store",
        default=2,
        type=int,
        help="--n_jobs: Number of cores to use"
    )


@pytest.fixture()
def save(request):
    return request.config.getoption("--save")


@pytest.fixture()
def save_dir(request):
    return request.config.getoption("--save_dir")


@pytest.fixture()
def g1_noise(request):
    return request.config.getoption("--g1_noise")


@pytest.fixture()
def g2_noise(request):
    return request.config.getoption("--g2_noise")


@pytest.fixture()
def n_jobs(request):
    return request.config.getoption("--n_jobs")
