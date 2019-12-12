from setuptools import setup, find_packages

setup(
    name="descwl_shear_testing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=['numpy', 'galsim'],
    include_package_data=True,
)
