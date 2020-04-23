from setuptools import setup, find_packages

setup(
    name="descwl_shear_sims",
    version="0.3.0",
    packages=find_packages(),
    install_requires=['numpy', 'galsim'],
    include_package_data=True,
    author='Matthew R. Becker',
    author_email='becker.mr@gmail.com',
    url='https://github.com/LSSTDESC/descwl-shear-sims',
)
