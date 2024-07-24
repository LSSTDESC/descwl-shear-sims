import os
from setuptools import setup, find_packages

__version__ = ""

pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "descwl_shear_sims",
    "version.py"
)
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name="descwl_shear_sims",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    author='Matthew R. Becker',
    author_email='becker.mr@gmail.com',
    url='https://github.com/LSSTDESC/descwl-shear-sims',
)
