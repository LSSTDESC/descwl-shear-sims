# descwl-shear-sims
[![Build Status](https://travis-ci.com/LSSTDESC/descwl-shear-sims.svg?branch=master)](https://travis-ci.com/LSSTDESC/descwl-shear-sims) [![shear-meas-tests](https://github.com/LSSTDESC/descwl-shear-sims/actions/workflows/shear_meas_tests.yml/badge.svg)](https://github.com/LSSTDESC/descwl-shear-sims/actions/workflows/shear_meas_tests.yml)

Simulations for testing weak lensing shear measurement

## Example Usage

### A simple sim
```python
import numpy as np
from descwl_shear_sims.galaxies import FixedGalaxyCatalog
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.psfs import make_fixed_psf
seed = 8312
rng = np.random.RandomState(seed)

ntrial = 2
coadd_dim = 351
buff = 50

for trial in range(ntrial):
    print('trial: %d/%d' % (trial+1, ntrial))

    # galaxy catalog; you can make your own
    galaxy_catalog = FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout='random',
        mag=25,
        hlr=1.0,
    )

    # make a constant gaussian psf
    psf = make_fixed_psf(psf_type='gauss')

    # generate some simulation data, with a particular shear

    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )
```

### A sim with lots of features turned on

```python
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog  # one of the galaxy catalog classes
from descwl_shear_sims.stars import StarCatalog  # star catalog class
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.psfs import make_ps_psf  # for making a power spectrum PSF
from descwl_shear_sims.sim import get_se_dim  # convert coadd dims to SE dims

seed = 8312
rng = np.random.RandomState(seed)

ntrial = 2
coadd_dim = 351
buff = 50
rotate = True
dither = True

# this is the single epoch image sized used by the sim, we need
# it for the power spectrum psf
se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)

for trial in range(ntrial):
    print('trial: %d/%d' % (trial+1, ntrial))

    # galaxy catalog; you can make your own
    galaxy_catalog = WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
    )
    # star catalog; you can make one of these too
    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
    )
    # make a power-spectrum PSF, again you can make your own PSF
    psf = make_ps_psf(rng=rng, dim=se_dim)

    # generate some simulation data, with a particular shear,
    # and dithering, rotation, cosmic rays, bad columns, star bleeds
    # turned on.  By sending the star catalog we generate stars and
    # some can be saturated and bleed

    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=dither,
        rotate=rotate,
        bands=['r', 'i', 'z'],
        noise_factor=0.58,
        cosmic_rays=True,
        bad_columns=True,
        star_bleeds=True,
    )
```

## Installation

See the requirements.txt for a list of dependencies.  Note hexalattice is
optional for the hex grid layout

This code uses data structures from the LSST science pipelines.  If you need to
install that code, it is best to use the `stackvana` package in conda forge, as
listed in the requirements.txt.  If you already have that code installed, you
can remove it from the requirements.

```bash
conda create -n sims
conda install --file requirements.txt
pip install .
```

## Getting the Simulation Input Data

We have packaged some optional but useful data that can be used
with descwl-shear-sims.
- Galaxy models from WeakLensingDeblending
- Realistic star fluxes
- Realistic galctic star spatial density distribution
- use realistic star bleed trail masks

Do the following to make that data available to descwl-shear-sims
```shell
wget https://www.cosmo.bnl.gov/www/esheldon/data/catsim.tar.gz
tar xvfz catsim.tar.gz
export CATSIM_DIR=/path/to/catsim

# or for tcsh
# setenv CATSIM_DIR /path/to/catsim
```

## more examples

More examples are given in the examples/ sub directory

## Further Documentation

The doc strings for the main public APIs are complete. See them for more details.
