# descwl-shear-sims
[![Build Status](https://travis-ci.com/LSSTDESC/descwl-shear-sims.svg?branch=master)](https://travis-ci.com/LSSTDESC/descwl-shear-sims)

simple simulations for testing weak lensing shear measurement

## Installation

It is best to use `pip` to install

```bash
pip install .
```

If you are installing into a `conda` environment, you should add `--no-deps` to the 
command above and make sure to install the dependencies with `conda`.

## Installing with all the dependencies

Matt Becker has put up examples to install the full dependencies, including
the DM STack:

[Full Installation with Dependencies](https://github.com/beckermr/mdet-lsst-sim-runs)

## Example Usage

### A simple sim
```python
import numpy as np
from descwl_shear_sims.sim import (
    FixedGalaxyCatalog,  # one of the galaxy catalog classes
    make_sim,  # for making a simulation realization
    make_psf,  # for making a simple PSF
    get_se_dim,  # convert coadd dims to SE dims
)

seed = 8312
rng = np.random.RandomState(seed)

ntrial = 2
coadd_dim = 351
buff = 50

# this is the single epoch image sized used by the sim, we need
# it for the power spectrum psf
se_dim = get_se_dim(coadd_dim=coadd_dim)

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
    # make a power-spectrum PSF, again you can make your own PSF
    psf = make_psf(psf_type='gauss')

    # generate some simulation data, with a particular shear

    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )

    # the sim_data has keys
    #    band_data: a dict keyed by band with a list of single-epoch
    #      observations objects, one for each epoch.  The class is
    #      SEObs, defined in descwl_shear_sims.se_obs.py and has attributes
    #      for the image, weight map, wcs, noise image, bmask and a psf
    #      image generating method get_psf(x, y)
    #    coadd_wcs:  the wcs for the coadd
    #    psf_dims:  dimensions of the psf
    #    coadd_dims: dimensions of the coadd
```

### A sim with lots of features turned on

```python
import numpy as np
from descwl_shear_sims.sim import (
    WLDeblendGalaxyCatalog,  # one of the galaxy catalog classes
    StarCatalog,  # star catalog class
    make_sim,  # for making a simulation realization
    make_ps_psf,  # for making a power spectrum PSF
    get_se_dim,  # convert coadd dims to SE dims
)

seed = 8312
rng = np.random.RandomState(seed)

ntrial = 2
coadd_dim = 351
buff = 50

# this is the single epoch image sized used by the sim, we need
# it for the power spectrum psf
se_dim = get_se_dim(coadd_dim=coadd_dim)

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
        dither=True,
        rotate=True,
        bands=['r', 'i', 'z'],
        noise_factor=0.58,
        cosmic_rays=True,
        bad_columns=True,
        star_bleeds=True,
    )
```

## Documentation

The doc strings for the main public APIs are complete. See them for more details.

## Getting the Simulation Input Data

To use galaxy models from WeakLensingDeblending, and to use realistic star masks, get this
tar ball, untar it and set the $CATSIM_DIR environment variable to that location
```shell
wget https://www.cosmo.bnl.gov/www/esheldon/data/catsim.tar.gz
tar xvfz catsim.tar.gz
export CATSIM_DIR=/path/to/catsim
```
