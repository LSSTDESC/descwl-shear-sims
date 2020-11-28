# descwl-shear-sims basic tutorial

In this tutorial we will

1. Install the code
1. Install some simulation data (optional)
1. Run basic examples
1. Generate your own galaxy catalog class for use in the sim

Note this tutorial is designed to work with v0.2.0 of the descwl-shear-sims backage, and
may not work with newer versions.

## Installing the Code

I highly recommend you follow this tutorial on your laptop or a local machine
rather than at nersc.  It is possible to work at nersc but it will likely take
you longer to get it working.

First make sure you have anaconda installed.  If you don't yet have it
installed, I recommend installing
[miniconda]:https://docs.conda.io/en/latest/miniconda.html

Matt becker has put together a [conda environment file]:https://raw.githubusercontent.com/beckermr/mdet-lsst-sim-runs/main/environment.yml  Get this file and run
```bash
conda env create -f environment.yml
```
This will create a new environment called `mdet-lsst-sims` (short for metadetect lsst sims).

Now activate the environment.  You can do this two ways
```bash
# probably most users will do
conda activate mdet-lsst-sims
# some will prefer to do this
source activate mdet-lsst-sims
```

To make sure the basic code is working, bring up a python session and try to import
the library
```python
import descwl_shear_sims
```


# Install simulation data (optional)

This is data used by the the WeakLensingDeblending to generate galaxies, as well
as for generating stars and star masks and bleeds.  You can skip this if you only
plan to work with simpler simulations

```bash
wget https://www.cosmo.bnl.gov/www/esheldon/data/catsim.tar.gz
tar xvfz catsim.tar.gz

# for bash users
export CATSIM_DIR=/path/to/catsim
# or for csh/tcsh
setenv CATSIM_DIR /path/to/catsim
```

I recommend putting that environment variable into your shell so you don't have to type that every time.  E.g. in your ~/.bashrc or ~/.csrhc etc.

# Run some basic examples

I will be creating a python script and running it, but if you want to use a
notebook feel free to do so.  Also note if you must work with a notebook at
nersc you may need to follow the instructions
[here]:https://github.com/beckermr/mdet-lsst-sim-runs/ to get the environment
to work.

Let's start with a simple simulation.  Copy and past this into a file or notebook:
```python
import numpy as np
from descwl_shear_sims.sim import (
    FixedGalaxyCatalog,  # one of the galaxy catalog classes
    make_sim,  # for making a simulation realization
    make_psf,  # for making a simple PSF
)

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
Now run it, and if all goes well you will see two lines of output
```bash
python test-simple-sim.py
trial: 1/2
trial: 2/2
```
