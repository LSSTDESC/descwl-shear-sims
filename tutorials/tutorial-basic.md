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
This will create a new environment called `mdet-lsst-sims` (short for
metadetect lsst sims).  Note this installation uses quite a lot of memory. You
may want to close high memory programs such as your browser before doing the
install.

Now activate the environment.  You can do this two ways
```bash
# probably most people will do this
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

We will get a tar file with data used by the the WeakLensingDeblending to
generate galaxies, as well as some data for generating stars and star masks and
bleeds.  You can skip this if you only plan to work with simpler simulations

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
    # make a constant gaussian psf
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
    #      image generating method get_psf(x, y).  The images are galsim
    #      Image objects.
    #    coadd_wcs:  the wcs for the coadd
    #    psf_dims:  dimensions of the psf
    #    coadd_dims: dimensions of the coadd
    print('bands:', sim_data['band_data'].keys())
    print('nepocs i:', len(sim_data['band_data']['i']))
    print('image shape:', sim_data['band_data']['i'][0].image.array.shape)
    print('psf shape:', sim_data['psf_dims'])
    print('coadd shape:', sim_data['coadd_dims'])
```
Now run it, and if all goes well you will see this output:
```bash
python test-simple-sim.py
trial: 1/2
bands: dict_keys(['i'])
nepocs i: 1
image shape: (517, 517)
psf shape: [51, 51]
coadd shape: [351, 351]
trial: 2/2
bands: dict_keys(['i'])
nepocs i: 1
image shape: (517, 517)
psf shape: [51, 51]
coadd shape: [351, 351]
```

Now if you installed the data you can run a complex example.
This example has complex galaxies with realistic flux, colors, and
size, as well as stars, star bleed trails and masking, cosmic rays,
bad columns, image dithers and rotations.  If you didn't install the
data but want to try this, you can just use the same galaxy catalog
as above and turn off stars.
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
        psf_dim=51,
        dither=True,
        rotate=True,
        bands=['r', 'i', 'z'],
        epochs_per_band=1,
        noise_factor=0.58,
        cosmic_rays=True,
        bad_columns=True,
        star_bleeds=True,
    )
    print('bands:', sim_data['band_data'].keys())
    print('nepocs i:', len(sim_data['band_data']['i']))
    print('image shape:', sim_data['band_data']['i'][0].image.array.shape)
    print('psf shape:', sim_data['psf_dims'])
    print('coadd shape:', sim_data['coadd_dims'])
```

Now run it, and if all goes well you will see this output:
```bash
python test-full-sim.py
trial: 1/2
bands: dict_keys(['r', 'i', 'z'])
nepocs i: 1
image shape: (517, 517)
psf shape: [51, 51]
coadd shape: [351, 351]
trial: 2/2
bands: dict_keys(['r', 'i', 'z'])
nepocs i: 1
image shape: (517, 517)
psf shape: [51, 51]
coadd shape: [351, 351]
```
