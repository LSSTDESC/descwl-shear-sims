# wl-shear-testing-sims
[![Build Status](https://travis-ci.com/LSSTDESC/wl-shear-testing-sims.svg?branch=master)](https://travis-ci.com/LSSTDESC/wl-shear-testing-sims)

simple simulations for testing weak lensing shear measurement

## Installation

It is best to use `pip` to install

```bash
pip install .
```

If you are installing into a `conda` environment, you should add `--no-deps` to the 
command above and make sure to install the dependencies with `conda`.

## Example Usage

```python
from descwl_shear_testing.simple_sim import Sim

data = Sim(rng=10, epochs_per_band=3).gen_sim()

for band_ind, band in enumerate(data):
    for epoch_ind, obs in enumerate(data[band]):
        # do something with obs here
        # obs.image is the image
        # obs.weight is the weight map
        # obs.get_psf(x, y) will produce an image of the PSF
        # obs.wcs is the galsim WCS object associated with the image
        pass
```

## Documentation

The doc strings for the main public APIs are complete. See them for more details.
