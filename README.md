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

## SIM todo

- [x] create noise field
- [ ] offset and rotate images
- [ ] complex wcs in image and noise field
- [ ] spatially varying PSF
- [ ] defects with correct bit mask values, for image and noise field

## measure TODO

- [x] add noise field to CoaddObs
- [x] add test of CoaddObs, including noise and weight compatibility
- [ ] add tests of SimMetadetect code
- [ ] do real coadds for image and noise field
- [ ] run centroiding algorithm in stack rather than using integer positions
- [ ] run size measurement alg. in stack, to get postage stamp size, rather than using default footprint
- [ ] evaluate stamp size is better, alg. based or footprint based
- [ ] test with noise field propagated (need to add that in simple sim)
- [ ] test with deblending and noise replacers
- [ ] add per band measurements to metadetect/fitting, initially in moments then maybe as some kind of fit
- [ ] evaluate if we can use measures from stack for metadetect shapes
- [ ] learn to extract sky noise value, which must be used in the noise replacers, although currently we are adding this by hand with the variance array
