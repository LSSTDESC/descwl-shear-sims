## 0.4.3
### new features
    - add support for ring test
    - add random circular layout where galaxies are randomly distribution in a
      circle
    - the objlist now includs the input galaxy's position, redshift, index (in
      the truth catalog)
    - use shear object to apply distortion, enabling adding lensing distortion
      from clusters

## 0.4.2

### new features
    - add randomized psf, fixed in an image but with random size and shape.
      New function .psf.make_rand_psf


## 0.4.0/0.4.1

### changes to defaults

    - coadd dims set to 250 in DEFAULT_SIM_CONFIG to match plans for lsst cells
      This dict is used in the convenience function "get_sim_config" code.
    - default buff=0 in DEFAULT_SIM_CONFIG and in functions/classes, allowing
      objects drawn to the edge of the coadd.  Only used for random layouts.

### new features

   - get_se_dim now takes in dither and rotation, adjusting the dims
       accordingly to just cover the coadd.
   - added draw_stars bool keyword to make_sim and part of DEFAULT_SIM_CONFIG
     to isolate the effect of star masking from the presence of stars in the
     image
   - added "hex" layout which is a randomly rotated hexagonal lattice
