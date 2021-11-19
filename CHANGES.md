## 0.4.0  (unreleased)

### changes to defaults

    - coadd dims set to 250 in DEFAULT_SIM_CONFIG to match plans for lsst cells
      This dict is used in the convenience function "get_sim_config" code.
    - default buff=0 in DEFAULT_SIM_CONFIG and in functions/classes, allowing
      objects drawn to the edge of the coadd.  Only used for random layouts.

### new features

   - get_se_dim now takes in dither and rotation, adjusting the dims
       accordingly to just cover the coadd.