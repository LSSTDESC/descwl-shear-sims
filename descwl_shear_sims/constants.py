import galsim

# magnitude zero point for all images
ZERO_POINT = 30.0

# pixel scale for all images in arcseconds/pixel
SCALE = 0.2

# density of random layout (not wldeblend) per square arcmin
RANDOM_DENSITY = 80

# spacing of the square grid in arcsec
GRID_SPACING = 9.5

# spacing of the hex grid in arcsec
HEX_SPACING = 9.5

# fwhm of the special fixed psf
FIXED_PSF_FWHM = 0.8

# beta for fixed moffat psfs
FIXED_MOFFAT_BETA = 2.5

RAND_PSF_FWHM_MEAN = 0.8
RAND_PSF_FWHM_STD = 0.1
RAND_PSF_FWHM_MIN = 0.6
RAND_PSF_FWHM_MAX = 1.3
RAND_PSF_E_STD = 0.01
RAND_PSF_E_MAX = 0.10

WORLD_ORIGIN = galsim.CelestialCoord(
    ra=200 * galsim.degrees,
    dec=0 * galsim.degrees,
)
