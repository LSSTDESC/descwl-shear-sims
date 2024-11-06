"""
example with OpenUniverse2024 Rubin+Roman galaxies
"""
import os
import numpy as np

import lsst.afw.image as afw_image
import lsst.afw.geom as afw_geom
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_ps_psf, make_fixed_psf
from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.constants import SCALE, SCALE_ROMAN_COADD

def go():
    if 'CATSIM_DIR' not in os.environ:
        # this contains the galaxy and star catalogs for generatig
        # WeakLensingDeblending galaxies and stars
        print('you need CATSIM_DIR defined to run this example')

    seed = 761
    rng = np.random.RandomState(seed)

    dither = False
    rotate = False
    coadd_length_arcsec = 120.2
    psf_length_arcsec = 10.2
    buff_length_arcsec = 6
    survey_names = ['LSST', 'Roman']
    band_lists = [['r', 'i'], ['H158']]

    coadd_dim = np.zeros(len(survey_names))
    psf_dim = np.zeros(len(survey_names))
    buff = np.zeros(len(survey_names))
    # generate simulated data
    data = {}
    for i, sname in enumerate(survey_names):
        if sname == 'LSST':
            pix_scale = SCALE
        elif sname == 'Roman':
            pix_scale = SCALE_ROMAN_COADD
        coadd_dim[i] = np.round(coadd_length_arcsec/pix_scale)
        psf_dim[i] = np.round(psf_length_arcsec/pix_scale)      
        buff[i] = np.round(buff_length_arcsec/pix_scale)

        # power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim[i], rotate=rotate, dither=dither)
        if sname == 'LSST':
            # Fixed Moffat PSF, default setting
            psf = make_fixed_psf(psf_type='moffat')
        elif sname == 'Roman':
            # For roman, the target PSF is Gaussian with 
            # FWHM=0.242 for H158
            # TODO make more realistic PSF
            psf = make_fixed_psf(psf_type='gauss', psf_fwhm=0.242)

        if i==0:
            # this makes OpenUniverse2024RubinRoman galaxies
            galaxy_catalog = make_galaxy_catalog(
                rng=rng,
                gal_type='ou2024rubinroman',
                coadd_dim=int(coadd_dim[i]),
                pixel_scale=pix_scale,
                buff=int(buff[i]),
            )
        else:
            # resize the galaxy position and sim layout 
            # according to the new pixel scale
            galaxy_catalog._resize_dimension(
                new_coadd_dim = int(coadd_dim[i]),
                new_buff = int(buff[i]),
                new_pixel_scale = pix_scale,
                )

        data[sname] = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim[i],
            psf_dim=psf_dim[i],
            bands=band_lists[i],
            survey_name=sname,
            g1=0.02,
            g2=0.00,
            psf=psf,
            dither=dither,
            rotate=rotate,
            draw_bright=False,
            draw_stars=False,
        )

    # data is a dict with the following keys.
    # band_data: a dict, keyed by band name, with values that are a list of
    #   exps
    # coadd_wcs: is a DM wcs for use in coadding
    # psf_dims: is the psf dim we sent in (psf_dim, psf_dim)
    # coadd_dims: shape of the coadd image (dim, dim)
    # coadd_bbox: is an lsst Box2I, for use in coadding
    # bright_info: is a structured array with position and mask info for bright
    #   objects
    # se_wcs: list of WCS for each single epoch image

    for sname in survey_names:
        for key in ['band_data', 'coadd_wcs', 'psf_dims', 'coadd_bbox', 'bright_info']:
            assert key in data[sname]
    for i, sname in enumerate(survey_names):
        for band in band_lists[i]:
            assert band in data[sname]['band_data']
            assert isinstance(data[sname]['band_data'][band][0], afw_image.ExposureF)
        assert isinstance(data[sname]['coadd_wcs'], afw_geom.SkyWcs)
        assert data[sname]['psf_dims'] == (psf_dim[i], )*2

        extent = data[sname]['coadd_bbox'].getDimensions()
        edims = (extent.getX(), extent.getY())
        assert edims == (coadd_dim[i], )*2

        for band in band_lists[i]:
            im = data[sname]['band_data'][band][0]
            fname = f'example_{sname}_{band}.fits' 
            fname = os.path.join('/media/taeshin/research', fname)
            im.writeFits(fname)

if __name__ == '__main__':
    go()
