"""
    - do a straight coadd and store it in an ngmix Observation for the
    simple_sim, which makes perfectly registered images with a pixel scale and
    same psf and wcs for all images

    - run detection and deblending
    - run a stub for measurement on deblended images (not doing anything yet)
    - optionally make a plot and overplot detections on the image
"""
import sys
import logging
import galsim
import numpy as np
import matplotlib.pyplot as plt
import ngmix

import lsst.afw.table as afw_table
import lsst.afw.image as afw_image
from lsst.afw.geom import makeSkyWcs, makeCdMatrix
import lsst.geom as geom
from lsst.meas.algorithms import SingleGaussianPsf
from lsst.meas.base import NoiseReplacerConfig, NoiseReplacer
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig

from descwl_shear_testing import Sim
from descwl_shear_testing.coadd_obs import CoaddObs
from descwl_shear_testing.metadetect import SimMetadetect
import fitsio
import esutil as eu
import argparse


def make_schema():
    # This schema holds all the measurements that will be run within the stack
    # It needs to be constructed before running anything and passed to
    # algorithms that make additional measurents.
    schema = afw_table.SourceTable.makeMinimalSchema()
    return schema


def detect_and_deblend(exposure):

    schema = make_schema()
    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = 10
    detection_task = SourceDetectionTask(config=detection_config)

    deblend_config = SourceDeblendConfig()
    deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)

    # Detect objects
    table = afw_table.SourceTable.make(schema)
    result = detection_task.run(table, exposure)
    sources = result.sources

    # run the deblender
    deblend_task.run(exposure, sources)

    return sources


def measure_deblended(exposure, sources):
    # Run on deblended images
    noise_replacer_config = NoiseReplacerConfig()
    footprints = {record.getId(): (record.getParent(), record.getFootprint())
                  for record in sources}

    # This constructor will replace all detected pixels with noise in the image
    replacer = NoiseReplacer(
        noise_replacer_config,
        exposure=exposure,
        footprints=footprints,
    )

    for record in sources:
        # Skip parent objects where all children are inserted
        if record.get('deblend_nChild') != 0:
            continue

        # This will insert a single source into the image
        replacer.insertSource(record.getId())

        # Get the peak as before
        peak = record.getFootprint().getPeaks()[0]
        x, y = peak.getIx(), peak.getIy()

        # The bounding box will be for the parent object
        bbox = record.getFootprint().getBBox()

        # get other info as before
        local_image = exposure.image[bbox]  # noqa

        local_psf = exposure.getPsf().computeKernelImage(geom.Point2D(x, y))  # noqa
        affine_wcs = exposure.getWcs().linearizePixelToSky(
            geom.Point2D(x, y),
            geom.arcseconds,
        )
        local_jacobian = affine_wcs.getLinear().getMatrix()  # noqa

        # Remove object
        replacer.removeSource(record.getId())

    # Insert all objects back into image
    replacer.end()


def get_exposure(coadd_obs, psf_sigma_pixels):
    ny, nx = coadd_obs.image.shape

    masked_image = afw_image.MaskedImageF(nx, ny)
    masked_image.image.array[:] = coadd_obs.image

    var = 1.0/coadd_obs.weight[0, 0]
    masked_image.variance.array[:] = var
    masked_image.mask.array[:] = 0

    exp = afw_image.ExposureF(masked_image)

    # PSF
    pny, pnx = coadd_obs.psf.image.shape
    exp_psf = SingleGaussianPsf(pny, pnx, psf_sigma_pixels)
    exp.setPsf(exp_psf)

    # set WCS
    orientation = 0*geom.degrees
    scale = coadd_obs.jacobian.scale

    cd_matrix = makeCdMatrix(
        scale=scale*geom.arcseconds,
        orientation=orientation,
    )
    crpix = geom.Point2D(nx/2, ny/2)
    crval = geom.SpherePoint(0, 0, geom.degrees)
    wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
    exp.setWcs(wcs)

    return exp


def coadd_sim_data(sim_data):

    ntot = 0
    wsum = 0.0
    for band_ind, band in enumerate(sim_data):
        for epoch_ind, se_obs in enumerate(sim_data[band]):

            x, y = 100, 100

            wt = se_obs.weight[0, 0]
            wsum += wt

            if ntot == 0:

                image = se_obs.image.array.copy()
                weight = se_obs.weight.array.copy()

                psf_image = se_obs.get_psf(x, y).array
                psf_weight = psf_image*0 + 1.0/1.0**2

                pos = galsim.PositionD(x=x, y=y)
                wjac = se_obs.wcs.jacobian(image_pos=pos)
                wscale, wshear, wtheta, wflip = wjac.getDecomposition()
                jac = ngmix.DiagonalJacobian(
                    x=x,
                    y=x,
                    scale=wscale,
                    dudx=wjac.dudx,
                    dudy=wjac.dudy,
                    dvdx=wjac.dvdx,
                    dvdy=wjac.dvdy,
                )

            else:
                image += se_obs.image.array[:, :]*wt
                weight[:, :] += se_obs.weight.array[:, :]

            ntot += 1

    image *= 1.0/wsum

    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=psf_weight,
        jacobian=jac,
    )
    return ngmix.Observation(
        image=image,
        weight=weight,
        jacobian=jac,
        psf=psf_obs,
    )


def make_comb_data(res):
    add_dt = [('shear_type', 'S7')]

    dlist = []
    for stype in res.keys():
        data = res[stype]
        newdata = eu.numpy_util.add_fields(data, add_dt)
        newdata['shear_type'] = stype
        dlist.append(newdata)

    return eu.numpy_util.combine_arrlist(dlist)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ntrial', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--noise', type=float, default=180)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)
    config = {
        'bmask_flags': 0,
        'metacal': {'psf': 'fitgauss'},
        'psf': {
            'model': 'gauss',
            'lm_pars': {},
            'ntry': 2,
        },
        'weight': {
            'fwhm': 1.2,
        },
        'meds': {},
    }

    logging.basicConfig(stream=sys.stdout)
    logging.getLogger('descwl_shear_testing').setLevel(
        getattr(logging, 'INFO'))

    dlist = []
    for trial in range(args.ntrial):
        print('-'*70)
        print('trial: %d/%d' % (trial+1, args.ntrial))
        sim = Sim(
            rng=rng,
            epochs_per_band=args.nepochs,
            noise_per_band=args.noise,
        )
        data = sim.gen_sim()

        # faking ngmix MultiBandObsList
        # note data is an OrderedDict
        coadd_mbobs = ngmix.MultiBandObsList(
            meta={'psf_fwhm': sim.psf_kws['fwhm']},
        )
        for band in data:
            coadd_obs = CoaddObs(data[band])
            obslist = ngmix.ObsList()
            obslist.append(coadd_obs)
            coadd_mbobs.append(obslist)

        md = SimMetadetect(config, coadd_mbobs, rng)
        md.go()
        res = md.result
        # print(res.keys())

        comb_data = make_comb_data(res)
        dlist.append(comb_data)

        if args.show:
            plt.imshow(coadd_obs.image, interpolation='nearest', cmap='gray')
            plt.scatter(
                res['noshear']['col'], res['noshear']['row'],
                c='r', s=0.5,
            )
            plt.show()
            '''
            for record in sources:
                # Skip parent objects where all children are inserted
                if record.get('deblend_nChild') != 0:
                    continue

                # Get the peak as before
                peak = record.getFootprint().getPeaks()[0]
                x, y = peak.getIx(), peak.getIy()

                plt.scatter([x], [y], c='r', s=0.5)

            plt.show()
            '''

    data = eu.numpy_util.combine_arrlist(dlist)
    print('writing:', args.output)
    fitsio.write(args.output, data, clobber=True)


if __name__ == '__main__':
    main()
