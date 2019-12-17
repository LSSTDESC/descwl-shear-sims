import logging
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.meas.algorithms import SingleGaussianPsf
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs, makeCdMatrix
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
import ngmix
import metadetect
from meds.util import get_image_info_struct

LOGGER = logging.getLogger(__name__)


class SimMEDSInterface(metadetect.detect.MEDSInterface):
    def __init__(self, obs, sources):
        self.obs = obs
        self._image_types = (
            'image', 'weight', 'bmask', 'noise')
        self.sources = sources
        self._image_info = get_image_info_struct(1, 20)
        self._set_exposure()

    def get_cutout(self, iobj, icutout, type='image'):
        """
        Get a single cutout for the indicated entry

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.
        type: string, optional
            Cutout type. Default is 'image'.  Allowed
            values are 'image','weight','seg','bmask'

        returns
        -------
        The cutout image
        """

        self._check_indices(iobj, icutout=icutout)

        if type == 'psf':
            return self.get_psf(iobj, icutout)

        im = self._get_type_image(type)
        dims = im.shape

        c = self._cat
        orow = c['orig_start_row'][iobj, icutout]
        ocol = c['orig_start_col'][iobj, icutout]
        bsize = c['box_size'][iobj]

        orow_box, row_box = self._get_clipped_boxes(dims[0], orow, bsize)
        ocol_box, col_box = self._get_clipped_boxes(dims[1], ocol, bsize)

        read_im = im[orow_box[0]:orow_box[1],
                     ocol_box[0]:ocol_box[1]]

        subim = np.zeros((bsize, bsize), dtype=im.dtype)
        subim += defaults.DEFAULT_IMAGE_VALUES[type]

        subim[row_box[0]:row_box[1],
              col_box[0]:col_box[1]] = read_im

        return subim

    def _set_exposure(self):
        ny, nx = self.obs.image.shape

        scale = self.obs.jacobian.scale
        masked_image = afwImage.MaskedImageF(nx, ny)
        masked_image.image.array[:] = self.detim

        var = self.detnoise**2
        masked_image.variance.array[:] = var
        masked_image.mask.array[:] = 0

        exp = afwImage.ExposureF(masked_image)

        # PSF for detection
        """
        psf_sigma = ngmix.moments.fwhm_to_sigma(self.psf_fwhm_arcsec)
        psf_sigma_pixels = psf_sigma/scale

        pny, pnx = self.mbobs[0][0].psf.image.shape
        exp_psf = SingleGaussianPsf(pny, pnx, psf_sigma_pixels)
        exp.setPsf(exp_psf)
        """

        # set WCS
        orientation = 0*geom.degrees

        cd_matrix = makeCdMatrix(
            scale=scale*geom.arcseconds,
            orientation=orientation,
        )
        crpix = geom.Point2D(nx/2, ny/2)
        crval = geom.SpherePoint(0, 0, geom.degrees)
        wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
        exp.setWcs(wcs)

        self.exposure = exp


class SimMEDSifier(metadetect.detect.MEDSifier):
    def __init__(self, *, mbobs, meds_config, psf_fwhm_arcsec):
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.psf_fwhm_arcsec = psf_fwhm_arcsec

        assert len(mbobs[0]) == 1, 'multi-epoch is not supported'

        self._set_meds_config(meds_config)

        LOGGER.info('setting detection image')
        self._set_detim()

        LOGGER.info('setting detection exposure')
        self._set_detim_exposure()

        LOGGER.info('detecting and deblending')
        self._detect_and_deblend()

    def get_meds(self, band):
        return SimMEDSInterface(
            self.mbobs[band],
            self.sources,
        )

    def _set_detim_exposure(self):
        ny, nx = self.detim.shape

        scale = self.mbobs[0][0].jacobian.scale
        masked_image = afwImage.MaskedImageF(nx, ny)
        masked_image.image.array[:] = self.detim

        var = self.detnoise**2
        masked_image.variance.array[:] = var
        masked_image.mask.array[:] = 0

        exp = afwImage.ExposureF(masked_image)

        # PSF for detection
        psf_sigma = ngmix.moments.fwhm_to_sigma(self.psf_fwhm_arcsec)
        psf_sigma_pixels = psf_sigma/scale

        pny, pnx = self.mbobs[0][0].psf.image.shape
        exp_psf = SingleGaussianPsf(pny, pnx, psf_sigma_pixels)
        exp.setPsf(exp_psf)

        # set WCS
        orientation = 0*geom.degrees

        cd_matrix = makeCdMatrix(
            scale=scale*geom.arcseconds,
            orientation=orientation,
        )
        crpix = geom.Point2D(nx/2, ny/2)
        crval = geom.SpherePoint(0, 0, geom.degrees)
        wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
        exp.setWcs(wcs)

        self.det_exp = exp

    def _detect_and_deblend(self):
        exposure = self.det_exp

        # This schema holds all the measurements that will be run within the stack
        # It needs to be constructed before running anything and passed to
        # algorithms that make additional measurents.
        schema = afwTable.SourceTable.makeMinimalSchema()

        detection_config = SourceDetectionConfig()
        detection_config.reEstimateBackground = False
        detection_config.thresholdValue = 10
        detection_task = SourceDetectionTask(config=detection_config)

        deblend_config = SourceDeblendConfig()
        deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)

        # Detect objects
        table = afwTable.SourceTable.make(schema)
        result = detection_task.run(table, exposure)
        sources = result.sources

        # run the deblender
        deblend_task.run(exposure, sources)

        self.sources = sources
