import numpy as np
import metadetect
from .medsifier import SimMEDSifier


class SimMetadetect(metadetect.Metadetect):
    def _do_detect(self, mbobs):
        psf_fwhm = self.mbobs.meta['psf_fwhm']
        return SimMEDSifier(
            mbobs=mbobs,
            meds_config=self['meds'],
            psf_fwhm_arcsec=psf_fwhm,
        )

    def _set_ormask(self):
        """
        fake ormask for now
        """
        self.ormask = np.zeros(self.mbobs[0][0].image.shape, dtype='i4')

    def _set_config(self, config):
        """
        set the config, dealing with defaults
        """

        self.update(config)
        assert 'metacal' in self, \
            'metacal setting must be present in config'
        # assert 'meds' in self, \
        #     'meds setting must be present in config'
