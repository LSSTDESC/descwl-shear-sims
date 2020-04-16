import descwl
import math


class RoundGalaxyBuilder(descwl.model.GalaxyBuilder):
    def from_catalog(self,
                     entry,
                     dx_arcsecs,
                     dy_arcsecs,
                     filter_band):

        """
        unfortunately the galaxy builder method from_catalog is not
        modular, so we need to copy and paste the whole thing just
        to make objects round

        Build a :class:Galaxy object from a catalog entry.

        Fluxes are distributed between the three possible components
        (disk,bulge,AGN) assuming that each component has the same spectral
        energy distribution, so that the resulting proportions are independent
        of the filter band.

        Args:

            entry(astropy.table.Row): A single row from a galaxy
                :mod:`descwl.catalog`.

            dx_arcsecs(float): Horizontal offset of catalog entry's centroid
              from image center in arcseconds.

            dy_arcsecs(float): Vertical offset of catalog entry's centroid from
              image center in arcseconds.

            filter_band(str): The LSST filter band to use for calculating flux,
              which must be one of 'u','g','r','i','z','y'.


        Returns:
            :class:`Galaxy`: A newly created galaxy source model.

        Raises:

            SourceNotVisible: All of the galaxy's components are being ignored.

            RuntimeError: Catalog entry is missing AB flux value in requested
            filter band.
        """
        # Calculate the object's total flux in detected electrons.
        try:
            ab_magnitude = entry[filter_band + '_ab']
            ri_color = entry['r_ab'] - entry['i_ab']
        except KeyError:
            raise RuntimeError(
                'Catalog entry is missing required AB magnitudes.')

        total_flux = self.survey.get_flux(ab_magnitude)

        # Calculate the flux of each component in detected electrons.
        total_fluxnorm = (
            entry['fluxnorm_disk'] +
            entry['fluxnorm_bulge'] +
            entry['fluxnorm_agn']
        )

        if self.no_disk:
            disk_flux = 0.0
        else:
            disk_flux = entry['fluxnorm_disk']/total_fluxnorm*total_flux

        if self.no_bulge:
            bulge_flux = 0.0
        else:
            bulge_flux = entry['fluxnorm_bulge']/total_fluxnorm*total_flux

        if self.no_agn:
            agn_flux = 0.0
        else:
            agn_flux = entry['fluxnorm_agn']/total_fluxnorm*total_flux

        # Is there any flux to simulate?
        if disk_flux + bulge_flux + agn_flux == 0:
            raise descwl.model.SourceNotVisible

        disk_q = 1
        bulge_q = 1
        beta_radians = 0.0

        # Calculate shapes hlr = sqrt(a*b) and q = b/a of Sersic components.
        if disk_flux > 0:
            a_d, b_d = entry['a_d'], entry['b_d']
            disk_hlr_arcsecs = math.sqrt(a_d*b_d)
        else:
            disk_hlr_arcsecs, disk_q = None, None

        if bulge_flux > 0:
            a_b, b_b = entry['a_b'], entry['b_b']
            bulge_hlr_arcsecs = math.sqrt(a_b*b_b)
        else:
            bulge_hlr_arcsecs, bulge_q = None, None

        # Look up extra catalog metadata.
        identifier = entry['galtileid']
        redshift = entry['redshift']

        if self.verbose_model:
            print('Building galaxy model for '
                  'id=%d with z=%.3f' % (identifier, redshift))
            print(
                'flux = %.3g detected '
                'electrons (%s-band AB = %.1f)' % (
                    total_flux, filter_band, ab_magnitude)
            )
            print(
                'centroid at (%.6f,%.6f) '
                'arcsec relative to image center, beta = %.6f rad' % (
                    dx_arcsecs, dy_arcsecs, beta_radians)
            )

            if disk_flux > 0:
                print(' disk: frac = %.6f, hlr = %.6f arcsec, q = %.6f' % (
                    disk_flux/total_flux, disk_hlr_arcsecs, disk_q))
            if bulge_flux > 0:
                print('bulge: frac = %.6f, hlr = %.6f arcsec, q = %.6f' % (
                    bulge_flux/total_flux, bulge_hlr_arcsecs, bulge_q))
            if agn_flux > 0:
                print('  AGN: frac = %.6f' % (agn_flux/total_flux))

        return descwl.model.Galaxy(
            identifier,
            redshift,
            ab_magnitude,
            ri_color,
            self.survey.cosmic_shear_g1,
            self.survey.cosmic_shear_g2,
            dx_arcsecs,
            dy_arcsecs,
            beta_radians,
            disk_flux,
            disk_hlr_arcsecs,
            disk_q,
            bulge_flux,
            bulge_hlr_arcsecs,
            bulge_q,
            agn_flux,
        )
