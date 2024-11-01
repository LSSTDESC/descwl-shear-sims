import os
import galsim
from galsim import roman
import numpy as np
import pyarrow as pa

from ..constants import SCALE
from ..layout import Layout
from ..cache_tools import cached_catalog_read


class OpenUniverse2024RubinRomanCatalog(object):
    """
    Catalog of galaxies from OpenUniverse2024 Rubin-Roman input galaxies
    https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html

    Parameters
    ---------
    rng: np.random.RandomState
        The random number generator
    layout: str|Layout, optional
    coadd_dim: int, optional
        Dimensions of the coadd
    buff: int, optinal
        Buffer region with no objects, on all sides of image.
        Ignored for layout 'grid'. Default 0.
    pixel_scale: float, optional
        pixel scale
    select_observable: list[str] | str
        A list of observables (data columns) to apply selection
    select_lower_limit: list | ndarray
        lower limits of the selection cuts
    select_upper_limit: list | ndarray
        upper limits of the selection cuts
    """

    def __init__(
        self,
        *,
        rng,
        layout="random",
        coadd_dim=None,
        buff=None,
        pixel_scale=SCALE,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ):
        self.gal_type = "ou2024rubinroman"
        self.rng = rng

        (
            self._ou2024rubinroman_cat,
            self.sed_cat,
            self.mask
        ) = read_ou2024rubinroman_cat(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )
        self.gal_ids = np.array(self._ou2024rubinroman_cat["galaxy_id"])

        # The catalog corresponds to an nside=32 healpix pixel.
        area_tot_arcmin = (
            60.0**2 * (180.0 / np.pi) ** 2 * 4.0 * np.pi / (12.0 * 32.0**2)
        )
        density = len(self._ou2024rubinroman_cat) / area_tot_arcmin
        if isinstance(layout, str):
            self.layout = Layout(layout, coadd_dim, buff, pixel_scale)
        else:
            assert isinstance(layout, Layout)
            self.layout = layout
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
            density=density,
        )

        # Randomly sample from the catalog
        num = len(self)
        self.indices = self.rng.randint(
            0,
            len(self._ou2024rubinroman_cat),
            size=num,
        )
        # do a random rotation for each galaxy
        self.angles = self.rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indices

        Parameters
        ----------
        survey: object with survey parameters

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        if survey.name == "roman":
            filters = roman.getBandpasses(AB_zeropoint=True)
            self.bandpass = filters[survey.band]
        elif survey.name == "lsst":
            fname_bandpass = os.path.join(
                os.environ.get("CATSIM_DIR", "."),
                f"LSST_{survey.band}.dat",
            )
            #  zeropoint for LSST is hard-coded for now
            zeropoint = {
                "u": 12.627880975839947,
                "g": 14.486360566315488,
                "r": 14.324886825965237,
                "i": 13.998598224053055,
                "z": 13.612736383992512,
                "y": 12.785119907991263,
            }
            self.bandpass = galsim.Bandpass(
                fname_bandpass, wave_type="nm"
            ).withZeropoint(zeropoint[survey.band])
        else:
            raise ValueError("survey name not supported")

        sarray = self.shifts_array
        indexes = []
        objlist = []
        shifts = []
        redshifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(survey, i))
            shifts.append(galsim.PositionD(sarray["dx"][i], sarray["dy"][i]))
            index = self.indices[i]
            indexes.append(index)
            redshifts.append(
                self._ou2024rubinroman_cat["redshift"][index].as_py()
            )

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

    def _get_galaxy(self, survey, i):
        """
        Get a galaxy

        Parameters
        ----------
        survey: object with survey parameters
            see surveys.py
        i: int
            Index of galaxies

        Returns
        -------
        galsim.GSObject
        """
        index = self.indices[i]

        # From the SED data, read the SED for the galaxy.
        # Note that how SEDs are stored can be a bit confusing.
        gal_id = str(self.gal_ids[index])
        f_sed = self.sed_cat["galaxy"][gal_id[:9]][gal_id][()]
        wave_list = self.sed_cat["meta"]["wave_list"][()]
        galaxy = _generate_rubinroman_galaxies(
            self.rng,
            survey,
            self._ou2024rubinroman_cat.slice(index, 1),
            f_sed,
            wave_list,
            self.bandpass,
        )

        return galaxy


def _generate_bulge_frac(rng):
    assert rng is not None, 'send rng to generate bulge fraction'
    return rng.uniform(low=0.0, high=1.0)


def _generate_rubinroman_galaxies(
    rng, survey, entry, f_sed, wave_list, bandpass, build_from_sed=False
):
    """
    Generate a GSObject from an entry
    from the OpenUniverse2024 Rubin-Roman catalog

    Parameters
    ---------
    rng: random number generator
    survey: object with survey parameters
    entry: pyarrow table (len=1)
        Galaxy properties, sliced from the pyarrow table
    f_sed: list[float]
        Flux density values of the SED
    wave_list: list[float]
        List of wavelengths corresponding to f_sed
    bandpass: galsim bandpass object
        Bandpass corresponding to this simulation
    build_from_sed: bool
        whether want to build galaxies out of SED (slow)
        if False, it generate a random bulge fraction

    Returns
    -------
    A galsim galaxy object: GSObject
    """

    band = survey.band
    sname = survey.name

    bulge_hlr = np.array(entry["spheroidHalfLightRadiusArcsec"])[0]
    disk_hlr = np.array(entry["diskHalfLightRadiusArcsec"])[0]
    disk_e1, disk_e2 = (
        np.array(entry["diskEllipticity1"])[0],
        np.array(entry["diskEllipticity2"])[0],
    )
    bulge_e1, bulge_e2 = (
        np.array(entry["spheroidEllipticity1"])[0],
        np.array(entry["spheroidEllipticity2"])[0],
    )
    mag = np.array(entry[sname + "_mag_" + band])[0]
    flux = survey.get_flux(mag)
    redshift = np.array(entry["redshift"][0])

    if build_from_sed:
        # set up bulge and disk for sed
        bulge_lookup = galsim.LookupTable(x=wave_list, f=f_sed[0])
        disk_lookup = galsim.LookupTable(x=wave_list, f=f_sed[1])
        knots_lookup = galsim.LookupTable(x=wave_list, f=f_sed[2])
        bulge_sed = galsim.SED(
            bulge_lookup, wave_type="Ang", flux_type="fnu", redshift=redshift
        )
        disk_sed = galsim.SED(
            disk_lookup, wave_type="Ang", flux_type="fnu", redshift=redshift
        )
        knots_sed = galsim.SED(
            knots_lookup, wave_type="Ang", flux_type="fnu", redshift=redshift
        )

        # light profile
        bulge = galsim.Sersic(4, half_light_radius=bulge_hlr).shear(
            e1=bulge_e1, e2=bulge_e2,
        )
        disk = galsim.Sersic(1, half_light_radius=disk_hlr).shear(
            e1=disk_e1, e2=disk_e2,
        )

        # Make galaxy object. Note that we are not drawing with knots,
        # so we add the knots' SED to the
        # disk SED to preserve the correct flux.
        gal = bulge * bulge_sed + disk * (disk_sed + knots_sed)
        gal = gal.withFlux(flux, bandpass)
    else:
        bulge_frac = _generate_bulge_frac(rng)
        bulge = galsim.Sersic(
            4, half_light_radius=bulge_hlr, flux=flux * bulge_frac
        ).shear(e1=bulge_e1, e2=bulge_e2)
        disk = galsim.Sersic(
            1, half_light_radius=disk_hlr, flux=flux * (1.0 - bulge_frac)
        ).shear(e1=disk_e1, e2=disk_e2)
        gal = bulge + disk
        gal = gal.withFlux(flux)
    return gal


def read_ou2024rubinroman_cat(
    select_observable=None,
    select_lower_limit=None,
    select_upper_limit=None,
):
    """
    Read the catalog from the cache, but update the position angles each time

    Parameters
    ----------
    select_observable: list[str] | str
        A list of observables to apply selection
    select_lower_limit: list[float] | ndarray[float]
        lower limits of the slection cuts
    select_upper_limit: list[float] | ndarray[float]
        upper limits of the slection cuts

    Returns
    -------
    array with fields
    """
    # galaxy catalog
    fname = os.path.join(
        os.environ.get("CATSIM_DIR", "."),
        "galaxy_10307.parquet",
    )
    # flux catalog (same length)
    fname_flux = os.path.join(
        os.environ.get("CATSIM_DIR", "."),
        "galaxy_flux_10307.parquet",
    )
    fname_sed = os.path.join(
        os.environ.get("CATSIM_DIR", "."),
        "galaxy_sed_10307.hdf5",
    )
    cat = cached_catalog_read(fname, format="parquet")
    cat_flux = cached_catalog_read(fname_flux, format="parquet")
    cat_sed = cached_catalog_read(fname_sed, format="h5py")

    # Merge the flux catalog to the galaxy catalog
    for nn in cat_flux.column_names:
        if nn != "galaxy_id":
            cat = cat.append_column(nn, cat_flux[nn])

    # Got this from TianQing ==> need to properly calculate them in the future
    zeropoint = {
        "u": 12.627880975839947,
        "g": 14.486360566315488,
        "r": 14.324886825965237,
        "i": 13.998598224053055,
        "z": 13.612736383992512,
        "y": 12.785119907991263,
    }
    roman_filters = roman.getBandpasses(AB_zeropoint=True)
    for band in ["W146", "R062", "Z087", "Y106", "J129", "H158", "F184", "K213"]:
        bandpass = roman_filters[band]
        zeropoint[band] = bandpass.zeropoint

    for filter_name in zeropoint.keys():
        if filter_name in "ugrizy":
            flux_filter = cat["lsst_flux_" + filter_name]
            mag_filter = pa.compute.add(
                pa.compute.multiply(-2.5, pa.compute.log10(flux_filter)),
                zeropoint[band],
            )
            cat = cat.append_column("lsst_mag_" + filter_name, mag_filter)
        else:
            flux_filter = cat["roman_flux_" + filter_name]
            mag_filter = pa.compute.add(
                pa.compute.multiply(-2.5, pa.compute.log10(flux_filter)),
                zeropoint[band],
            )
            cat = cat.append_column("roman_mag_" + filter_name, mag_filter)

    if select_observable is not None:
        select_observable = np.atleast_1d(select_observable)
        if not set(select_observable) < set(cat.column_names):
            raise ValueError("Selection observables not in the catalog columns")
        mask = np.ones(len(cat)).astype(bool)
        if select_lower_limit is not None:
            select_lower_limit = np.atleast_1d(select_lower_limit)
            assert len(select_observable) == len(select_lower_limit)
            for nn, ll in zip(select_observable, select_lower_limit):
                mask = mask & (cat[nn] > ll)
        if select_upper_limit is not None:
            select_upper_limit = np.atleast_1d(select_upper_limit)
            assert len(select_observable) == len(select_upper_limit)
            for nn, ul in zip(select_observable, select_upper_limit):
                mask = mask & (cat[nn] <= ul)
        cat = cat[mask]
    else:
        mask = None
    return cat, cat_sed, mask
