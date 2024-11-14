import os
import galsim
import numpy as np

from ..constants import SCALE
from ..layout import Layout
from ..cache_tools import cached_catalog_read


class OpenUniverse2024RubinRomanCatalog(object):
    """
    Diffsky galaxy catalog from OpenUniverse2024 Rubin-Roman input galaxies
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

        self._ou2024rubinroman_cat = read_ou2024rubinroman_cat(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

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

        self.gal_ids = np.array(self._ou2024rubinroman_cat["galaxy_id"])

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
            redshifts.append(self._ou2024rubinroman_cat["redshift"][index].as_py())

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

    def _resize_dimension(
        self, *, new_coadd_dim, new_buff, new_pixel_scale, new_layout="random"
    ):
        """
        resize the pixel scale,
        preserving all the galaxy properties

        Parameters
        ----------
        new_coadd_dim: int
            new coadd dimension

        new_buff: int
            new buffer length

        new_pixel_scale: float
            new pixel scale
        """
        self.layout = Layout(
            new_layout, new_coadd_dim, new_buff, new_pixel_scale,
        )

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
        entry = self._ou2024rubinroman_cat.slice(index, 1)

        galaxy = _generate_rubinroman_galaxies(
            self.rng,
            survey=survey,
            entry=entry,
        )
        return galaxy


def _generate_rubinroman_galaxies(
    rng,
    *,
    survey,
    entry,
    f_sed=None,
    wave_list=None,
    bandpass=None,
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

    Returns
    -------
    A galsim galaxy object: GSObject
    """

    band = survey.band
    sname = survey.name

    bulge_hlr = np.array(entry["spheroidHalfLightRadiusArcsec"])[0]
    disk_hlr = np.array(entry["diskHalfLightRadiusArcsec"])[0]
    # The ellipticity in the catalog is shear ellipticity
    # e = 1-q / 1+q
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

    bulge_frac = np.array(entry[sname + "_bulgefrac_" + band])[0]
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
        "rubinroman_nside32_10307.parquet",
    )
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            "Cannot find 'rubinroman_nside32_10307.parquet'",
            "Please donwload it from and place it under $CATSIM_DIR"
        )

    cat = cached_catalog_read(fname, format="parquet")
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

    return cat
