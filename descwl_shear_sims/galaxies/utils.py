import os
import numpy as np
from ..cache_tools import cached_catalog_read
import galsim
import galsim.roman as roman
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def prepare_rubinroman_catalog():
    """
    prepare rubin+roman joint catalog including
    precalculated magnitudes from the flux+SED
    input catalog
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

    # Define galsim bandpasses to calculate magnitudes
    bandpasses = {}
    for band in "ugrizy":
        fname_bandpass = os.path.join(
            os.environ.get("CATSIM_DIR", "."),
            f"LSST_{band}.dat",
        )
        bandpasses[band] = galsim.Bandpass(
            fname_bandpass, wave_type="nm"
        ).withZeropoint("AB")
    for band in ["W146", "R062", "Z087", "Y106", "J129", "H158", "F184", "K213"]:
        filters = roman.getBandpasses(AB_zeropoint=True)
        bandpasses[band] = filters[band]

    bands = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "W146",
        "R062",
        "Z087",
        "Y106",
        "J129",
        "H158",
        "F184",
        "K213",
    ]
    surveys = [
        "lsst",
        "lsst",
        "lsst",
        "lsst",
        "lsst",
        "lsst",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
    ]

    # galaxy id for reading SED
    gal_id = np.array(cat["galaxy_id"])
    # wavelengths for the SED data
    wave_list = cat_sed["meta"]["wave_list"][()]

    # Arrays of magnitudes/bulge fractions for each band stacked
    mag_arr = np.zeros((len(surveys), len(cat)))
    bulge_frac_arr = np.zeros((len(surveys), len(cat)))
    for i in tqdm(range(len(cat))):
        mags, bulge_fracs = _calculate_mag_from_sed(
            cat.slice(i, 1),
            cat_sed["galaxy"][str(int(gal_id[i]) // 100000)][str(gal_id[i])][()],
            wave_list,
            bandpasses,
        )
        mag_arr[:, i] = mags
        bulge_frac_arr[:, i] = bulge_fracs

    for i in range(len(surveys)):
        cat = cat.append_column(
            surveys[i] + "_mag_" + bands[i],
            pa.array(mag_arr[i], type=pa.float64()),
        )
    for i in range(len(surveys)):
        cat = cat.append_column(
            surveys[i] + "_bulgefrac_" + bands[i],
            pa.array(bulge_frac_arr[i], type=pa.float64()),
        )

    fname_save = os.path.join(
        os.environ.get("CATSIM_DIR", "."),
        "galaxy_combined_10307.parquet",
    )
    pq.write_table(cat, fname_save)


def _calculate_mag_from_sed(
    entry,
    f_sed,
    wave_list,
    bandpasses,
):
    """
    calculate magnitude and bulge fraction from SED
    """

    # Read morphology and redshift data
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
    redshift = np.array(entry["redshift"])[0]

    # Setting up a galsim object to calcualte the bulge fraction
    # Note that we ignore the knots, and absorb the knot flux
    # proportionately into bulge and disk.
    bulge_lookup = galsim.LookupTable(x=wave_list, f=f_sed[0])
    disk_lookup = galsim.LookupTable(x=wave_list, f=f_sed[1])
    bulge_sed = galsim.SED(
        bulge_lookup, wave_type="Ang", flux_type="fnu", redshift=redshift
    )
    disk_sed = galsim.SED(
        disk_lookup, wave_type="Ang", flux_type="fnu", redshift=redshift
    )

    # light profile
    bulge = galsim.Sersic(4, half_light_radius=bulge_hlr).shear(
        e1=bulge_e1,
        e2=bulge_e2,
    )
    disk = galsim.Sersic(1, half_light_radius=disk_hlr).shear(
        e1=disk_e1,
        e2=disk_e2,
    )

    # Make galaxy object. Note that we are not drawing with knots,
    # so we add the knots' SED to the
    # disk SED to preserve the correct flux.
    gal_bulge = bulge * bulge_sed
    gal_disk = disk * disk_sed

    # Must follow this order
    bands = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "W146",
        "R062",
        "Z087",
        "Y106",
        "J129",
        "H158",
        "F184",
        "K213",
    ]
    surveys = [
        "lsst",
        "lsst",
        "lsst",
        "lsst",
        "lsst",
        "lsst",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
        "roman",
    ]
    mags = []
    bulge_fracs = []

    for band, survey in zip(bands, surveys):
        bandpass = bandpasses[band]
        flux = np.array(entry[survey + "_flux_" + band])[0]
        mag = -2.5 * np.log10(flux) + bandpass.zeropoint
        bulge_flux = gal_bulge.calculateFlux(bandpass)
        disk_flux = gal_disk.calculateFlux(bandpass)
        bulge_frac = bulge_flux / (bulge_flux + disk_flux)
        mags.append(mag)
        bulge_fracs.append(bulge_frac)

    return mags, bulge_fracs
