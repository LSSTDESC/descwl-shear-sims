def get_objlist(*, galaxy_catalog, survey, star_catalog=None, noise=None):
    """
    get the objlist and shifts, possibly combining the galaxy catalog
    with a star catalog

    Parameters
    ----------
    galaxy_catalog: catalog
        e.g. WLDeblendGalaxyCatalog
    survey: descwl Survey
        For the appropriate band
    star_catalog: catalog
        e.g. StarCatalog
    noise: float
        Needed for star catalog

    Returns
    -------
    objlist, shifts
        objlist is a list of galsim GSObject with transformations applied. Shifts
        is an array with fields dx and dy for each object
    """
    gal_res = galaxy_catalog.get_objlist(
        survey=survey,
    )

    if star_catalog is not None:
        assert noise is not None
        star_res = star_catalog.get_objlist(
            survey=survey, noise=noise,
        )

    else:
        star_res = {
            "star_objlist": None,
            "star_shifts": None,
            "bright_objlist": None,
            "bright_shifts": None,
            "bright_mags": None,
        }
    gal_res.update(star_res)
    return gal_res
