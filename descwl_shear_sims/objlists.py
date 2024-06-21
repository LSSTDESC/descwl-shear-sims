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
    objlist, shifts, redshifts, indexes = galaxy_catalog.get_objlist(
        survey=survey,
    )

    if star_catalog is not None:
        assert noise is not None
        res = star_catalog.get_objlist(
            survey=survey, noise=noise,
        )
        sobjlist, sshifts, bright_objlist, bright_shifts, bright_mags = res

    else:
        sobjlist = None
        sshifts = None
        bright_objlist = None
        bright_shifts = None
        bright_mags = None

    return {
        'objlist': objlist,
        'shifts': shifts,
        'redshifts': redshifts,
        'indexes': indexes,
        'star_objlist': sobjlist,
        'star_shifts': sshifts,
        'bright_objlist': bright_objlist,
        'bright_shifts': bright_shifts,
        'bright_mags': bright_mags,
    }
