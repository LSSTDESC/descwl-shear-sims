import galsim


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
    objlist, shifts = galaxy_catalog.get_objlist(survey=survey)

    if star_catalog is not None:
        assert noise is not None
        res = star_catalog.get_objlist(
            survey=survey, noise=noise,
        )
        sobjlist, sshifts, bright_objlist, bright_shifts, bright_mags = res

        # objlist = objlist + sobjlist
        # shifts = np.hstack((shifts, sshifts))
    else:
        sobjlist = None
        sshifts = None
        bright_objlist = None
        bright_shifts = None
        bright_mags = None

    return {
        'objlist': objlist,
        'shifts': shifts,
        'star_objlist': sobjlist,
        'star_shifts': sshifts,
        'bright_objlist': bright_objlist,
        'bright_shifts': bright_shifts,
        'bright_mags': bright_mags,
    }


def get_convolved_objects(*, objlist, psf, shifts, se_wcs, se_origin):
    """
    get a list of convolved objects.  This code is used for bright stars only.

    Parameters
    ----------
    objlist: list of GSObject
        The list of objects to convolve
    psf: GSObject or PowerSpectrumPSF
        The PSF for convolution
    shifts: list of shifts for each object
        Only used for the spatially variable power spectrum psf
    se_wcs: galsim WCS
        Only used for the spatially variable power specrum psf
    se_origin: galsim.PositionD
        Origin, shifts are relative to this origin

    Returns
    -------
    list of convolved GSObject
    """
    if isinstance(psf, galsim.GSObject):
        convolved_objects = [galsim.Convolve(obj, psf) for obj in objlist]
        psf_gsobj = psf
    else:
        convolved_objects = get_convolved_objlist_variable_psf(
            objlist=objlist,
            shifts=shifts,
            psf=psf,
            wcs=se_wcs,
            origin=se_origin,
        )
        psf_gsobj = psf.getPSF(se_origin)

    return convolved_objects, psf_gsobj


def get_convolved_objlist_variable_psf(
    *,
    objlist,
    shifts,
    psf,
    wcs,
    origin,  # pixel origin
):
    """
    Get a list of psf convolved objects for a variable psf

    Parameters
    ----------
    objlist: list
        List of GSObject
    shifts: array
        Array with fields dx and dy, which are du, dv offsets
        in sky coords.
    psf: PowerSpectrumPSF
        See ps_psf
    wcs: galsim wcs
        For the SE image
    origin: galsim.PositionD
        Origin of SE image (with offset included)
    """

    jac_wcs = wcs.jacobian(world_pos=wcs.center)

    new_objlist = []
    for i, obj in enumerate(objlist):
        shift_pos = galsim.PositionD(
            x=shifts['dx'][i],
            y=shifts['dy'][i],
        )
        pos = jac_wcs.toImage(shift_pos) + origin

        psf_gsobj = psf.getPSF(pos)

        obj = galsim.Convolve(obj, psf_gsobj)

        new_objlist.append(obj)

    return new_objlist
