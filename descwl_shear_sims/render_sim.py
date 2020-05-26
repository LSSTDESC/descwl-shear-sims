"""Function for rendering simulated sets of galaxies.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
"""

import logging
from numba import njit
import numpy as np
import galsim

LOGGER = logging.getLogger(__name__)


def append_wcs_info_and_render_objs_with_psf_shear(
        *,
        objs, psf_function,
        wcs, img_dim, method, g1, g2, shear_scene,
        threshold,
        trim_stamps=True):
    """Render objects into a scene with some PSF function, shear, and WCS.

    Parameters
    ----------
    objs : list of dict
        The list of dicts with keys 'type', either 'galaxy' or 'star', 'dudv'
        the offset of the object in the u-v plane, and 'obj',
        the galsim GSObject to be rendered.  Stars are not sheared. Note that
        this function adds WCS info to the object in the fields 'overlaps' and
        'pos' giving whether or not the object overlaps with the image and its
        position in the image. These fields are lists that get appened to for each
        epioch in each band.
    psf_function : callable
        A callable with signature `psf_function(*, x, y)` that returns the
        PSF at a given location in the image.
    wcs : galsim.TanWCS
        The WCS function to use for the image.
    img_dim : int
        The size of the image in pixels.
    method : string
        The method used to render the image. This should usually be 'auto'
        unless you are doing something special.
    g1 : float
        The 1-component of the shear.
    g2 : float
        The 2-component of the shear.
    shear_scene : bool
        If True, the object positions and their shapes are sheared. Otherwise,
        only the object shapes are sheared.
    threshold: float
        For bright stars we will calculate the radius at which the profile
        reaches this threshold
    trim_stamps: bool
        If true, trim stamps larger than the input image to avoid huge
        ffts.  Default True.

    Returns
    -------
    se_image : galsim.ImageD
        The rendered image.
    """

    shear_mat = galsim.Shear(g1=g1, g2=g2).getMatrix()

    se_im = galsim.ImageD(
        nrow=img_dim,
        ncol=img_dim,
        xmin=0,
        ymin=0,
    )

    jac_wcs = wcs.jacobian(world_pos=wcs.center)
    center_image_pos = wcs.toImage(wcs.center)

    for obj_data in objs:

        obj = obj_data['obj']
        uv_offset = obj_data['dudv']

        du = uv_offset.x
        dv = uv_offset.y

        if obj_data['type'] == 'galaxy':
            # shear object and maybe position
            # this does not alter the original GSObject
            obj = obj.shear(g1=g1, g2=g2)
            if shear_scene:
                du, dv = np.dot(shear_mat, np.array([du, dv]))

        uv_pos = galsim.PositionD(x=du, y=dv)

        # deal with WCS stuff
        # we convert the uv position to xy using the jacobian
        # then from xy we go back to radec on the sphere (world_pos)
        # then we use the local WCS there to render the image
        pos = jac_wcs.toImage(uv_pos) + center_image_pos
        world_pos = wcs.toWorld(pos)
        local_wcs = wcs.local(world_pos=world_pos)

        # get the psf
        psf = psf_function(x=pos.x, y=pos.y)

        convolved_obj = galsim.Convolve(obj, psf)

        # draw with setup_only to get the image size
        _im = convolved_obj.drawImage(
            wcs=local_wcs,
            method=method,
            setup_only=True,
        ).array

        shape = _im.shape

        if trim_stamps:
            # to avoid "fft too big" errors from galsim
            if shape[0] > img_dim:
                shape = (img_dim, img_dim)

        assert shape[0] == shape[1]

        # now get location of the stamp
        x_ll = int(pos.x - (shape[1] - 1)/2)
        y_ll = int(pos.y - (shape[0] - 1)/2)

        # get the offset of the center
        dx = pos.x - (x_ll + (shape[1] - 1)/2)
        dy = pos.y - (y_ll + (shape[0] - 1)/2)

        offset = galsim.PositionD(x=dx, y=dy)

        # draw and set the proper origin
        stamp = convolved_obj.drawImage(
            nx=shape[1],
            ny=shape[0],
            wcs=local_wcs,
            offset=offset,
            method=method,
        )

        radius = get_mask_radius(
            obj_data=obj_data,
            stamp=stamp.array,
            offset=(offset.y, offset.x),
            threshold=threshold,
        )

        stamp.setOrigin(x_ll, y_ll)

        # intersect and add to total image
        overlap = stamp.bounds & se_im.bounds
        oshape = overlap.numpyShape()
        if oshape[0]*oshape[1] > 0:
            overlaps = True
            se_im[overlap] += stamp[overlap]
        else:
            overlaps = False

        if 'overlaps' not in obj_data:
            obj_data['overlaps'] = []
            obj_data['overlap'] = []
            obj_data['stamp'] = []

        obj_data['overlaps'].append(overlaps)
        obj_data['overlap'].append(overlap)

        if 'pos' not in obj_data:
            obj_data['pos'] = []
        obj_data['pos'].append(pos)

        if 'radius' not in obj_data:
            obj_data['radius'] = []
        obj_data['radius'].append(radius)

        if obj_data['type'] == 'star':
            obj_data['stamp'].append(stamp)
        else:
            obj_data['stamp'].append(None)

    return se_im


def get_mask_radius(*, obj_data, stamp, offset, threshold, mag_thresh=18):
    """
    get the radius at which the profile drops to frac*noise, or 0.0
    for faint stars or galaxies

    Parameters
    ----------
    obj_data: dict
        Object data.
    stamp: 2d array
        The stamp
    offset: tuple
        Offset in stamp (row_offset, col_offset)
    threshold: float
        Radius goes out to this level
    mag_thresh: float
        Magnitude threshold for doing the calculation.  Default 18

    Returns
    -------
    radius: float
        The radius if the object is a star and brighter than minmag,
        otherwise 0.0
    """

    radius = 0.0

    if obj_data['type'] == 'star' and obj_data['mag'] < mag_thresh:
        radius = calculate_mask_radius(
            stamp=stamp,
            offset=offset,
            threshold=threshold,
        )

    return radius


@njit
def calculate_mask_radius(*, stamp, offset, threshold):
    """
    get the radius at which the profile drops to frac*noise.

    Parameters
    ----------
    obj_data: dict
        Object data.
    stamp: 2d array
        The stamp
    offset: tuple
        Offset in stamp (row_offset, col_offset)
    threshold: float
        Radius goes out to this level
    mag_thresh: float
        Magnitude threshold for doing the calculation.  Default 18

    Returns
    -------
    radius: float
        The radius
    """

    nrows, ncols = stamp.shape

    row_offset, col_offset = offset

    rowcen = int((nrows-1.0)/2.0 + row_offset)
    colcen = int((ncols-1.0)/2.0 + col_offset)

    radius2 = 0.0

    for row in range(nrows):
        row2 = (rowcen - row)**2

        for col in range(ncols):
            col2 = (colcen - col)**2

            tradius2 = row2 + col2
            if tradius2 < radius2:
                # we are already within a previously calculated radius
                continue

            val = stamp[row, col]
            if val > threshold:
                radius2 = tradius2

    radius = np.sqrt(radius2)
    return radius
