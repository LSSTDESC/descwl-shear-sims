"""Function for rendering simulated sets of galaxies.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
"""

import logging

import numpy as np
import galsim

LOGGER = logging.getLogger(__name__)


def append_wcs_info_and_render_objs_with_psf_shear(
        *,
        objs, psf_function,
        wcs, img_dim, method, g1, g2, shear_scene,
        expand_star_stamps=True,
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
    expand_star_stamps: bool
        If True, expand bright star stamps to avoid rendering issues, default
        True
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

        if expand_star_stamps and obj_data['type'] == 'star':
            # avoid stamp-edge issues
            if obj_data['mag'] < 15:
                shape = [s*4 for s in shape]
            elif obj_data['mag'] < 18:
                shape = [s*3 for s in shape]

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

        # draw and set the proper origin
        stamp = convolved_obj.drawImage(
            nx=shape[1],
            ny=shape[0],
            wcs=local_wcs,
            offset=galsim.PositionD(x=dx, y=dy),
            method=method,
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
        obj_data['overlaps'].append(overlaps)

        if 'pos' not in obj_data:
            obj_data['pos'] = []
        obj_data['pos'].append(pos)

    return se_im
