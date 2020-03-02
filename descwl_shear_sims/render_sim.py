"""Function for rendering simulated sets of galaxies.

Copied from https://github.com/beckermr/metadetect-coadding-sims under BSD
"""

import logging

import numpy as np
import galsim

LOGGER = logging.getLogger(__name__)


def render_objs_with_psf_shear(
        *,
        objs, psf_function, uv_offsets,
        wcs, img_dim, method, g1, g2, shear_scene):
    """Render objects into a scene with some PSF function, shear, and WCS.

    Parameters
    ----------
    objs : list of dict
        The list of dicts with keys type, either 'galaxy' or 'star'
        and obj, the galsim GSObject to be rendered.  Stars are not
        sheared.
    psf_function : callable
        A callable with signature `psf_function(*, x, y)` that returns the
        PSF at a given location in the image.
    uv_offsets : list of galsim.PositionD
        The offset from the center of the image for each object in u,v. The
        units of u,v are usualy arcseconds.
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

    Returns
    -------
    se_image : galsim.ImageD
        The rendered image.
    overlap_info: list of dict
        List of dict for each object containing
            'overlaps': whether or not the stamp for the object overlapped
            the image
            'pos': the position of the object center in the image
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

    overlap_info = []
    for obj_data, uv_offset, in zip(objs, uv_offsets):

        obj = obj_data['obj']

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

        # draw with setup_only to get the image size
        _im = galsim.Convolve(obj, psf).drawImage(
            wcs=local_wcs,
            method=method,
            setup_only=True).array
        shape = _im.shape

        if obj_data['type'] == 'star':
            # avoid stamp-edge issues
            if obj_data['mag'] < 15:
                shape = [s*3 for s in shape]
            elif obj_data['mag'] < 18:
                shape = [s*2 for s in shape]

        assert shape[0] == shape[1]

        # now get location of the stamp
        x_ll = int(pos.x - (shape[1] - 1)/2)
        y_ll = int(pos.y - (shape[0] - 1)/2)

        # get the offset of the center
        dx = pos.x - (x_ll + (shape[1] - 1)/2)
        dy = pos.y - (y_ll + (shape[0] - 1)/2)

        # draw and set the proper origin
        stamp = galsim.Convolve(obj, psf).drawImage(
            nx=shape[1],
            ny=shape[0],
            wcs=local_wcs,
            offset=galsim.PositionD(x=dx, y=dy),
            method=method)
        stamp.setOrigin(x_ll, y_ll)

        # intersect and add to total image
        overlap = stamp.bounds & se_im.bounds
        oshape = overlap.numpyShape()
        if oshape[0]*oshape[1] > 0:
            overlaps = True
            se_im[overlap] += stamp[overlap]
        else:
            overlaps = False

        overlap_info.append({
            'overlaps': overlaps,
            'pos': pos,
        })

    return se_im, overlap_info
