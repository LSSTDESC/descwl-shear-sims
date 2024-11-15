import warnings
from ..constants import (
    RANDOM_DENSITY,
    GRID_SPACING,
    HEX_SPACING,
    SCALE,
)
from .shifts import (
    get_pair_shifts,
    get_grid_shifts,
    get_hex_shifts,
    get_random_shifts,
    get_random_disk_shifts,
)
from ..wcs import make_coadd_dm_wcs
import numpy as np


class Layout(object):
    def __init__(
        self,
        layout_name,
        coadd_dim=None,
        buff=0.0,
        pixel_scale=SCALE,
    ):
        """
        Layout object to make position shifts for galaxy and star objects
        The scale of the layout is coadd_dim * pixel_scale

        Parameters
        ----------
        layout_name: string
            'grid', 'pair', 'hex', or 'random'
        coadd_dim: int
            Dimensions of final coadd
        buff: int, optional
            Buffer region where no objects will be drawn.  Default 0.
        pixel_scale: float
            pixel scale (arcsec)
        """
        self.pixel_scale = pixel_scale
        self.layout_name = layout_name
        if layout_name == 'random':
            if coadd_dim is None:
                raise ValueError("Please input `coadd_dim` for random layout")
            # need to calculate number of objects first this layout is random
            # in a square
            if (coadd_dim - 2*buff) < 2:
                warnings.warn("dim - 2*buff <= 2, force it to 2.")
                self.area = (2*pixel_scale/60)**2.  # [arcmin^2]
            else:
                # [arcmin^2]
                self.area = ((coadd_dim - 2*buff)*pixel_scale/60)**2
        elif layout_name == 'random_disk':
            if coadd_dim is None:
                raise ValueError(
                    "Please input `coadd_dim` for random_disk layout"
                )
            # need to calculate number of objects first
            # this layout_name is random in a circle
            if (coadd_dim - 2*buff) < 2:
                warnings.warn("dim - 2*buff <= 2, force it to 2.")
                radius = 2.*pixel_scale/60
                self.area = np.pi*radius**2
            else:
                radius = (coadd_dim/2. - buff)*pixel_scale/60
                self.area = np.pi*radius**2  # [arcmin^2]
        elif layout_name == "hex":
            self.area = 0
        elif layout_name == "grid":
            self.area = 0
        elif layout_name == "pair":
            return
        else:
            raise ValueError("layout_name can only be 'random', 'random_disk' \
                    'hex', 'grid' or 'pair'!")
        self.coadd_dim = coadd_dim
        self.buff = buff
        # How should we control xoff, yoff and paddings properly?
        # For now, I set it to zero (Tae)
        self.wcs, self.bbox = make_coadd_dm_wcs(
            coadd_dim,
            pixel_scale=pixel_scale,
        )
        return

    def get_shifts(
        self,
        rng,
        density=RANDOM_DENSITY,
        sep=None,
    ):
        """
        Make position shifts for objects. The position shifts

        rng: numpy.random.RandomState
            Numpy random state
        density: float, optional
            galaxy number density [/arcmin^2] ,default set to RANDOM_DENSITY
        sep: float, optional
            The separation in arcseconds for layout='pair'
        """

        if self.layout_name == 'pair':
            if sep is None:
                raise ValueError(f'send sep= for layout {self.layout_name}')
            shifts = get_pair_shifts(
                rng=rng,
                sep=sep,
                pixel_scale=self.pixel_scale
            )
        else:
            if self.layout_name == 'grid':
                shifts = get_grid_shifts(
                    rng=rng,
                    dim=self.coadd_dim,
                    buff=self.buff,
                    pixel_scale=self.pixel_scale,
                    spacing=GRID_SPACING,
                )
            elif self.layout_name == 'hex':
                shifts = get_hex_shifts(
                    rng=rng,
                    dim=self.coadd_dim,
                    buff=self.buff,
                    pixel_scale=self.pixel_scale,
                    spacing=HEX_SPACING,
                )
            elif self.layout_name == 'random':
                # area covered by objects
                if self.area <= 0:
                    raise ValueError(
                        f"nonpositive area for layout {self.layout_name}"
                    )
                if density != 0:
                    nobj_mean = max(self.area * density, 1)
                else:
                    nobj_mean = 0.0
                nobj = rng.poisson(nobj_mean)
                shifts = get_random_shifts(
                    rng=rng,
                    dim=self.coadd_dim,
                    buff=self.buff,
                    pixel_scale=self.pixel_scale,
                    size=nobj,
                )
            elif self.layout_name == 'random_disk':
                if self.area <= 0:
                    raise ValueError(
                        f"nonpositive area for layout {self.layout_name}"
                    )
                if density != 0:
                    nobj_mean = max(self.area * density, 1)
                else:
                    nobj_mean = 0.0
                nobj = rng.poisson(nobj_mean)
                shifts = get_random_disk_shifts(
                    rng=rng,
                    dim=self.coadd_dim,
                    buff=self.buff,
                    pixel_scale=self.pixel_scale,
                    size=nobj,
                )
            else:
                raise ValueError("bad layout: '%s'" % self.layout_name)
        return shifts
