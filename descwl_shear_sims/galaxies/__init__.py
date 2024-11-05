from ..constants import SCALE
from ..layout import Layout
from .FixedGalaxy import (
    FixedGalaxyCatalog,
    FixedPairGalaxyCatalog,
    GalaxyCatalog, PairGalaxyCatalog,
    get_fixed_gal_config,
    DEFAULT_FIXED_GAL_CONFIG,
)
from .skyCatalog import OpenUniverse2024RubinRomanCatalog
from .WLDeblendGalaxy import WLDeblendGalaxyCatalog
from .utils import _prepare_rubinroman_catalog


def make_galaxy_catalog(
    *,
    rng,
    gal_type,
    coadd_dim=None,
    buff=0,
    pixel_scale=SCALE,
    layout=None,
    gal_config=None,
    sep=None,
):
    """
    rng: numpy.random.RandomState
        Numpy random state
    gal_type: string
        'fixed', 'varying', 'wldeblend' or 'ou2024rubinroman'
    coadd_dim: int
        Dimensions of coadd
    buff: int, optional
        Buffer around the edge where no objects are drawn.  Ignored for
        layout 'grid'.  Default 0.
    pixel_scale: float
        pixel scale in arcsec
    layout: string, optional
        'grid' or 'random'.  Ignored for gal_type "wldeblend", otherwise
        required.
    gal_config: dict or None
        Can be sent for fixed galaxy catalog.  See DEFAULT_FIXED_GAL_CONFIG
        for defaults mag, hlr and morph
    sep: float, optional
        Separation of pair in arcsec for layout='pair'
    """

    if (layout is None) and (
        gal_type in ['wldeblend', 'ou2024rubinroman']
    ):
        layout = 'random'

    if isinstance(layout, str):
        layout = Layout(
            layout_name=layout,
            coadd_dim=coadd_dim,
            buff=buff,
            pixel_scale=pixel_scale,
        )
    else:
        assert isinstance(layout, Layout)

    if layout.layout_name == 'pair':
        if sep is None:
            raise ValueError(
                f'send sep= for gal_type {gal_type} and layout {layout}'
            )
        gal_config = get_fixed_gal_config(config=gal_config)

        if gal_type in ['fixed', 'exp']:  # TODO remove exp
            cls = FixedPairGalaxyCatalog
        else:
            cls = PairGalaxyCatalog

        galaxy_catalog = cls(
            rng=rng,
            mag=gal_config['mag'],
            hlr=gal_config['hlr'],
            morph=gal_config['morph'],
            sep=sep,
        )
    else:
        if gal_type == 'wldeblend':
            galaxy_catalog = WLDeblendGalaxyCatalog(
                rng=rng,
                layout=layout,
            )
        elif gal_type == 'ou2024rubinroman':
            galaxy_catalog = OpenUniverse2024RubinRomanCatalog(
                rng=rng,
                layout=layout,
            )
        elif gal_type in ['fixed', 'varying', 'exp']:  # TODO remove exp
            gal_config = get_fixed_gal_config(config=gal_config)

            if gal_type == 'fixed':
                cls = FixedGalaxyCatalog
            else:
                cls = GalaxyCatalog

            galaxy_catalog = cls(
                rng=rng,
                mag=gal_config['mag'],
                hlr=gal_config['hlr'],
                morph=gal_config['morph'],
                layout=layout,
            )

        else:
            raise ValueError(f'bad gal_type "{gal_type}"')
    return galaxy_catalog


__all__ = ["make_galaxy_catalog", "DEFAULT_FIXED_GAL_CONFIG"]
