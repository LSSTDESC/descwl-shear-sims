import functools
import fitsio


@functools.lru_cache(maxsize=8)
def cached_catalog_read(fname, format="fits"):
    return fitsio.read(fname)
