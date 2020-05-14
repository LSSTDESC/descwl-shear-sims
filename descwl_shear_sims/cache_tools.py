import functools
import fitsio


@functools.lru_cache(maxsize=8)
def cached_catalog_read(fname):
    return fitsio.read(fname)
