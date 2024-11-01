import functools
import fitsio
import pyarrow.parquet as pq
import h5py


@functools.lru_cache(maxsize=8)
def cached_catalog_read(fname, format="fits"):
    if format == "fits":
        return fitsio.read(fname)
    elif format == "parquet":
        return pq.read_table(fname)
    elif format == "h5py":
        return h5py.File(fname, "r")
    else:
        raise Exception(
            "data format other than fits and parquet is not implemented"
        )
