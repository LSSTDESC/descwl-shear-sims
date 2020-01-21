import numpy as np

# default mask bits from the stack
BAD_COLUMN = np.int64(2**0)
COSMIC_RAY = np.int64(2**3)
EDGE = np.int64(2**4)
