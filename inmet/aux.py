import numpy as np

import inmet as im
from ctypes import *


__all__ = ["test", "eval_poly"]


def empty(other, **kwargs):
    dtype = np.complex128 if np.iscomplexobj(other) else np.float64
    
    return np.empty(dtype=dtype, **kwargs)


inmet = im.CLib("inmet_aux")

test = inmet.wrap("test", [im.inarray])
eval_poly = inmet.wrap("eval_poly_c", [c_int, im.inarray, im.inarray,
                                       im.inarray, im.outarray])
