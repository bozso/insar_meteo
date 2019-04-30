import inmet as im
import numpy as np
import ctypes as ct


__all__ = ["test", "eval_poly"]


inmet = im.CLib("inmet_aux")

test = inmet.wrap("test", ["Array"])
eval_poly = inmet.wrap("eval_poly_c", ["PolyFitC", "Array", "Array"])
