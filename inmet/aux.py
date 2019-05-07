import numpy as np

import inmet as im


__all__ = ["test", "eval_poly"]


inmet = im.CLib("inmet_aux")

test = inmet.wrap("test", [im.inarray])
eval_poly = inmet.wrap("eval_poly_c", [im.PolyFit, im.inarray, im.outarray])
