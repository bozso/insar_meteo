#!/usr/bin/env python

from timeit import timeit


ntimes = 1000

setup = """
from inmet_aux import test
import numpy as np
asd = np.ones(shape=1000000)
"""

print("C-extension: %f s" % timeit("test(asd)", setup=setup, number=ntimes))


setup = """
import os
import numpy as np
asd = np.ones(shape=1000000)
from ctypes import c_size_t
import numpy.ctypeslib as ct
lib = ct.load_library("test", os.path.dirname('__file__'))
lib.test.argtypes = [ct.ndpointer(np.double, ndim=1, flags="aligned"), c_size_t]
lib.test.restype = None
"""

print("ctypes call: %f s" % timeit("lib.test(asd, len(asd))", setup=setup, number=ntimes))
