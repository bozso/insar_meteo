from numpy import ctypeslib as nct
import numpy as np
from ctypes import *
from os.path import dirname, realpath, join

filedir = dirname(realpath(__file__))

ia = nct.load_library("libinmet_aux", join(filedir, "..", "src", "build"))

c_idx = c_long
c_idx_p = POINTER(c_idx)


type_conversion = {
    np.dtype(np.bool_)       : 1, # byte
    np.dtype(np.int_)        : 2, # C long
    np.dtype(np.intc)        : 3, # C int
    np.dtype(np.intp)        : 4, # C ssize_t

    np.dtype(np.int8)        : 5,
    np.dtype(np.int16)       : 6,
    np.dtype(np.int32)       : 7,
    np.dtype(np.int64)       : 8,

    np.dtype(np.uint8)        : 9,
    np.dtype(np.uint16)       : 10,
    np.dtype(np.uint32)       : 11,
    np.dtype(np.uint64)       : 12,

    np.dtype(np.float32)     : 14,
    np.dtype(np.float64)     : 13,

    np.dtype(np.complex64)   : 15,
    np.dtype(np.complex128)  : 16
}

class Carray(Structure):
    _fields_ = [("type", c_int),
                ("ndim", c_idx),
                ("ndata", c_idx),
                ("datasize", c_idx),
                ("shape", c_idx_p), 
                ("strides", c_idx_p),
                ("data", c_char_p)]


def npc(array):
    array = np.array(array)
    act = array.ctypes
    
    print(array.ndim)
    
    return Carray(c_int(type_conversion[array.dtype]),
                  c_idx(array.ndim), c_idx(array.size), c_idx(array.itemsize),
                  act.shape_as(c_idx), act.strides_as(c_idx),
                  act.data_as(c_char_p))
    

ia.test.argtypes = [POINTER(Carray)]
ia.test.restypes = c_int

ia.test(npc([1,2,3]))
