from numpy import ctypeslib as nct
import numpy as np
from ctypes import *
from os.path import dirname, realpath, join

filedir = dirname(realpath(__file__))

ia = nct.load_library("libinmet_aux", join(filedir, "..", "src"))

print(np.dtype(np.float32))

class Carray(Structure):
    _fields_ = [("type", c_int),
                ("ndim", c_ssize_t),
                ("ndata", c_ssize_t),
                ("shape", POINTER(c_ssize_t)), 
                ("strides", POINTER(c_ssize_t)),
                ("data", c_void_p)]


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


def npc(array):
    array = np.array(array)
    act = array.ctypes
    
    return Carray(type_conversion[array.dtype],
                  array.ndim, array.size, act.shape_as(c_size_t),
                  act.strides_as(c_size_t), act.data_as(c_void_p))
    

ia.test.argtypes = [POINTER(Carray)]
ia.test.restypes = c_int

ia.test(npc([1,2,3]))
