import numpy as np
from numpy import ctypeslib as nct

from distutils.ccompiler import new_compiler
from ctypes import *
from os.path import dirname, realpath, join

filedir = dirname(realpath(__file__))

fpath = join(filedir, "..", "src", "build")
ia = nct.load_library("libinmet_aux", )

c_idx = c_long
c_idx_p = POINTER(c_idx)


class Carray(Structure):
    _fields_ = [("type", c_int),
                ("is_numpy", c_int),
                ("ndim", c_idx),
                ("ndata", c_idx),
                ("datasize", c_idx),
                ("shape", c_idx_p), 
                ("strides", c_idx_p),
                ("data", c_char_p)]


class CLib(object):
    lib_filename = new_compiler().library_filename
    
    def __init__(self, name, path="."):
            self.path = join(path, lib_filename(name, lib_type="shared"))
            self.lib = CDLL(self.path)

    
    def wrap(self, funcname, argtypes, restype=c_int):
        ''' Simplify wrapping ctypes functions '''
        func = self.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        
        
        def fun(*args):
            ret = func(args)
            
            if ret == -1:
                raise RuntimeError("Library function returned with -1")
        
        return fun


inmet = CLib("inmet_aux", fpath)

inmet.test = inmet.wrap("test", [POINTER(Carray)])


type_conversion = {
    np.dtype(np.int_)        : 1, # C long
    np.dtype(np.intc)        : 2, # C int
    np.dtype(np.intp)        : 3, # C ssize_t

    np.dtype(np.int8)        : 4,
    np.dtype(np.int16)       : 5,
    np.dtype(np.int32)       : 6,
    np.dtype(np.int64)       : 7,

    np.dtype(np.uint8)        : 8,
    np.dtype(np.uint16)       : 9,
    np.dtype(np.uint32)       : 10,
    np.dtype(np.uint64)       : 11,

    np.dtype(np.float32)     : 12,
    np.dtype(np.float64)     : 13,

    np.dtype(np.complex64)   : 14,
    np.dtype(np.complex128)  : 15
}



def npc(array, **kwargs):
    array = np.array(array, **kwargs)
    act = array.ctypes
    
    return Carray(type_conversion[array.dtype], 1,
                  c_idx(array.ndim), c_idx(array.size), c_idx(array.itemsize),
                  act.shape_as(c_idx), act.strides_as(c_idx),
                  act.data_as(c_char_p))
    

def main():
    _a1 = np.array([1 for ii in range(128)], dtype=np.float64)
    _a2 = np.array([1 for ii in range(129)], dtype=np.float64)
    
    a1, a2 = npc(_a1), npc(_a2)
    
    inmet.test(a1)
    # ia.test(a2)
    
    return 0


if __name__ == "__main__":
    main()
