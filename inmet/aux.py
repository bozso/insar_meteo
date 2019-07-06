import numpy as np

import inmet as im
from ctypes import *


__all__ = {
    "empty",
    "is_cpx",
    "set_ellipsoid",
    "print_ellipsoid",
    "aaa",
    "lib"
}


def is_cpx(obj):
    return np.iscomplexobj(obj)


def empty(other, **kwargs):
    dtype = np.complex128 if np.iscomplexobj(other) else np.float64
    
    return np.empty(dtype=dtype, **kwargs)


class Ellipsoid(im.CStruct):
    _fields_ = [
        ("a", c_double),
        ("b", c_double),
        ("e2", c_double)
    ]
    
    
    def __init__(self, a, b):
        self.a, self.b = a, b
        
        a2 = a * a

        self.e2 = (a2 - b * b) / a2


lib = im.CLib("inmet_aux")

aaa = lib.wrap("aaa", [im.inarray])
set_ell = lib.wrap("set_ellipsoid", [Ellipsoid], restype=None)
print_ellipsoid = lib.wrap("print_ellipsoid", [], restype=None)



class Memory(im.CStruct):
    # __get_mem = inmet.wrap("get_memory", [c_size_t], restype=c_void_p)
    # __del_mem = inmet.wrap("release_memory", [c_void_p], restype=None)
    
    _fields_ = [
        ("size", c_size_t),
        ("ptr", c_void_p)
    ]
    
    
    def __init__(self, size):
        self.size, self.ptr = size, None
        ptr = Memory.__get_mem(size)
        
        if ptr is None:
            raise RuntimeError("AAA")
        
        self.ptr = ptr
    
    
    def __del__(self):
        if self.ptr is not None:
            Memory.__del_mem(self.ptr)
    

ellipsoids = {
    "WGS84": Ellipsoid(6378137.0, 6356752.3142)
}


def set_ellipsoid(name=None, a=None, b=None):
    if name not in ellipsoids:
        ellipsoids[name] = Ellipsoid(a, b)
    
    set_ell(ellipsoids[name])
