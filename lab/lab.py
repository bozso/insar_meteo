import os

from tempfile import _get_default_tempdir, _get_candidate_names
from distutils.ccompiler import new_compiler
from os.path import dirname, realpath, join, isfile
from itertools import accumulate

from ctypes import *

filedir = dirname(realpath(__file__))
tmpdir = _get_default_tempdir()


def get_library(libname, searchdir):
    lib_filename = new_compiler().library_filename
    
    libpath = join(searchdir, lib_filename(libname, lib_type="shared"))

    return CDLL(libpath)


lb = get_library("lab", join(filedir, "..", "src", "build"))
im = get_library("inmet", join(filedir, "..", "src", "build"))

lb.is_big_endian.restype = c_int
lb.is_big_endian.argtype = None

lb.dtype_size.restype = c_long
lb.dtype_size.argtype = c_long

big_end = c_int(lb.is_big_endian())

memptr = POINTER(c_char)
c_idx = c_long
c_idx_p = POINTER(c_long)
c_int_p = POINTER(c_int)


class _FileInfo(Structure):
    _fields_ = (
        ("path", c_char_p),
        ("offsets", c_idx_p),
        ("ntypes", c_idx),
        ("recsize", c_idx),
        ("dtypes", c_int_p),
        ("filetype", c_int),
        ("endswap", c_int)
    )


class FileInfo(object):
    filetype2int = {
        "Unknown" : 0,
        "Array" : 1,
        "Records" : 2
    }

    dtype2int = {
        "Unknown"    : 0,
        "Bool"       : 1,
        "Int"        : 2,
        "Long"       : 3,
        "Size_t"     : 4,
        "Int8"       : 5,
        "Int16"      : 6,
        "Int32"      : 7,
        "Int64"      : 8,
        "UInt8"      : 9,
        "UInt16"     : 10,
        "UInt32"     : 11,
        "UInt64"     : 12,
        "Float32"    : 13,
        "Float64"    : 14,
        "Complex64"  : 15,
        "Complex128" : 16
    }

    def __init__(self, filetype, dtypes, path=None, keep=False,
                 endian="native"):

        self.keep, self.info = keep, None
        ntypes = len(dtypes)
        
        if filetype is "Array":
            assert ntypes == 1, "Array one dtype needed!"

        if path is None:
            _path = join(tmpdir, next(_get_candidate_names()))
            path = bytes(_path, "ascii")
        
        filetype = FileInfo.filetype2int[filetype]
        dtypes = (c_int * ntypes)(*(FileInfo.dtype2int[elem] for elem in dtypes))
        
        sizes = tuple(lb.dtype_size(dtypes[ii]) for ii in range(ntypes))
        
        offsets = (c_idx * (ntypes))(0, *accumulate(sizes[:ntypes - 1]))

        
        if endian == "native":
            swap = 0
        elif endian == "big":
            if big_end:
                swap = 0
            else:
                swap = 1
        elif endian == "little":
            if big_end:
                swap = 1
            else:
                swap = 0
        else:
            raise ValueError('endian should either be "big", "little" '
                             'or "native"')
        
        
        self.info = _FileInfo(path, offsets, ntypes, sum(sizes),
                              dtypes, filetype, swap)

        
    def ptr(self):
        return byref(self.info)    

        
    def __del__(self):
        if not self.keep and self.info is not None \
        and isfile(self.info.path):
            os.remove(self.info.path.decode("ascii"))
        
        

def open(name):
    path = bytes(workspace.get(name, "path"), "ascii")
    ntypes = workspace.getint(name, "ntypes")
    recsize = workspace.getint(name, "recsize")
    filetype = workspace.get(name, "filetype")
    endian = workspace.get(name, "endian")


def save(info, name, path):
    pass


def main():
    a = FileInfo("Records", ["Float32", "Float32", "Float32", "Complex128"])
    
    im.test(a.ptr())
    
    
    return 0


if __name__ == "__main__":
    main()
