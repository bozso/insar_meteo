from distutils.ccompiler import new_compiler
from ctypes import *
from os.path import dirname, realpath, join


filedir = dirname(realpath(__file__))


def get_library(libname, searchdir):
    lib_filename = new_compiler().library_filename
    
    libpath = join(searchdir, lib_filename(libname, lib_type="shared"))

    return CDLL(libpath)


lb = get_library("lab", join(filedir, "..", "src", "build"))


memptr = POINTER(c_ubyte)

class Memory(Structure):
    _fields_ = (
        ("memory", memptr),
        ("_size", c_long)
    )

    def __del__(self):
        lb.dtor_memory(byref(self))


class DataFile(Structure):
    _fields_ = (
        ("filetype", c_int),
        ("dtypes", POINTER(c_int)),
        ("ntypes", c_long),
        ("recsize", c_long),
        ("nio", c_long),
        ("file", memptr),
        ("mem", Memory),
        ("buffer", memptr),
        ("offsets", POINTER(c_long)),
    )

    def __del__(self):
        lb.dtor_datafile(byref(self))


def main():
    DataFile()
    return 0


if __name__ == "__main__":
    main()
