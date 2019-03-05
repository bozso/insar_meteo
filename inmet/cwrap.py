from numpy import ctypeslib as nct
from ctypes import *
from os.path import dirname, realpath, join


filedir = dirname(realpath(__file__))

ia = nct.load_library("libinmet_aux", join(filedir, "..", "src")).inmet_aux


def main():
    ia.restypes = c_int
    
    argv = "test".split()
    n = len(argv)
    
    mem = (c_char_p * n)()
    
    for ii, elem in enumerate(argv):
        mem[ii] = elem.encode("ascii")
    
    ia(c_int(n), byref(mem))


if __name__ == "__main__":
    main()
