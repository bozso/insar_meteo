from numpy import ctypeslib as nct
from ctypes import *
from os.path import dirname, realpath, join


filedir = dirname(realpath(__file__))

ia = nct.load_library("libinmet", join(filedir, "..", "src"))
ia.inmet.restypes = c_int


def call_inmet(modname, *args):
    argv = [str(elem) for elem in args]
    n = len(argv)
    
    mem = (c_char_p * n)()

    for ii, elem in enumerate(argv):
        mem[ii] = elem.encode("ascii")

    
    ia.inmet(c_int(n), byref(mem))
    


def main():
    call_inmet("test")


if __name__ == "__main__":
    main()
