#!/usr/bin/env python

from os.path import join as pjoin, isfile
from os import remove
from shutil import move
from distutils.ccompiler import new_compiler

c_file = ["daisy.c"]
libs = ["m"]
flags = ["-O3"]

def main():
    c_basename = c_file[0].split(".")[0]
    
    ccomp = new_compiler()
    ccomp.compile(c_file, extra_postargs=flags)
    
    ccomp.link_executable([c_basename + ".o"],
                          pjoin("..", "bin", c_basename),
                          libraries=libs,
                          extra_postargs=flags)
    
if __name__ == "__main__":
    main()
