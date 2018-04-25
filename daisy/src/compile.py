from os.path import join as pjoin, basename
import sys
from distutils.ccompiler import new_compiler

sys.path.append(pjoin("..", "..", "compile_all.py"))

from compile_all import cmd

c_file = ["daisy.c"]
libs = ["m"]
flags = ["-O3"]

def main():
    c_basename = c_file[0].split(".")[0]
    
    ccomp = new_compiler()
    ccomp.compile(c_file, extra_postargs=flags)
    
    ccomp.link_executable([c_basename + ".o"],
                          c_basename,
                          libraries=libs,
                          extra_postargs=flags)
                           

if __name__ == "__main__":
    main()
