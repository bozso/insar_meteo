from os.path import join as pjoin, basename
import sys
from distutils.ccompiler import new_compiler

sys.path.append(pjoin("..", "..", "compile_all.py"))

from compile_all import cmd

flags = ["-std=c99", "-static"]
c_file = "daisy.c"
lib_dirs = ["/home/istvan/progs/gsl/lib"]
libs = ["m", "gsl", "gslcblas"]
auxdir = pjoin("..", "..", "aux")
depend = [auxdir, "/home/istvan/progs/gsl/include"]

macros = [("GSL_C99_INLINE", None), ("GSL_RANGE_CHECK_OFF", None)]

def main():
    c_basename = c_file.split(".")[0]
    
    ccomp = new_compiler()
    ccomp.compile([c_file, pjoin(auxdir, "aux_module.c")],
                  include_dirs=depend, extra_postargs=flags,
                  macros=macros)
    
    ccomp.link_executable([c_basename + ".o",
                           pjoin(auxdir, "aux_module.o")],
                           c_basename,
                           extra_postargs=flags,
                           libraries=libs,
                           library_dirs=lib_dirs)
                           

if __name__ == "__main__":
    main()
