from os.path import join as pjoin, basename
import sys
from distutils.ccompiler import new_compiler

sys.path.append(pjoin("..", "..", "compile_all.py"))

from compile_all import compile_c

flags = ["-std=c99", "-lm"]
c_file = "daisy.c"
depend = pjoin("..", "..", "aux")

def main():
    c_basename = c_file.split(".")[0]

    ccomp = new_compiler()
    ccomp.compile([c_file, pjoin(depend, "aux_module.c")],
                  include_dirs=[depend], extra_postargs=flags)
    
    ccomp.link_executable([c_basename + ".o",
                           pjoin(depend, "aux_module.o")],
                           c_basename, extra_postargs=flags)
                           

if __name__ == "__main__":
    main()
