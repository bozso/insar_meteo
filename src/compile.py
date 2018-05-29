from os.path import join as pjoin, isfile
from distutils.ccompiler import new_compiler
from sys import argv

c_file = [argv[1]]
libs = ["m"]
flags = ["-std=c99", "-O3"]

def main():
    c_basename = c_file[0].split(".")[0]
    
    ccomp = new_compiler()
    ccomp.compile(c_file, extra_postargs=flags)
    ccomp.compile(["insar.c"], extra_postargs=flags)
    
    ccomp.link_executable([c_basename + ".o", "insar.o"],
                          pjoin("..", "bin", c_basename),
                          libraries=libs,
                          extra_postargs=flags)
    
if __name__ == "__main__":
    main()
