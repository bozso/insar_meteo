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
                          c_basename,
                          libraries=libs,
                          extra_postargs=flags)
    
    dest = pjoin("..", "bin", "daisy")
    
    # remove prevoius executable
    if isfile(dest): remove(dest)
    move(c_basename, dest)

if __name__ == "__main__":
    main()
