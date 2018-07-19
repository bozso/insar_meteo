from os.path import join as pjoin
from inmet.cwrap import cmd
from distutils.ccompiler import new_compiler

f_file = ["inmet.f95"]
libs = ["gfortran"]
lib_dirs = ["/home/istvan/miniconda3/lib"]
#flags = ["-std=c99", "-Ofast", "-march=native", "-ffast-math"]
flags = None

def main():
    
    f_basename = f_file[0].split(".")[0]
    
    
    comp = new_compiler()
    exe_name = comp.executable_filename(f_basename)
    
    # trick the compiler into thinking fortran files are c files
    comp.src_extensions.append(".f95")
    
    comp.compile(f_file)
    comp.compile(["main_functions.f95"])
    
    comp.link_executable([f_basename + ".o", "main_functions.o"],
                         pjoin("..", "bin", exe_name),
                         libraries=libs, library_dirs=lib_dirs,
                         extra_postargs=flags)

if __name__ == "__main__":
    main()

