from distutils.ccompiler import new_compiler
from os.path import basename, join
from glob import iglob
from os import remove
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def compile_project(c_file, *adds, macros=None, flags=None, inc_dirs=None,
                    lib_dirs=None, libs=None, outdir="."):
    
    c_basename = basename(c_file).split(".")[0]

    sources = [c_file]
    sources.extend(adds)
    
    ccomp = new_compiler()
    obj = [ccomp.compile([source], extra_postargs=flags, include_dirs=inc_dirs,
                         macros=macros)[0] for source in sources]
    
    ccomp.link_executable(obj, join(outdir, c_basename), libraries=libs,
                          library_dirs=lib_dirs, extra_postargs=flags)


def compile_exe(obj, *objects, macros=None, flags=None, inc_dirs=None,
                lib_dirs=None, libs=None, outdir="."):
    
    objs = [obj]
    objs.extend(objects)
    
    ccomp = new_compiler()
    
    ccomp.link_executable(obj, join(outdir, c_basename), libraries=libs,
                          library_dirs=lib_dirs, extra_postargs=flags)


def compile_object(*sources, macros=None, flags=None, inc_dirs=None):
    
    ccomp = new_compiler()
    
    return [ccomp.compile([source], extra_postargs=flags, include_dirs=inc_dirs,
                          macros=macros)[0] for source in sources]

def parse_args():
    
    ap = ArgumentParser(description=__doc__, formatter_class=
                        ArgumentDefaultsHelpFormatter)
    
    ap.add_argument(
        "--clean",
        action="store_true",
        help="If defined the program will clean the object files after "
             "compilation.")
    
    return ap.parse_args()

def main():

    args = parse_args()
    
    flags = ["-O3", "-march=native"]
    
    compile_project("daisy.c", outdir=join("..", "bin"), libs=["m"],
                    flags=flags)

    if args.clean:
        print("\nCleaning up object file.", end="\n\n")
        remove("daisy.o")


if __name__ == "__main__":
    main()
