from os.path import join
from numpy.distutils.core import Extension, setup


def main():
    #flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    #flags = ["-std=c++03", "-O0", "-save-temps"]
    flags = ["-std=c++11", "-O3"]
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    sources = ["test.cc"]
    
    ext_modules = [
        Extension(name="cext", sources=sources,
                  define_macros=macros,
                  extra_compile_args=flags,
                  libraries=["m"])
    ]
    
    setup(ext_modules=ext_modules)


if __name__ == "__main__":
    main()
