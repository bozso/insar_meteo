from numpy.distutils.core import Extension, setup
# from os.path import join

comp_args = ["-std=c99", "-O3"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

ext_insar = Extension(name="insar_aux", sources=["insar_auxmodule.c"],
                      define_macros=macros, extra_compile_args=comp_args)

setup(ext_modules=[ext_insar])
