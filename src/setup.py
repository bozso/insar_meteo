# Copyright (C) 2018  István Bozsó
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from os.path import join
#from distutils.ccompiler import new_compiler
from numpy.distutils.core import Extension, setup
from sysconfig import get_config_var

def main():
    miniconda = "/home/istvan/miniconda3"
    #flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    #flags = ["-std=c++03", "-O0", "-save-temps"]

    flags = get_config_var('CFLAGS').split()
    flags += ["-std=c99", "-Wall", "-Wextra"]

    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    inc_dirs = [join(miniconda, "include"), "aux"]
    lib_dirs = [join(miniconda, "lib")]
    
    libs = []
    
    sources = ["inmet_auxmodule.c", "implement.c"]

    ext_modules = [
        Extension(name="inmet_aux", sources=sources,
                  define_macros=macros,
                  extra_compile_args=flags,
                  include_dirs=inc_dirs)
    ]

    setup(ext_modules=ext_modules)
    
    return 0


if __name__ == "__main__":
    main()


