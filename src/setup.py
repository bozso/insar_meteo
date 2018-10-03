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

from distutils.ccompiler import new_compiler

def main():
    flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    
    #macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    comp = new_compiler()
    
    comp.set_include_dirs(["/home/istvan/miniconda3/include", "include"])
    comp.add_library_dir("/home/istvan/miniconda3/lib")
    
    utils = join("aux", "utils.cc")
    math_aux = join("aux", "math_aux.cc")
    satorbit = join("aux", "satorbit.cc")
    main = "main_functions.cc"
    inmet = "inmet.cc"
    
    objs = [comp.compile([utils], extra_preargs=flags),
            comp.compile([math_aux], extra_preargs=flags, depends=[utils]),
            comp.compile([satorbit], extra_preargs=flags,
                         depends=[utils, math_aux]),
            comp.compile([main], extra_preargs=flags,
                         depends=[utils, math_aux, satorbit]),
            comp.compile([inmet], extra_preargs=flags,
                         depends=[utils, math_aux, satorbit, main])]
    
    objs = tuple(obj[0] for obj in objs)
    
    comp.link_executable(objs, "inmet", output_dir=join("..", "bin"),
                         extra_preargs=flags,
                         libraries=["stdc++", "m", "gsl", "gslcblas"])

if __name__ == "__main__":
    main()


