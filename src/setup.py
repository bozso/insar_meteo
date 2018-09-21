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

from inmet.compilers import Compiler

def main():
    lib_dirs = ["/home/istvan/miniconda3/lib"]
    inc_dirs = ["/home/istvan/miniconda3/include"]
    #flags = ["-std=c++98", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    flags = ["-std=c++03"]
    #macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    comp = Compiler()
    
    comp.add_obj("utils.cc", flags=flags)
    comp.add_obj("aramdillo.cc", flags=flags, inc_dirs=inc_dirs)
    comp.add_obj("satorbit.cc", flags=flags)
    comp.add_obj("main_functions.cc", flags=flags, inc_dirs=inc_dirs)
    
    comp.make_exe("inmet.cc", flags=flags, outdir=join("..", "bin"),
                  libs=["stdc++", "m", "armadillo"], lib_dirs=lib_dirs)

if __name__ == "__main__":
    main()


