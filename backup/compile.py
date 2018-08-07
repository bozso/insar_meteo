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

from os.path import join as pjoin, isfile
from distutils.ccompiler import new_compiler

from compilers import compile_project

libs = ["m", "gsl", "gslcblas"]
flags = ["-std=c99", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
macros = [("HAVE_INLINE", None), ("GSL_RANGE_CHECK_OFF", None)]
inc_dirs = ["/home/istvan/progs/gsl/include"]
lib_dirs = ["/home/istvan/progs/gsl/lib"]

#libs = ["m", "stdc++"]
#flags = ["-std=c++11"]
#inc_dirs = ["/home/istvan/progs/flens"]
#lib_dirs = None
#macros = None

def main():

    compile_project("inmet.c", "matrix.c", "main_functions.c", macros=macros,
                    inc_dirs=inc_dirs, lib_dirs=lib_dirs, flags=flags,
                    libs=libs, outdir=pjoin("..", "bin"))

if __name__ == "__main__":
    main()
