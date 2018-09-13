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

from inmet.compilers import compile_project

#lib_dirs = ["/home/istvan/miniconda3/lib"]
#flags = ["-std=c++98", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
flags = ["-std=c++98"]
#macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

def main():
    compile_project("inmet.cpp", "satorbit.cpp", flags=flags,
                    libs=["stdc++", "m"], outdir=join("..", "bin"))

if __name__ == "__main__":
    main()


