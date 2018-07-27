# Copyright (C) 2018  MTA CSFK GGI
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

from numpy.distutils.core import Extension, setup
from distutils.ccompiler import new_compiler

from os.path import join, isfile
from os import remove
from shutil import move
from glob import iglob
import sys


libs = [":libgfortran.so.3"]
lib_dirs = ["/home/istvan/miniconda3/lib"]
flags = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

ext_modules = [
    Extension(name="insar_aux", sources=["insar_aux.f95"],
              define_macros=macros)
]

def main():
    setup(ext_modules=ext_modules)
    
    for so in iglob("*.so"):
        dst = join("..", "inmet", so)
        
        if isfile(dst):
            remove(dst)
        move(so, dst)

if __name__ == "__main__":
    main()


